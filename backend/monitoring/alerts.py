from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable, Dict, Tuple, List
import os, time, ssl, smtplib, hashlib, json
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Opcjonalny loguru; jeśli brak – używamy std logging w stylu „no-op”.
try:
    from loguru import logger
except Exception:  # pragma: no cover
    class _L:
        def info(self, *a, **k): ...
        def warning(self, *a, **k): ...
        def error(self, *a, **k): ...
        def debug(self, *a, **k): ...
    logger = _L()  # type: ignore


# === KONFIG ===

@dataclass(frozen=True)
class AlertConfig:
    # Slack
    slack_webhook_url: Optional[str] = os.getenv("SLACK_WEBHOOK_URL") or None
    slack_timeout_s: float = float(os.getenv("SLACK_TIMEOUT_S", "5"))
    slack_retries: int = int(os.getenv("SLACK_RETRIES", "3"))
    slack_backoff_s: float = float(os.getenv("SLACK_BACKOFF_S", "0.5"))

    # SMTP / Email
    smtp_host: Optional[str] = os.getenv("SMTP_HOST") or None
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: Optional[str] = os.getenv("SMTP_USER") or None
    smtp_password: Optional[str] = os.getenv("SMTP_PASSWORD") or None
    smtp_from: Optional[str] = os.getenv("SMTP_FROM") or os.getenv("SMTP_USER") or None
    smtp_use_ssl: bool = os.getenv("SMTP_USE_SSL", "false").lower() in {"1", "true", "yes"}
    smtp_starttls: bool = os.getenv("SMTP_STARTTLS", "true").lower() in {"1", "true", "yes"}

    # Dedup / Rate-limit (global per-process)
    dedup_window_s: int = int(os.getenv("ALERTS_DEDUP_WINDOW_S", "300"))  # 5 min

CFG = AlertConfig()

# Pamięć deduplikacji: {signature: ts_last_sent}
_DEDUP_CACHE: Dict[str, float] = {}


def _signature(*parts: str) -> str:
    """Stabilna sygnatura wiadomości do dedup (hash z ważnych pól)."""
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
    return h.hexdigest()


def _should_suppress(sig: str, window_s: int) -> bool:
    """Czy powinniśmy pominąć (duplikat w oknie czasu)?"""
    now = time.time()
    ts = _DEDUP_CACHE.get(sig)
    if ts is not None and (now - ts) < window_s:
        return True
    _DEDUP_CACHE[sig] = now
    return False


def _retry_loop(attempts: int, backoff_s: float):
    """Generator opóźnień po niepowodzeniu."""
    for i in range(attempts):
        yield i
        if i < attempts - 1:
            time.sleep(backoff_s * (2 ** i))


# === SLACK ===

def send_slack(text: str, *, title: Optional[str] = None, dedup: bool = True) -> bool:
    """
    Wysyła prostą wiadomość do Slacka (Incoming Webhook).
    Zwraca True/False – kompatybilne z Twoim wcześniejszym API.
    """
    url = CFG.slack_webhook_url
    if not url:
        logger.debug("send_slack: brak SLACK_WEBHOOK_URL – pomijam")
        return False

    payload = {"text": f"*{title}*\n{text}" if title else text}
    sig = _signature("SLACK", url, payload["text"])
    if dedup and _should_suppress(sig, CFG.dedup_window_s):
        logger.info("send_slack: suppressed duplicate within dedup window")
        return True  # traktujemy jako sukces, żeby nie eskalować

    for _ in _retry_loop(CFG.slack_retries, CFG.slack_backoff_s):
        try:
            r = requests.post(url, json=payload, timeout=CFG.slack_timeout_s)
            if 200 <= r.status_code < 300:
                return True
            logger.warning(f"send_slack: HTTP {r.status_code} – {r.text[:200]}")
        except Exception as e:  # pragma: no cover
            logger.warning(f"send_slack: exception {e}")
    return False


# === EMAIL (TXT/HTML + ZAŁĄCZNIKI) ===

def _smtp_connect():
    """Zwraca połączenie SMTP (SSL lub STARTTLS)."""
    if not CFG.smtp_host or not CFG.smtp_user or not CFG.smtp_password:
        raise RuntimeError("SMTP config incomplete (SMTP_HOST/USER/PASSWORD)")

    if CFG.smtp_use_ssl:
        context = ssl.create_default_context()
        server = smtplib.SMTP_SSL(CFG.smtp_host, CFG.smtp_port, context=context, timeout=15)
        server.login(CFG.smtp_user, CFG.smtp_password)
        return server

    server = smtplib.SMTP(CFG.smtp_host, CFG.smtp_port, timeout=15)
    if CFG.smtp_starttls:
        context = ssl.create_default_context()
        server.starttls(context=context)
    server.login(CFG.smtp_user, CFG.smtp_password)
    return server


def _normalize_recipients(to: str | Iterable[str]) -> List[str]:
    if isinstance(to, str):
        # Rozdziel po przecinkach i spacjach
        parts = [x.strip() for x in to.replace(";", ",").split(",") if x.strip()]
        return parts
    return list(to)


def send_email(subject: str, body: str, to: str | Iterable[str]) -> bool:
    """
    Wysyła email w formacie TEXT.
    Zwraca True/False — kompatybilne z Twoim wcześniejszym API.
    """
    try:
        tos = _normalize_recipients(to)
        if not tos:
            logger.warning("send_email: brak odbiorców")
            return False

        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = CFG.smtp_from or (CFG.smtp_user or "")
        msg["To"] = ", ".join(tos)

        sig = _signature("EMAIL", msg["From"], msg["To"], subject, body[:200])
        if _should_suppress(sig, CFG.dedup_window_s):
            logger.info("send_email: suppressed duplicate within dedup window")
            return True

        for _ in _retry_loop(3, 0.5):
            try:
                with _smtp_connect() as s:
                    s.sendmail(msg["From"], tos, msg.as_string())
                return True
            except Exception as e:  # pragma: no cover
                logger.warning(f"send_email: attempt failed: {e}")
        return False
    except Exception as e:  # pragma: no cover
        logger.error(f"send_email: fatal error: {e}")
        return False


def send_email_html(
    subject: str,
    html_body: str,
    to: str | Iterable[str],
    *,
    attachments: Optional[List[str]] = None,
    cc: Optional[Iterable[str]] = None,
    bcc: Optional[Iterable[str]] = None,
) -> bool:
    """
    Wysyła email HTML + opcjonalne załączniki (ścieżki plików).
    """
    try:
        tos = _normalize_recipients(to)
        ccs = _normalize_recipients(cc) if cc else []
        bccs = _normalize_recipients(bcc) if bcc else []
        all_rcpts = list(dict.fromkeys(tos + ccs + bccs))  # unique, zachowaj kolejność
        if not all_rcpts:
            logger.warning("send_email_html: brak odbiorców")
            return False

        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = CFG.smtp_from or (CFG.smtp_user or "")
        msg["To"] = ", ".join(tos)
        if ccs:
            msg["Cc"] = ", ".join(ccs)

        msg.attach(MIMEText(html_body, "html", "utf-8"))

        for path in attachments or []:
            try:
                with open(path, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(path)}"')
                msg.attach(part)
            except Exception as e:  # pragma: no cover
                logger.warning(f"send_email_html: cannot attach {path}: {e}")

        sig = _signature("EMAIL_HTML", msg["From"], ",".join(all_rcpts), subject, hashlib.sha256(html_body.encode()).hexdigest())
        if _should_suppress(sig, CFG.dedup_window_s):
            logger.info("send_email_html: suppressed duplicate within dedup window")
            return True

        for _ in _retry_loop(3, 0.5):
            try:
                with _smtp_connect() as s:
                    s.sendmail(msg["From"], all_rcpts, msg.as_string())
                return True
            except Exception as e:  # pragma: no cover
                logger.warning(f"send_email_html: attempt failed: {e}")
        return False
    except Exception as e:  # pragma: no cover
        logger.error(f"send_email_html: fatal error: {e}")
        return False


# === DISPATCH & HEALTH ===

def dispatch_alert(
    subject: str,
    message: str,
    *,
    channels: Iterable[str] = ("slack", "email"),
    email_to: Optional[str | Iterable[str]] = None,
    html: bool = False,
) -> Dict[str, bool]:
    """
    Wysyła alert jednocześnie kilkoma kanałami.
    Przykład:
        dispatch_alert("PSI ALERT", "PSI=0.31", channels=("slack","email"), email_to="ops@example.com")
    """
    results: Dict[str, bool] = {}
    if "slack" in channels:
        results["slack"] = send_slack(message, title=subject)
    if "email" in channels:
        to = email_to or os.getenv("ALERTS_EMAIL_TO", "")
        if html:
            results["email"] = send_email_html(subject, message, to=to)
        else:
            results["email"] = send_email(subject, message, to=to)
    return results


def health_check() -> Dict[str, str | bool]:
    """
    Szybkie sprawdzenie konfiguracji (bez łączenia się na zewnątrz).
    """
    return {
        "slack_configured": bool(CFG.slack_webhook_url),
        "smtp_configured": bool(CFG.smtp_host and CFG.smtp_user and CFG.smtp_password),
        "smtp_ssl": CFG.smtp_use_ssl,
        "smtp_starttls": CFG.smtp_starttls,
        "dedup_window_s": CFG.dedup_window_s,
    }
