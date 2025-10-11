# src/utils/logging_config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from loguru import logger
import logging, sys, os, pathlib, contextvars

# ======== Context (request_id) ========
_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)

def set_request_id(value: Optional[str]) -> None:
    """Ustaw/wyczyść request_id w bieżącym kontekście."""
    _request_id.set(value)

# ======== Intercept std logging -> loguru ========
class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except Exception:
            level = record.levelno
        # Głębia 6, by wskazać prawdziwe miejsce wywołania
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())

def _install_std_logging_bridge(level: str) -> None:
    root = logging.getLogger()
    root.handlers = [InterceptHandler()]
    root.setLevel(level)
    # Uvicorn / FastAPI
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logging.getLogger(name).handlers = [InterceptHandler()]
        logging.getLogger(name).setLevel(level)

# ======== Konfiguracja z ENV ========
def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes"}

@dataclass(frozen=True)
class LogCfg:
    level: str
    json_console: bool
    file_path: Optional[str]
    rotation: str
    retention: str
    compression: Optional[str]
    enqueue: bool
    backtrace: bool
    diagnose: bool
    service: str
    environment: str

def _read_cfg() -> LogCfg:
    return LogCfg(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        json_console=_env_bool("LOG_JSON", "0"),
        file_path=os.getenv("LOG_FILE", "logs/app.log") or None,  # pusty -> brak pliku
        rotation=os.getenv("LOG_ROTATION", "5 MB"),
        retention=os.getenv("LOG_RETENTION", "7 days"),
        compression=os.getenv("LOG_COMPRESSION", ""),  # np. "zip" | "" = brak
        enqueue=_env_bool("LOG_ENQUEUE", "1"),
        backtrace=_env_bool("LOG_BACKTRACE", "0"),
        diagnose=_env_bool("LOG_DIAGNOSE", "0"),
        service=os.getenv("SERVICE_NAME", "DataGenius"),
        environment=os.getenv("ENVIRONMENT", os.getenv("ENV", "dev")),
    )

# ======== Patcher: automatycznie wstrzykuj request_id ========
def _inject_request_id(record: dict) -> bool:
    record["extra"].setdefault("service", _cfg.service)
    record["extra"].setdefault("environment", _cfg.environment)
    rid = _request_id.get()
    if rid is not None:
        record["extra"]["request_id"] = rid
    return True

# ======== Public API ========
_cfg = _read_cfg()  # odczyt raz; zmień ENV przed importem jeśli potrzeba

def configure_logger() -> "logger.__class__":
    """
    PRO+++ konfiguracja Loguru:
    - konsola (JSON/tekst) + plik (rotacja/retencja/kompresja),
    - intercept std logging/uvicorn,
    - wstrzykiwanie request_id/service/environment.
    Zwraca globalny `logger`.
    """
    # 1) Usuń domyślne
    logger.remove()

    # 2) Console sink
    console_fmt_text = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "rid={extra[request_id]} | svc={extra[service]} env={extra[environment]} | "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        level=_cfg.level,
        serialize=_cfg.json_console,         # JSON gdy True
        backtrace=_cfg.backtrace,
        diagnose=_cfg.diagnose,
        enqueue=_cfg.enqueue,
        format=None if _cfg.json_console else console_fmt_text,
        filter=_inject_request_id,
    )

    # 3) File sink (opcjonalny)
    if _cfg.file_path:
        path = pathlib.Path(_cfg.file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(path),
            level="DEBUG",
            rotation=_cfg.rotation,
            retention=_cfg.retention,
            compression=(_cfg.compression or None),
            enqueue=_cfg.enqueue,
            backtrace=_cfg.backtrace,
            diagnose=_cfg.diagnose,
            serialize=False,                  # plik czytelny (możesz zmienić na True)
            filter=_inject_request_id,
        )

    # 4) Przechwyć std logging/uvicorn
    _install_std_logging_bridge(_cfg.level)

    # 5) Sentry (opcjonalnie)
    try:
        import sentry_sdk  # type: ignore
        dsn = os.getenv("SENTRY_DSN", "")
        if dsn:
            from sentry_sdk.integrations.logging import LoggingIntegration  # type: ignore
            sentry_sdk.init(
                dsn=dsn,
                traces_sample_rate=float(os.getenv("SENTRY_TRACES", "0.0")),
                integrations=[LoggingIntegration(level=logging.ERROR, event_level=logging.ERROR)],
                environment=_cfg.environment,
            )
            logger.bind(sentry=True).info("Sentry logging enabled")
    except Exception:
        # brak sentry lub błąd — pomijamy cicho
        pass

    # 6) Zbinduj stałe pola
    base = logger.bind(service=_cfg.service, environment=_cfg.environment, request_id=None)
    return base

# Helper do pobrania nazwanej instancji
def get_logger(name: str = "app"):
    """Zwraca logger z podbindowaną nazwą modułu."""
    return logger.bind(mod=name)
