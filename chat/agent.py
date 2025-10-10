from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os, time, math, logging, re
from dataclasses import dataclass

# Optional token counter (lepsze przycinanie historii)
try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None  # pragma: no cover

# OpenAI SDK (oficjalny)
try:
    from openai import OpenAI  # pip install openai
except Exception as _e:  # pragma: no cover
    OpenAI = None  # type: ignore

import pandas as pd

# === NAZWA_SEKCJI === KONFIG / TRYBY
SYSTEM_MODES: Dict[str, str] = {
    "Data Assistant": "You are a helpful data analyst focusing on data quality and EDA.",
    "Modeling Coach": "You are a senior ML engineer optimizing models and thresholds.",
    "Forecasting Guru": "You are a time-series expert focused on trends and seasonality.",
    # Dodatkowe, bez zmian zachowania poprzednich:
    "MLOps Partner": "You are an MLOps engineer focused on reproducibility, monitoring and deployment.",
    "Feature Wizard": "You design and critique feature engineering strategies."
}

log = logging.getLogger("chat_engine")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# === NAZWA_SEKCJI === NARZĘDZIA (kontekst, przycinanie)
def _model_name() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def _use_responses_api() -> bool:
    # Responses API daje szersze możliwości (structured outputs, narzędzia).
    return os.getenv("OPENAI_USE_RESPONSES", "0").strip() in {"1", "true", "TRUE", "yes"}

def _max_output_tokens() -> int:
    try:
        return int(os.getenv("OPENAI_MAX_TOKENS", "600"))
    except Exception:
        return 600

def _temperature() -> float:
    try:
        return float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    except Exception:
        return 0.2

def _count_tokens(messages: List[Dict[str, str]], model: str) -> int:
    if tiktoken is None:
        # aproksymacja: ~4 znaki ~ 1 token
        total_chars = sum(len(m.get("content", "")) for m in messages) + sum(len(m.get("role","")) for m in messages)
        return max(1, total_chars // 4)
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    toks = 0
    for m in messages:
        toks += 4  # priorytety + nagłówki
        toks += len(enc.encode(m.get("content","")))
        toks += len(enc.encode(m.get("role","")))
    toks += 2  # assistant priming
    return toks

def _truncate_messages(messages: List[Dict[str,str]], model: str, max_input_tokens: int = 6000) -> List[Dict[str,str]]:
    """
    Zostaw system + ostatnie n komunikatów tak, aby wejść <= max_input_tokens.
    """
    if not messages:
        return messages
    # system message (jeśli jest) zostawiamy na początku
    sys_msgs = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]

    out = sys_msgs + rest[-12:]  # szybkie ograniczenie okna
    # token tighten
    while _count_tokens(out, model) > max_input_tokens and len(rest) > 1:
        rest = rest[1:]
        out = sys_msgs + rest
    return out

# === NAZWA_SEKCJI === OFFLINE NARZĘDZIA (gdy brak klucza lub błąd)
def _offline_reply(last: str, mode: str, df: Optional[pd.DataFrame]) -> str:
    """
    Prosty 'tooling' offline: schema/head/describe/nulls/cols/value_counts col=...
    """
    if df is None:
        return f"[{mode}] Offline. Dostępne: schema | head [n] | describe | nulls | cols | value_counts col=<nazwa> | sample [n]."

    last_l = (last or "").strip().lower()

    def _int_in(text: str, default: int) -> int:
        m = re.search(r"\b(\d{1,4})\b", text)
        return int(m.group(1)) if m else default

    if "schema" in last_l:
        return f"Schema: shape={df.shape}, cols={len(df.columns)}\nColumns: {', '.join(map(str, df.columns[:30]))}{' ...' if df.shape[1]>30 else ''}"
    if "head" in last_l:
        n = _int_in(last_l, 5)
        return df.head(n).to_markdown(index=False)
    if "describe" in last_l:
        return df.describe(include="all").T.head(25).to_markdown()
    if "nulls" in last_l or "missing" in last_l:
        miss = df.isna().sum().sort_values(ascending=False)
        top = miss[miss>0].head(30)
        return "Braki danych:\n" + (top.to_frame("missing").to_markdown() if not top.empty else "brak")
    if "cols" in last_l or "columns" in last_l:
        return "\n".join([f"- {c} ({str(t)})" for c,t in df.dtypes.items()])
    if "value_counts" in last_l and "col=" in last_l:
        m = re.search(r"col=([A-Za-z0-9_\-\. ]+)", last, re.I)
        if m and m.group(1) in df.columns:
            vc = df[m.group(1)].value_counts(dropna=False).head(30)
            return vc.to_frame("count").to_markdown()
        return "Podaj poprawny 'col=<nazwa kolumny>'."
    if "sample" in last_l:
        n = _int_in(last_l, 5)
        return df.sample(min(n, len(df)), random_state=42).to_markdown(index=False)
    # fallback
    return f"[{mode}] Offline. Dostępne: schema | head [n] | describe | nulls | cols | value_counts col=<nazwa> | sample [n]."

# === NAZWA_SEKCJI === CALL OPENAI
def _call_openai(messages: List[Dict[str,str]], *, sys_prompt: str, timeout: float = 30.0) -> str:
    """
    Wywołanie OpenAI. Preferuj Responses API gdy włączone, z fallbackiem do Chat Completions.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        last = messages[-1]["content"] if messages else ""
        return _offline_reply(last, "Data Assistant", None)

    client = OpenAI(api_key=api_key)
    model = _model_name()
    temp = _temperature()
    max_toks = _max_output_tokens()

    # Wstaw system na początek – zachowujemy user history
    msgs = [{"role": "system", "content": sys_prompt}] + messages

    # Przycinamy wejście (konserwatywnie)
    msgs = _truncate_messages(msgs, model=model, max_input_tokens=int(os.getenv("OPENAI_MAX_INPUT_TOKENS", "6000")))

    # Retry z backoffem
    attempts = int(os.getenv("OPENAI_RETRIES", "3"))
    for i in range(attempts):
        try:
            if _use_responses_api():
                # Responses API (zalecane przez OpenAI dla nowych funkcji)
                # Uwaga: jeśli SDK/endpoint różni się wersją – fallback niżej zadziała.
                resp = client.responses.create(
                    model=model,
                    input=msgs,                     # SDK akceptuje listę messages jako input
                    temperature=temp,
                    max_output_tokens=max_toks,
                )
                # Helper (w nowszym SDK): resp.output_text
                content = getattr(resp, "output_text", None)
                if not content:
                    # bez helpera – wyciągnij tekst z pierwszego bloku
                    try:
                        content = resp.output[0].content[0].text
                    except Exception:
                        content = None
                if content:
                    return str(content).strip()
                # jeśli brak – spróbuj jednak Chat
                log.debug("Responses API yielded empty content; falling back to Chat.")
            # Chat Completions (stabilny punkt odniesienia)
            chat = client.chat.completions.create(
                model=model,
                messages=msgs,
                temperature=temp,
                max_tokens=max_toks,
            )
            return (chat.choices[0].message.content or "").strip()
        except Exception as e:
            # ostatnia próba → rzuć dalej do offline fallback
            delay = 0.5 * (2 ** i)
            log.warning(f"OpenAI call failed (attempt {i+1}/{attempts}): {e}")
            time.sleep(min(delay, 4.0))
    # Offline fallback (zabezpieczenie)
    last = messages[-1]["content"] if messages else ""
    return _offline_reply(last, "Data Assistant", None)

# === NAZWA_SEKCJI === PUBLIC API (kompatybilne z Twoim podpisem)
def chat_reply(messages: List[Dict[str, str]], mode: str, df: Optional[pd.DataFrame]) -> str:
    """
    Drop-in replacement:
    - gdy brak OPENAI_API_KEY → tryb offline z EDA narzędziami (schema/head/describe/...).
    - gdy klucz jest → wywołanie OpenAI (Responses API jeśli włączony env, fallback Chat Completions).
    - sys prompt wybierany po `mode` z SYSTEM_MODES.
    """
    # tryb offline?
    key = os.getenv("OPENAI_API_KEY", "").strip()
    last = messages[-1]["content"] if messages else ""

    sys_prompt = SYSTEM_MODES.get(mode, SYSTEM_MODES["Data Assistant"])

    if not key:
        return _offline_reply(last, mode, df)

    # Tryb on-line: jednocześnie wstrzykujemy mini-kontekst o schemacie, jeśli user prosi
    if df is not None and last and "schema" in last.lower():
        schema_note = f"(schema) shape={df.shape}, cols={len(df.columns)}: {', '.join(map(str, df.columns[:25]))}{' ...' if df.shape[1]>25 else ''}"
        messages = messages + [{"role": "system", "content": f"Dataset context: {schema_note}"}]

    return _call_openai(messages, sys_prompt=sys_prompt)
