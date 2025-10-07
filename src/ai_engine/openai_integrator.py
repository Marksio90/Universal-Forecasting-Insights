from __future__ import annotations
import os
import time
import json
from typing import Optional
from openai import OpenAI

# Optional: Streamlit secrets (jeśli używasz w appce)
try:
    import streamlit as st
except ImportError:
    st = None

# ---------------------------
# Klient OpenAI (z cache)
# ---------------------------
_client_cache: Optional[OpenAI] = None

def get_client() -> Optional[OpenAI]:
    """
    Tworzy instancję OpenAI z kluczem API:
    - st.secrets["openai"]["api_key"]
    - os.environ["OPENAI_API_KEY"]
    - .env (jeśli załadowany przez dotenv)
    """
    global _client_cache
    if _client_cache is not None:
        return _client_cache

    key = None
    # 1️⃣ Streamlit secrets
    if st is not None:
        try:
            key = st.secrets.get("openai", {}).get("api_key") or st.secrets.get("OPENAI_API_KEY")
        except Exception:
            pass
    # 2️⃣ Env
    if not key:
        key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None

    try:
        _client_cache = OpenAI(api_key=key)
        return _client_cache
    except Exception:
        return None

# ---------------------------
# Chat completion helper
# ---------------------------
def chat_completion(
    system: str,
    user: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 900,
    response_format: str = "text",  # "text" | "json"
    retries: int = 2,
    timeout: int = 40,
) -> str:
    """
    Wywołuje OpenAI ChatCompletion z retry i obsługą JSON outputu.

    - system: kontekst roli systemu
    - user: prompt użytkownika
    - response_format: "text" (domyślnie) lub "json"
    """
    client = get_client()
    if client is None:
        return "Brak klucza OpenAI. Ustaw OPENAI_API_KEY w .env, os.environ lub st.secrets."

    # Model fallback
    available_model = model
    if model not in ("gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-3.5-turbo"):
        available_model = "gpt-4o-mini"

    # Ustal format odpowiedzi
    resp_format = {"type": response_format} if response_format == "json" else None

    last_error = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=available_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                response_format=resp_format,
            )
            out = resp.choices[0].message.content
            return out.strip() if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
        except Exception as e:
            last_error = str(e)
            # prosty backoff
            time.sleep(1.5 * (attempt + 1))
            continue

    return f"Błąd OpenAI: {last_error or 'Nieznany błąd'}"
