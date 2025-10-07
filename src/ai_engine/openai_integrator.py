# openai_integrator.py — PRO++
from __future__ import annotations

import os
import re
import json
import time
import math
import random
import threading
from typing import Optional, Callable, Any, Dict, List, Literal, Union

try:
    import streamlit as st  # opcjonalnie
except Exception:
    st = None  # type: ignore

try:
    from openai import OpenAI
    from openai import APIError, RateLimitError, APITimeoutError, APIConnectionError, BadRequestError, AuthenticationError
except Exception:
    # Minimalny fallback typów wyjątków, gdyby SDK nie było jeszcze zainstalowane
    OpenAI = object  # type: ignore
    class APIError(Exception): ...
    class RateLimitError(APIError): ...
    class APITimeoutError(APIError): ...
    class APIConnectionError(APIError): ...
    class BadRequestError(APIError): ...
    class AuthenticationError(APIError): ...

# =========================
# Konfiguracja
# =========================
_DEFAULT_MODEL = "gpt-4o-mini"
_MODEL_ALIASES = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
    "gpt-4.1": "gpt-4.1",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
}

_CLIENT_CACHE: Optional[OpenAI] = None
_CLIENT_LOCK = threading.Lock()

# =========================
# Utils
# =========================
def _get_secret(key: str) -> Optional[str]:
    if st is not None:
        try:
            # pozwól na oba warianty: sekcja openai.api_key i bezpośrednio OPENAI_API_KEY
            if "openai" in st.secrets and key in st.secrets["openai"]:
                return st.secrets["openai"][key]
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
    return os.getenv(key)

def has_openai_key() -> bool:
    return bool(_get_secret("api_key") or _get_secret("OPENAI_API_KEY"))

def reset_client() -> None:
    global _CLIENT_CACHE
    with _CLIENT_LOCK:
        _CLIENT_CACHE = None

def _build_client(
    api_key: Optional[str],
    base_url: Optional[str],
    organization: Optional[str],
    project: Optional[str],
) -> Optional[OpenAI]:
    if not api_key:
        return None
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url  # wspiera np. proxy/self-host/azure-compatible
    if organization:
        kwargs["organization"] = organization
    if project:
        kwargs["project"] = project
    try:
        return OpenAI(**kwargs)  # type: ignore
    except Exception:
        return None

def get_client(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    project: Optional[str] = None,
) -> Optional[OpenAI]:
    """
    Zwraca singleton klienta OpenAI (cache).
    Priorytety:
    1) argumenty funkcji
    2) Streamlit secrets (openai.api_key / OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_ORG, OPENAI_PROJECT)
    3) Zmienne środowiskowe
    """
    global _CLIENT_CACHE
    with _CLIENT_LOCK:
        if _CLIENT_CACHE is not None:
            return _CLIENT_CACHE

        key = api_key or _get_secret("api_key") or _get_secret("OPENAI_API_KEY")
        base = base_url or _get_secret("OPENAI_BASE_URL")
        org = organization or _get_secret("OPENAI_ORG")
        proj = project or _get_secret("OPENAI_PROJECT")

        client = _build_client(key, base, org, proj)
        _CLIENT_CACHE = client
        return client

def _choose_model(model: str) -> str:
    return _MODEL_ALIASES.get(model, _DEFAULT_MODEL)

def _response_format_for(flag: str | None) -> Optional[Dict[str, str]]:
    if flag == "json":
        # nowe SDK wspiera {"type":"json_object"} do wymuszenia JSON
        return {"type": "json_object"}
    return None

def _is_transient_error(e: Exception) -> bool:
    return isinstance(e, (RateLimitError, APITimeoutError, APIConnectionError, APIError))

def _backoff_sleep(attempt: int, base: float = 0.8, cap: float = 8.0) -> None:
    # Exponential backoff z jitterem
    wait = min(cap, base * (2 ** attempt)) * (0.5 + random.random())
    time.sleep(wait)

def _strip_json_noise(text: str) -> str:
    """Usuwa ewentualny prefix/sufiks spoza JSON i wycina pierwszy „pełny” obiekt/array JSON."""
    if not isinstance(text, str):
        return text
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    return m.group(1) if m else text

def _safe_json_parse(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        text2 = _strip_json_noise(text)
        try:
            return json.loads(text2)
        except Exception:
            return None

# =========================
# Public API
# =========================
def chat_completion(
    system: str,
    user: str,
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 900,
    response_format: str = "text",  # "text" | "json"
    retries: int = 2,
    timeout: int = 40,
    *,
    stream: bool = False,
    on_token: Optional[Callable[[str], None]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    project: Optional[str] = None,
) -> str:
    """
    Wywołuje OpenAI Chat Completions z retry, backoffem i wymuszeniem JSON (jeśli `response_format="json"`).
    Zwraca string (dla "json" również string — surowy JSON). Do wygodnego dict: użyj `chat_completion_json`.

    Parametry dodatkowe:
    - stream: gdy True, zwraca zbuforowaną całość, ale emituje tokeny przez `on_token`.
    - on_token: callback(token_str) wywoływany przy streamingu.
    - api_key/base_url/organization/project: nadpisują wartości z secrets/env.
    """
    client = get_client(api_key=api_key, base_url=base_url, organization=organization, project=project)
    if client is None:
        return "Brak klucza OpenAI. Ustaw OPENAI_API_KEY (lub st.secrets)."

    chosen_model = _choose_model(model)
    resp_fmt = _response_format_for(response_format if response_format in ("text", "json") else "text")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    last_error: Optional[str] = None
    for attempt in range(retries + 1):
        try:
            if stream:
                # streaming
                acc: List[str] = []
                with client.chat.completions.stream(  # type: ignore
                    model=chosen_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=resp_fmt,
                    timeout=timeout,
                ) as s:
                    for event in s:
                        if event.type == "token":
                            tok = event.token
                            if isinstance(tok, str):
                                acc.append(tok)
                                if on_token:
                                    try:
                                        on_token(tok)
                                    except Exception:
                                        pass
                        elif event.type == "completed":
                            break
                return "".join(acc).strip()

            # non-stream
            resp = client.chat.completions.create(  # type: ignore
                model=chosen_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=resp_fmt,
                timeout=timeout,
            )
            out = resp.choices[0].message.content  # type: ignore
            if isinstance(out, str):
                return out.strip()
            # niektóre SDK mogą zwrócić obiekt; zserializuj
            return json.dumps(out, ensure_ascii=False)
        except AuthenticationError as e:
            return "Błąd uwierzytelnienia OpenAI (sprawdź API key / uprawnienia)."
        except BadRequestError as e:
            # tu często wpada błąd schematu response_format vs. model
            last_error = str(e)
            if "response_format" in last_error and response_format == "json":
                # spróbuj bez wymuszania json_object (czasem starsze modele)
                resp_fmt = None
                continue
        except Exception as e:
            last_error = str(e)
            if _is_transient_error(e) and attempt < retries:
                _backoff_sleep(attempt)
                continue
            break

    # po wyczerpaniu prób
    return f"Błąd OpenAI: {last_error or 'Nieznany błąd'}"

def chat_completion_json(
    system: str,
    user: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Wymusza JSON i zwraca `dict`. Gdy się nie uda — zwraca minimalny obiekt z `raw_text`.
    """
    text = chat_completion(system=system, user=user, response_format="json", **kwargs)
    parsed = _safe_json_parse(text)
    if isinstance(parsed, dict):
        return parsed
    return {"error": "Nie udało się sparsować JSON.", "raw_text": text}
