# src/utils/secrets.py
from __future__ import annotations
# === KONTEKST ===
# PRO+++ zarządzanie sekretami: st.secrets → ENV (fallback), zagnieżdżone ścieżki "a.b.c",
# casty (bool/int/float/json), .env (opcjonalnie), bezpieczne mapowanie do ENV dla narzędzi.
# Stabilny kontrakt: get_secret(path, default) i prime_env_from_secrets() jak w Twojej wersji.

from typing import Any, Optional, Mapping, MutableMapping, Callable, Dict, List
from functools import lru_cache
import os, json, pathlib

# --- Streamlit (opcjonalny) ---
try:
    import streamlit as st  # type: ignore
    _SECRETS: Mapping[str, Any] = getattr(st, "secrets", {})  # MappingProxy
    _HAS_ST = True
except Exception:
    _SECRETS = {}
    _HAS_ST = False

# --- .env (opcjonalne, bez twardej zależności) ---
try:
    from dotenv import load_dotenv  # type: ignore
    _HAS_DOTENV = True
except Exception:
    _HAS_DOTENV = False


# === NAZWA_SEKCJI === HELPERY RDZENIOWE ===
def _deep_get(mapping: Mapping[str, Any], path: str) -> Optional[Any]:
    """
    Bezpieczne pobranie zagnieżdżonego klucza z mapy (np. "auth.JWT_SECRET" lub "a.b.c").
    Obsługuje tylko kropki jako separator.
    """
    if not path or not isinstance(mapping, Mapping):
        return None
    node: Any = mapping
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node

def _env_candidates(path: str) -> List[str]:
    """
    Kandydaci na nazwy ENV dla ścieżki "a.b": ["b".upper(), "A_B", "a.b"].
    Zachowujemy wsteczną kompatybilność: najpierw KEY (ostatni segment upper).
    """
    if "." in path:
        section, key = path.split(".", 1)
        return [key.upper(), f"{section}_{key}".upper(), path]
    return [path, path.upper()]

def _maybe_strip(v: Any, strip: bool) -> Any:
    if strip and isinstance(v, str):
        return v.strip()
    return v

def _coerce_bool(val: Any) -> Optional[bool]:
    if isinstance(val, bool):
        return val
    if not isinstance(val, str):
        return None
    t = val.strip().lower()
    if t in {"1","true","yes","y","on"}: return True
    if t in {"0","false","no","n","off"}: return False
    return None

def _cast_value(val: Any, cast: Optional[Callable[[Any], Any]], parse_json: bool) -> Any:
    if val is None:
        return None
    if parse_json:
        # przyjmij str lub już zdekodowane
        if isinstance(val, str):
            try:
                return json.loads(val)
            except Exception:
                # pozwól spaść do cast
                pass
    if cast:
        try:
            return cast(val)
        except Exception:
            # zachowaj oryginał, jeśli cast nieudany
            return val
    return val


# === NAZWA_SEKCJI === ŁADOWANIE .ENV (OPCJONALNE) ===
def _ensure_dotenv_loaded() -> None:
    """Załaduj .env tylko raz, jeśli biblioteka dostępna i plik istnieje."""
    if not _HAS_DOTENV:
        return
    flag = "_DG_DOTENV_LOADED"
    if os.environ.get(flag) == "1":
        return
    # szukaj w cwd oraz w katalogu nadrzędnym (częsty układ repo)
    for p in (pathlib.Path(".env"), pathlib.Path("..")/".env"):
        try:
            if p.exists():
                load_dotenv(dotenv_path=str(p), override=False)
                break
        except Exception:
            pass
    os.environ[flag] = "1"


# === NAZWA_SEKCJI === PUBLICZNE API: get_secret* ===
@lru_cache(maxsize=512)
def _get_secret_cached(path: str, default: Optional[Any], cast_id: str, parse_json: bool, strip: bool) -> Optional[Any]:
    """
    Wewnętrzny cache-owany accessor. cast_id służy do różnicowania wpisu cache dla różnych 'cast'.
    """
    # 1) Streamlit secrets (preferowane)
    val = _deep_get(_SECRETS, path) if _HAS_ST else None
    if val in (None, ""):
        # 2) .env/ENV
        _ensure_dotenv_loaded()
        for key in _env_candidates(path):
            env_v = os.getenv(key, None)
            if env_v not in (None, ""):
                val = env_v
                break
    # 3) default, jeśli nadal brak
    if val in (None, ""):
        val = default
    # 4) strip + cast/json
    val = _maybe_strip(val, strip)
    # Dekoduj JSON przy cast_id==":json" (dla get_json)
    if cast_id == ":json":
        try:
            return json.loads(val) if isinstance(val, str) else val
        except Exception:
            return default
    # Uniwersalny cast (mapa nazw → funkcji)
    _CASTS: Dict[str, Callable[[Any], Any]] = {
        ":str": str,
        ":int": lambda x: int(x) if x != "" else default,
        ":float": lambda x: float(x) if x != "" else default,
        ":bool": lambda x: _coerce_bool(x) if not isinstance(x, bool) else x,
        "": lambda x: x,
    }
    caster = _CASTS.get(cast_id, _CASTS[""])
    try:
        out = caster(val)
        # bool caster może zwrócić None (niejednoznaczny tekst) → użyj default
        if out is None:
            return default
        return out
    except Exception:
        return default

def clear_secrets_cache() -> None:
    """Czyści pamięć podręczną sekretów (użyteczne w testach)."""
    _get_secret_cached.cache_clear()  # type: ignore[attr-defined]

def get_secret(path: str, default: Optional[Any] = None, *, strip: bool = True) -> Optional[Any]:
    """
    Publiczny accessor – w pełni kompatybilny z Twoim API.
    Przykłady: "OPENAI_API_KEY", "mlflow.TRACKING_URI", "auth.JWT_SECRET".
    Preferuje st.secrets, następnie ENV/.env, inaczej zwraca default.
    """
    return _get_secret_cached(path, default, "", False, strip)

# Wygodne warianty typowane:
def get_secret_str(path: str, default: Optional[str] = None, *, strip: bool = True) -> Optional[str]:
    return _get_secret_cached(path, default, ":str", False, strip)  # type: ignore[return-value]

def get_secret_int(path: str, default: Optional[int] = None) -> Optional[int]:
    return _get_secret_cached(path, default, ":int", False, True)  # type: ignore[return-value]

def get_secret_float(path: str, default: Optional[float] = None) -> Optional[float]:
    return _get_secret_cached(path, default, ":float", False, True)  # type: ignore[return-value]

def get_secret_bool(path: str, default: Optional[bool] = None) -> Optional[bool]:
    return _get_secret_cached(path, default, ":bool", False, True)  # type: ignore[return-value]

def get_secret_json(path: str, default: Optional[Any] = None) -> Optional[Any]:
    """Dekoduje JSON z sekretu (string) → struktura Pythona."""
    return _get_secret_cached(path, default, ":json", True, True)


# === NAZWA_SEKCJI === MAPOWANIE → ENV (prime_env_from_secrets) ===
# Zachowujemy Twój mapping i dodajemy elastyczność: można rozszerzyć przez ENV JSON.
_DEFAULT_EXPORT_MAP: Dict[str, str] = {
    # LLM
    "OPENAI_API_KEY": "OPENAI_API_KEY",
    # JWT / RBAC
    "auth.JWT_SECRET": "JWT_SECRET",
    "auth.JWT_EXPIRE_MIN": "JWT_EXPIRE_MIN",
    # MLflow
    "mlflow.TRACKING_URI": "MLFLOW_TRACKING_URI",
    # MinIO / S3
    "minio.ENDPOINT_URL": "MLFLOW_S3_ENDPOINT_URL",
    "minio.AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
    "minio.AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
    # Redis
    "redis.URL": "REDIS_URL",
    # Postgres (MLflow backend)
    "db.POSTGRES_USER": "POSTGRES_USER",
    "db.POSTGRES_PASSWORD": "POSTGRES_PASSWORD",
    "db.POSTGRES_DB": "POSTGRES_DB",
    # Alerts
    "alerts.SLACK_WEBHOOK_URL": "SLACK_WEBHOOK_URL",
    "smtp.HOST": "SMTP_HOST",
    "smtp.PORT": "SMTP_PORT",
    "smtp.USER": "SMTP_USER",
    "smtp.PASSWORD": "SMTP_PASSWORD",
    # Domeny/public
    "domain.PUBLIC_BASE_URL": "PUBLIC_BASE_URL",
}

def _load_extra_map_from_env() -> Dict[str, str]:
    """
    Pozwala rozszerzyć mapowanie przez ENV:
      DATAGENIUS_SECRETS_MAP='{"foo.bar":"FOO_BAR"}'
    """
    raw = os.getenv("DATAGENIUS_SECRETS_MAP", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            # filtruj tylko pary str->str
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}

def prime_env_from_secrets() -> None:
    """
    Wypchnij istotne wartości do os.environ (tylko jeśli nie istnieją).
    Użyteczne, bo wiele bibliotek (mlflow, s3fs, boto3, SMTP) czyta konfigurację wyłącznie z ENV.
    Nigdy nie loguje wartości.
    """
    _ensure_dotenv_loaded()  # jeśli chcesz, aby .env nadpisał cache na starcie
    mapping: Dict[str, str] = {**_DEFAULT_EXPORT_MAP, **_load_extra_map_from_env()}
    for src_path, env_key in mapping.items():
        val = get_secret(src_path, None)
        if val is not None and str(val) != "":
            os.environ.setdefault(env_key, str(val))


# === NAZWA_SEKCJI === NARZĘDZIA DODATKOWE ===
def secret_exists(path: str) -> bool:
    """Szybkie sprawdzenie czy sekret istnieje w st.secrets lub ENV."""
    if _HAS_ST and _deep_get(_SECRETS, path) not in (None, ""):
        return True
    _ensure_dotenv_loaded()
    for k in _env_candidates(path):
        if os.getenv(k, None) not in (None, ""):
            return True
    return False

def redact(value: Any, keep_last: int = 4) -> str:
    """
    Zwraca zredagowaną wersję (np. 'sk-***abcd') – pomocne w UI, by nie ujawniać pełnych sekretów.
    """
    s = str(value or "")
    if len(s) <= keep_last:
        return "*" * len(s)
    return s[:2] + "***" + s[-keep_last:]
