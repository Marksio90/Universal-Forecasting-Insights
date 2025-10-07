from __future__ import annotations
import os
import io
import json
import zlib
import time
import uuid
import pickle
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Iterator, TypeVar, Dict, Tuple

# Optional: Streamlit secrets integracja (nie wymagane)
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None  # pragma: no cover

import yaml  # pyyaml
import redis  # redis-py

T = TypeVar("T")

# ===========================
# Konfiguracja i stałe
# ===========================

DEFAULT_NAMESPACE = "intelligent-predictor"
DEFAULT_TTL = 3600  # sekundy
CACHE_VERSION = "v1"
COMPRESS_LEVEL = 3  # 0..9

# ===========================
# Interfejs Cache
# ===========================

class Cache(Protocol):
    enabled: bool
    namespace: str

    def get(self, key: str, default: Any = None) -> Any: ...
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool: ...
    def delete(self, key: str) -> int: ...
    def exists(self, key: str) -> bool: ...
    def expire(self, key: str, ttl: int) -> bool: ...
    def ttl(self, key: str) -> int: ...
    def incr(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int: ...
    def flush(self, pattern: Optional[str] = None) -> int: ...
    def lock(self, name: str, ttl_ms: int = 10000, wait_ms: int = 10000, retry_ms: int = 150) -> "RedisLock": ...
    def status(self) -> str: ...
    def close(self) -> None: ...

# ===========================
# Konfiguracja z pliku/env/secrets
# ===========================

def _load_cache_config() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"enabled": False, "redis_url": None, "namespace": DEFAULT_NAMESPACE, "default_ttl": DEFAULT_TTL}
    # config.yaml
    try:
        import pathlib
        cfg_path = pathlib.Path("config.yaml")
        if cfg_path.exists():
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            if isinstance(raw, dict):
                c = (raw.get("cache") or {}) if isinstance(raw.get("cache"), dict) else {}
                cfg.update({
                    "enabled": bool(c.get("enabled", cfg["enabled"])),
                    "redis_url": c.get("redis_url", cfg["redis_url"]),
                    "namespace": c.get("namespace", cfg["namespace"]),
                    "default_ttl": int(c.get("default_ttl", cfg["default_ttl"])),
                    "compression": c.get("compression", "zlib"),
                })
    except Exception:
        pass

    # streamlit secrets
    if st is not None:
        try:
            s = st.secrets.get("cache", {})  # type: ignore[attr-defined]
            if s:
                cfg["enabled"] = bool(s.get("enabled", cfg["enabled"]))
                cfg["redis_url"] = s.get("redis_url", cfg["redis_url"])
                cfg["namespace"] = s.get("namespace", cfg["namespace"])
                cfg["default_ttl"] = int(s.get("default_ttl", cfg["default_ttl"]))
                cfg["compression"] = s.get("compression", cfg.get("compression", "zlib"))
        except Exception:
            pass

    # env overrides
    cfg["enabled"] = bool(os.getenv("CACHE_ENABLED", str(cfg["enabled"])).lower() in ("1", "true", "yes"))
    cfg["redis_url"] = os.getenv("REDIS_URL", cfg["redis_url"])
    cfg["namespace"] = os.getenv("CACHE_NAMESPACE", cfg["namespace"])
    cfg["default_ttl"] = int(os.getenv("CACHE_DEFAULT_TTL", cfg["default_ttl"]))
    cfg["compression"] = os.getenv("CACHE_COMPRESSION", cfg.get("compression", "zlib"))

    return cfg

# ===========================
# Serializacja
# ===========================

def _try_json_dumps(obj: Any) -> Optional[bytes]:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except Exception:
        return None

def _pickle_dumps(obj: Any) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

def _dumps(obj: Any, compress: bool = True) -> bytes:
    """
    Zapis obiektu do bajtów:
      - najpierw JSON (jeśli możliwe),
      - w przeciwnym razie pickle.
    Nagłówek: b"IP|v1|<fmt>|<cmp>|" + payload
    """
    payload: bytes
    fmt = "json"
    b = _try_json_dumps(obj)
    if b is None:
        fmt = "pickle"
        b = _pickle_dumps(obj)
    if compress:
        b = zlib.compress(b, COMPRESS_LEVEL)
        cmp_flag = "zlib"
    else:
        cmp_flag = "none"
    header = f"IP|{CACHE_VERSION}|{fmt}|{cmp_flag}|".encode("utf-8")
    return header + b

def _loads(b: Optional[bytes]) -> Any:
    if b is None:
        return None
    try:
        header, payload = b.split(b"|", 4)[0:4], b.split(b"|", 4)[-1]
        # header = [b"IP", b"v1", b"fmt", b"cmp"]
        parts = b.decode("utf-8", errors="ignore").split("|", 4)
        # robust parse (fallback in edge cases)
    except Exception:
        # brak nagłówka – spróbuj raw json/pickle/zlib
        try:
            return json.loads(b.decode("utf-8"))
        except Exception:
            try:
                return pickle.loads(b)
            except Exception:
                try:
                    return pickle.loads(zlib.decompress(b))
                except Exception:
                    return None

    try:
        _, ver, fmt, cmp_flag, payload = b.decode("utf-8", errors="ignore").split("|", 4)
    except Exception:
        # fallback „bezpieczny”
        try:
            return pickle.loads(zlib.decompress(b))
        except Exception:
            return None

    blob = payload.encode("utf-8")  # this isn't correct for binary payload; fix below
    # Prawidłowa ekstrakcja payload z pierwotnego bajtowego bufora:
    try:
        # znajdź offset po 4 separatorach '|'
        sep = b"|"
        idx = 0
        for _ in range(4):
            idx = b.find(sep, idx) + 1
        blob = b[idx:]
    except Exception:
        blob = b

    if cmp_flag == "zlib":
        try:
            blob = zlib.decompress(blob)
        except Exception:
            return None
    if fmt == "json":
        try:
            return json.loads(blob.decode("utf-8"))
        except Exception:
            # spróbuj pickle (gdy nagłówek był mylący)
            try:
                return pickle.loads(blob)
            except Exception:
                return None
    else:
        try:
            return pickle.loads(blob)
        except Exception:
            return None

# ===========================
# Klucze i nazwy
# ===========================

def _hash_obj(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        try:
            h.update(repr(p).encode("utf-8"))
        except Exception:
            h.update(str(p).encode("utf-8"))
    return h.hexdigest()[:40]

def _namespaced(ns: str, key: str) -> str:
    # bezpieczny klucz redis
    safe = key.replace(" ", "_").replace("\n", "_")
    return f"{ns}:{CACHE_VERSION}:{safe}"

# ===========================
# Lock (SET NX PX) + bezpieczne odblokowanie Lua
# ===========================

class RedisLock:
    def __init__(self, client: redis.Redis, name: str, ttl_ms: int, wait_ms: int, retry_ms: int):
        self.client = client
        self.name = f"lock:{name}"
        self.token = uuid.uuid4().hex
        self.ttl_ms = ttl_ms
        self.wait_ms = wait_ms
        self.retry_ms = retry_ms
        self.acquired = False

    def __enter__(self) -> "RedisLock":
        deadline = time.time() + self.wait_ms / 1000.0
        while time.time() < deadline:
            if self.client.set(self.name, self.token, nx=True, px=self.ttl_ms):
                self.acquired = True
                break
            time.sleep(self.retry_ms / 1000.0)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.acquired:
            return
        # bezpieczne zwolnienie: odblokuj tylko, jeśli token pasuje
        unlock_script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('del', KEYS[1])
        else
            return 0
        end
        """
        try:
            self.client.eval(unlock_script, 1, self.name, self.token)
        except Exception:
            # w ostateczności – spróbuj zwykłe del (może odblokować cudzy lock, ale to edge case)
            try:
                self.client.delete(self.name)
            except Exception:
                pass

# ===========================
# Implementacje Cache
# ===========================

@dataclass
class NoopCache:
    enabled: bool = False
    namespace: str = DEFAULT_NAMESPACE

    def get(self, key: str, default: Any = None) -> Any: return default
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool: return False
    def delete(self, key: str) -> int: return 0
    def exists(self, key: str) -> bool: return False
    def expire(self, key: str, ttl: int) -> bool: return False
    def ttl(self, key: str) -> int: return -2  # -2 = nie istnieje
    def incr(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int: return 0
    def flush(self, pattern: Optional[str] = None) -> int: return 0
    def lock(self, name: str, ttl_ms: int = 10000, wait_ms: int = 10000, retry_ms: int = 150) -> RedisLock:
        # Lock „udaje”, że nie został pozyskany
        return RedisLock(None, name, ttl_ms, wait_ms, retry_ms)  # type: ignore[arg-type]
    def status(self) -> str: return "Cache disabled (set cache.enabled=true in config.yaml or REDIS_URL)."
    def close(self) -> None: return None

class RedisCache:
    def __init__(self, client: redis.Redis, namespace: str, default_ttl: int, compression: bool):
        self.client = client
        self.namespace = namespace
        self.enabled = True
        self.default_ttl = default_ttl
        self.compression = compression

    # --------------- podstawowe ---------------
    def get(self, key: str, default: Any = None) -> Any:
        k = _namespaced(self.namespace, key)
        try:
            b = self.client.get(k)
            val = _loads(b)
            return default if val is None else val
        except Exception:
            return default

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        k = _namespaced(self.namespace, key)
        try:
            blob = _dumps(value, compress=self.compression)
            ex = ttl if ttl is not None else self.default_ttl
            return bool(self.client.set(k, blob, ex=ex))
        except Exception:
            return False

    def delete(self, key: str) -> int:
        k = _namespaced(self.namespace, key)
        try:
            return int(self.client.delete(k))
        except Exception:
            return 0

    def exists(self, key: str) -> bool:
        k = _namespaced(self.namespace, key)
        try:
            return bool(self.client.exists(k))
        except Exception:
            return False

    def expire(self, key: str, ttl: int) -> bool:
        k = _namespaced(self.namespace, key)
        try:
            return bool(self.client.expire(k, ttl))
        except Exception:
            return False

    def ttl(self, key: str) -> int:
        k = _namespaced(self.namespace, key)
        try:
            return int(self.client.ttl(k))
        except Exception:
            return -2

    def incr(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        k = _namespaced(self.namespace, key)
        try:
            val = int(self.client.incrby(k, amount))
            if ttl:
                self.client.expire(k, ttl)
            return val
        except Exception:
            return 0

    def flush(self, pattern: Optional[str] = None) -> int:
        """
        Usuwa klucze w namespace. Jeśli pattern podany, stosuje go po prefiksie.
        """
        try:
            pref = _namespaced(self.namespace, "")
            match = f"{pref}*{pattern or ''}"
            n = 0
            for key in self.client.scan_iter(match):
                n += int(self.client.delete(key))
            return n
        except Exception:
            return 0

    # --------------- lock ---------------
    def lock(self, name: str, ttl_ms: int = 10000, wait_ms: int = 10000, retry_ms: int = 150) -> RedisLock:
        lname = _namespaced(self.namespace, f"lock:{name}")
        return RedisLock(self.client, lname, ttl_ms=ttl_ms, wait_ms=wait_ms, retry_ms=retry_ms)

    # --------------- inne ---------------
    def status(self) -> str:
        try:
            t0 = time.time()
            pong = self.client.ping()
            dt = (time.time() - t0) * 1000
            info = self.client.info(section="memory")
            used = info.get("used_memory_human", "n/a")
            return f"Cache enabled • ping={dt:.1f}ms • used={used} • ns={self.namespace}"
        except Exception as e:
            return f"Cache error: {e}"

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

# ===========================
# Inicjalizacja singletonu
# ===========================

_CACHE: Cache = NoopCache()

def _init_cache() -> Cache:
    global _CACHE
    cfg = _load_cache_config()
    if not cfg.get("enabled"):
        _CACHE = NoopCache(enabled=False, namespace=cfg.get("namespace", DEFAULT_NAMESPACE))
        return _CACHE

    url = cfg.get("redis_url") or os.getenv("REDIS_URL")
    if not url:
        _CACHE = NoopCache(enabled=False, namespace=cfg.get("namespace", DEFAULT_NAMESPACE))
        return _CACHE

    try:
        client = redis.Redis.from_url(url, decode_responses=False, health_check_interval=30, socket_timeout=5)
        # proste sprawdzenie połączenia
        client.ping()
        compression = (cfg.get("compression", "zlib") != "none")
        _CACHE = RedisCache(client, namespace=cfg.get("namespace", DEFAULT_NAMESPACE), default_ttl=int(cfg.get("default_ttl", DEFAULT_TTL)), compression=compression)
        return _CACHE
    except Exception:
        _CACHE = NoopCache(enabled=False, namespace=cfg.get("namespace", DEFAULT_NAMESPACE))
        return _CACHE

def get_cache() -> Cache:
    """Zwraca instancję cache (Redis lub Noop)."""
    global _CACHE
    if isinstance(_CACHE, NoopCache) and _CACHE.namespace == DEFAULT_NAMESPACE:
        # pierwsze wywołanie – spróbuj zainicjalizować
        return _init_cache()
    return _CACHE

# ===========================
# Dekorator cache’ujący
# ===========================

def _default_key_builder(func: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> str:
    base = f"{func.__module__}.{func.__name__}"
    args_hash = _hash_obj(args, kwargs)
    return f"{base}:{args_hash}"

def cacheable(namespace: Optional[str] = None, ttl: Optional[int] = None, key_builder: Optional[Callable[[Callable[..., Any], Tuple[Any, ...], Dict[str, Any]], str]] = None):
    """
    Dekorator cache’ujący wynik funkcji w Redis (jeśli dostępny).
    - namespace: pod-namesapce dla kluczy (domyślnie z config)
    - ttl: czas życia klucza w sekundach (domyślnie z config)
    - key_builder: funkcja budująca klucz na bazie argumentów
    """
    def _decorator(func: Callable[..., T]) -> Callable[..., T]:
        def _wrapped(*args: Any, **kwargs: Any) -> T:
            cache = get_cache()
            ns = f"{cache.namespace}:{namespace}" if (namespace and namespace.strip()) else cache.namespace
            kb = key_builder or _default_key_builder
            k = kb(func, args, kwargs)
            full_key = f"{ns}:{k}"
            if cache.enabled:
                hit = cache.get(full_key)
                if hit is not None:
                    return hit  # type: ignore[return-value]
            result = func(*args, **kwargs)
            if cache.enabled:
                cache.set(full_key, result, ttl=ttl)
            return result  # type: ignore[return-value]
        return _wrapped
    return _decorator

# ===========================
# Publiczny status (kompatybilny podpis)
# ===========================

def status() -> str:
    """
    Zwraca zwięzły status cache (kompatybilny z poprzednim placeholderem).
    """
    c = get_cache()
    return c.status()
