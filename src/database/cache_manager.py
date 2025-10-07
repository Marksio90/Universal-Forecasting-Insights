# cache_engine.py — TURBO PRO (API-kompatybilny)
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

# Optional: Streamlit secrets
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
AUTO_COMPRESS_THRESHOLD = 4096  # 4KB

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
    def lock(self, name: str, ttl_ms: int = 10000, wait_ms: int = 10000, retry_ms: int = 150) -> "BaseLock": ...
    def status(self) -> str: ...
    def close(self) -> None: ...

# ===========================
# Konfiguracja z pliku/env/secrets
# ===========================
def _load_cache_config() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "enabled": False,
        "redis_url": None,
        "namespace": DEFAULT_NAMESPACE,
        "default_ttl": DEFAULT_TTL,
        "compression": "zlib",   # none | zlib | auto
    }

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
                    "compression": c.get("compression", cfg.get("compression")),
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
                cfg["compression"] = s.get("compression", cfg.get("compression"))
        except Exception:
            pass

    # env overrides
    enabled_env = os.getenv("CACHE_ENABLED")
    if enabled_env is not None:
        cfg["enabled"] = enabled_env.lower() in ("1", "true", "yes", "on")

    cfg["redis_url"] = os.getenv("REDIS_URL", cfg["redis_url"])
    cfg["namespace"] = os.getenv("CACHE_NAMESPACE", cfg["namespace"])
    cfg["default_ttl"] = int(os.getenv("CACHE_DEFAULT_TTL", cfg["default_ttl"]))
    cfg["compression"] = os.getenv("CACHE_COMPRESSION", cfg.get("compression", "zlib")).lower()

    if cfg["compression"] not in ("none", "zlib", "auto"):
        cfg["compression"] = "zlib"

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

def _should_compress(raw: bytes, mode: str) -> bool:
    if mode == "none":
        return False
    if mode == "zlib":
        return True
    # auto
    return len(raw) >= AUTO_COMPRESS_THRESHOLD

def _dumps(obj: Any, compression_mode: str) -> bytes:
    """
    Zapis obiektu do bajtów:
      - JSON (gdy możliwy) albo pickle.
    Nagłówek: b"IP|v1|<fmt>|<cmp>|" + payload
    """
    payload: bytes
    fmt = "json"
    b = _try_json_dumps(obj)
    if b is None:
        fmt = "pickle"
        b = _pickle_dumps(obj)

    cmp_flag = "zlib" if _should_compress(b, compression_mode) else "none"
    body = zlib.compress(b, COMPRESS_LEVEL) if cmp_flag == "zlib" else b
    header = f"IP|{CACHE_VERSION}|{fmt}|{cmp_flag}|".encode("utf-8")
    return header + body

def _loads(b: Optional[bytes]) -> Any:
    if b is None:
        return None

    # Szybka ścieżka: spróbuj odczytać nagłówek „IP|...|...|...|”
    try:
        # Znajdź offset po 4-tym '|' w strumieniu bajtów
        sep = b"|"
        idx = -1
        start = 0
        parts: list[bytes] = []
        for _ in range(4):
            pos = b.find(sep, start)
            if pos == -1:
                raise ValueError("header not found")
            parts.append(b[start:pos])
            start = pos + 1
        # parts = [b"IP", b"v1", b"fmt", b"cmp"]
        magic, ver, fmt_b, cmp_b = parts
        if magic != b"IP":
            raise ValueError("bad magic")
        blob = b[start:]  # reszta po nagłówku
        fmt = fmt_b.decode("utf-8", errors="ignore")
        cmp_flag = cmp_b.decode("utf-8", errors="ignore")

        if cmp_flag == "zlib":
            try:
                blob = zlib.decompress(blob)
            except Exception:
                return None

        if fmt == "json":
            try:
                return json.loads(blob.decode("utf-8"))
            except Exception:
                # spróbuj pickle w ostateczności
                try:
                    return pickle.loads(blob)
                except Exception:
                    return None
        else:
            try:
                return pickle.loads(blob)
            except Exception:
                return None
    except Exception:
        # Brak/niepoprawny nagłówek — spróbuj heurystyk
        for attempt in ("json", "pickle", "zlib+pickle", "zlib+json"):
            try:
                if attempt == "json":
                    return json.loads(b.decode("utf-8"))
                if attempt == "pickle":
                    return pickle.loads(b)
                if attempt == "zlib+pickle":
                    return pickle.loads(zlib.decompress(b))
                if attempt == "zlib+json":
                    return json.loads(zlib.decompress(b).decode("utf-8"))
            except Exception:
                continue
        return None

# ===========================
# Klucze i nazwy
# ===========================
def _hash_obj(*parts: Any) -> str:
    """
    Stabilny hash argumentów (używa json, a gdy się nie da — pickle).
    """
    h = hashlib.sha256()
    for p in parts:
        try:
            h.update(json.dumps(p, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8"))
        except Exception:
            h.update(pickle.dumps(p, protocol=pickle.HIGHEST_PROTOCOL))
    return h.hexdigest()[:48]

def _sanitize_key(s: str) -> str:
    s = s.replace(" ", "_").replace("\n", "_")
    if len(s) > 180:
        s = s[:180]
    return s

def _namespaced(ns: str, key: str) -> str:
    return f"{ns}:{CACHE_VERSION}:{_sanitize_key(key)}"

# ===========================
# Lock (SET NX PX) + bezpieczne odblokowanie Lua
# ===========================
class BaseLock:
    def __enter__(self) -> "BaseLock": return self
    def __exit__(self, exc_type, exc, tb) -> None: return None
    @property
    def acquired(self) -> bool: return False

class RedisLock(BaseLock):
    def __init__(self, client: redis.Redis, name: str, ttl_ms: int, wait_ms: int, retry_ms: int):
        self.client = client
        self.name = name
        self.token = uuid.uuid4().hex
        self.ttl_ms = ttl_ms
        self.wait_ms = wait_ms
        self.retry_ms = retry_ms
        self._acquired = False

    @property
    def acquired(self) -> bool:
        return self._acquired

    def __enter__(self) -> "RedisLock":
        deadline = time.time() + self.wait_ms / 1000.0
        while time.time() < deadline:
            try:
                if self.client.set(self.name, self.token, nx=True, px=self.ttl_ms):
                    self._acquired = True
                    break
            except Exception:
                break
            time.sleep(self.retry_ms / 1000.0)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._acquired:
            return
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
            try:
                self.client.delete(self.name)
            except Exception:
                pass
        finally:
            self._acquired = False

class NoopLock(BaseLock):
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
    def ttl(self, key: str) -> int: return -2
    def incr(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int: return 0
    def flush(self, pattern: Optional[str] = None) -> int: return 0
    def lock(self, name: str, ttl_ms: int = 10000, wait_ms: int = 10000, retry_ms: int = 150) -> BaseLock:
        return NoopLock()
    def status(self) -> str: return "Cache disabled (set cache.enabled=true in config.yaml or REDIS_URL)."
    def close(self) -> None: return None

class RedisCache:
    def __init__(self, client: redis.Redis, namespace: str, default_ttl: int, compression_mode: str):
        self.client = client
        self.namespace = namespace
        self.enabled = True
        self.default_ttl = default_ttl
        self.compression_mode = compression_mode  # none | zlib | auto

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
            blob = _dumps(value, compression_mode=self.compression_mode)
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
            match = f"{pref}{pattern or '*'}"
            n = 0
            for key in self.client.scan_iter(match):
                n += int(self.client.delete(key))
            return n
        except Exception:
            return 0

    # --------------- lock ---------------
    def lock(self, name: str, ttl_ms: int = 10000, wait_ms: int = 10000, retry_ms: int = 150) -> BaseLock:
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
        client = redis.Redis.from_url(
            url,
            decode_responses=False,
            health_check_interval=30,
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        client.ping()
        compression = cfg.get("compression", "zlib")
        _CACHE = RedisCache(
            client,
            namespace=cfg.get("namespace", DEFAULT_NAMESPACE),
            default_ttl=int(cfg.get("default_ttl", DEFAULT_TTL)),
            compression_mode=compression,
        )
        return _CACHE
    except Exception:
        _CACHE = NoopCache(enabled=False, namespace=cfg.get("namespace", DEFAULT_NAMESPACE))
        return _CACHE

def get_cache() -> Cache:
    """Zwraca instancję cache (Redis lub Noop)."""
    global _CACHE
    if isinstance(_CACHE, NoopCache) and _CACHE.namespace == DEFAULT_NAMESPACE:
        return _init_cache()
    return _CACHE

# ===========================
# Dekorator cache’ujący (anti-stampede opcjonalny)
# ===========================
def _default_key_builder(func: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> str:
    base = f"{func.__module__}.{func.__name__}"
    args_hash = _hash_obj(args, kwargs)
    return f"{base}:{args_hash}"

def cacheable(
    namespace: Optional[str] = None,
    ttl: Optional[int] = None,
    key_builder: Optional[Callable[[Callable[..., Any], Tuple[Any, ...], Dict[str, Any]], str]] = None,
    *,
    lock_ms: int = 0,
    grace_ttl: int = 0,
):
    """
    Dekorator cache’ujący wynik funkcji.
    - namespace: dodatkowy sub-namespace (doklejany do globalnego)
    - ttl: czas życia klucza w sekundach
    - key_builder: funkcja budująca klucz (domyślnie hash args/kwargs)
    - lock_ms: gdy >0, używa krótkiego locka przy miss (anti-stampede)
    - grace_ttl: gdy >0 i klucz wygasł — zwraca „stare” dane, a w tle odświeża (best-effort, bez wątku)
      (tu: jeśli znajdziemy klucz „stale”, oddajemy go zamiast wyliczać; implementacja prosta, bez async)
    """
    def _decorator(func: Callable[..., T]) -> Callable[..., T]:
        def _wrapped(*args: Any, **kwargs: Any) -> T:
            cache = get_cache()
            ns = f"{cache.namespace}:{namespace}" if (namespace and namespace.strip()) else cache.namespace
            kb = key_builder or _default_key_builder
            k_local = kb(func, args, kwargs)
            full_key = f"{ns}:{k_local}"
            stale_key = f"{full_key}:stale" if grace_ttl > 0 else None

            # Fast path: hit
            if cache.enabled:
                hit = cache.get(full_key)
                if hit is not None:
                    return hit  # type: ignore[return-value]
                # Spróbuj stale
                if stale_key:
                    stale = cache.get(stale_key)
                    if stale is not None:
                        # zwróć od razu dane z „grace period”
                        return stale  # type: ignore[return-value]

            # Miss → opcjonalny lock
            if cache.enabled and lock_ms > 0:
                with cache.lock(full_key, ttl_ms=lock_ms, wait_ms=lock_ms, retry_ms=min(200, lock_ms)) as lk:
                    # Jeszcze raz sprawdź po locku (double check)
                    if cache.enabled:
                        hit2 = cache.get(full_key)
                        if hit2 is not None:
                            return hit2  # type: ignore[return-value]
                    result = func(*args, **kwargs)
                    if cache.enabled:
                        cache.set(full_key, result, ttl=ttl)
                        if stale_key:
                            cache.set(stale_key, result, ttl=grace_ttl)
                    return result  # type: ignore[return-value]

            # Miss bez locka
            result = func(*args, **kwargs)
            if cache.enabled:
                cache.set(full_key, result, ttl=ttl)
                if stale_key:
                    cache.set(stale_key, result, ttl=grace_ttl)
            return result  # type: ignore[return-value]
        return _wrapped
    return _decorator

# ===========================
# Publiczny status (kompatybilny podpis)
# ===========================
def status() -> str:
    """
    Zwraca zwięzły status cache (kompatybilny).
    """
    c = get_cache()
    return c.status()
