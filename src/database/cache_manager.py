"""
cache_engine.py — ULTRA PRO Edition

Advanced caching engine with:
- Redis backend with connection pooling
- Automatic compression (zlib/gzip)
- Anti-stampede protection with distributed locks
- Grace period support (stale-while-revalidate)
- Thread-safe operations
- Comprehensive error handling
- Monitoring and metrics
- TTL management
- Namespace isolation
"""

from __future__ import annotations

import os
import json
import zlib
import time
import uuid
import pickle
import hashlib
import logging
import threading
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Optional, Protocol, TypeVar, Dict, Tuple,
    List, Union, runtime_checkable
)
from functools import wraps
from contextlib import contextmanager

# ========================================================================================
# OPTIONAL DEPENDENCIES
# ========================================================================================

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    st = None  # type: ignore
    HAS_STREAMLIT = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None  # type: ignore
    HAS_YAML = False

try:
    import redis
    from redis.connection import ConnectionPool
    HAS_REDIS = True
except ImportError:
    redis = None  # type: ignore
    ConnectionPool = None  # type: ignore
    HAS_REDIS = False

# ========================================================================================
# LOGGING
# ========================================================================================

def get_logger(name: str = "cache_engine") -> logging.Logger:
    """Konfiguruje logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        logger.addHandler(handler)
        logger.propagate = False
    return logger

LOGGER = get_logger()

# ========================================================================================
# CONSTANTS
# ========================================================================================

T = TypeVar("T")

# Versioning
CACHE_VERSION = "v2"
PROTOCOL_VERSION = "IP"  # Intelligent Predictor

# Defaults
DEFAULT_NAMESPACE = "intelligent-predictor"
DEFAULT_TTL = 3600  # 1 hour
DEFAULT_GRACE_TTL = 86400  # 24 hours for stale data

# Compression
COMPRESS_LEVEL = 6  # 0-9, 6 is good balance
AUTO_COMPRESS_THRESHOLD = 4096  # 4KB
MAX_COMPRESS_SIZE = 100 * 1024 * 1024  # 100MB

# Connection settings
DEFAULT_SOCKET_TIMEOUT = 5
DEFAULT_SOCKET_CONNECT_TIMEOUT = 5
DEFAULT_HEALTH_CHECK_INTERVAL = 30
DEFAULT_MAX_CONNECTIONS = 50

# Lock settings
DEFAULT_LOCK_TTL_MS = 10000  # 10 seconds
DEFAULT_LOCK_WAIT_MS = 10000  # 10 seconds
DEFAULT_LOCK_RETRY_MS = 100  # 100ms between retries

# Limits
MAX_KEY_LENGTH = 250
MAX_VALUE_SIZE = 512 * 1024 * 1024  # 512MB

# Metrics
METRICS_WINDOW = 300  # 5 minutes

# ========================================================================================
# DATACLASSES
# ========================================================================================

@dataclass
class CacheConfig:
    """Configuration for cache engine."""
    
    # Core settings
    enabled: bool = False
    redis_url: Optional[str] = None
    namespace: str = DEFAULT_NAMESPACE
    default_ttl: int = DEFAULT_TTL
    
    # Compression
    compression: str = "auto"  # none | zlib | gzip | auto
    compress_threshold: int = AUTO_COMPRESS_THRESHOLD
    compress_level: int = COMPRESS_LEVEL
    
    # Connection settings
    socket_timeout: int = DEFAULT_SOCKET_TIMEOUT
    socket_connect_timeout: int = DEFAULT_SOCKET_CONNECT_TIMEOUT
    health_check_interval: int = DEFAULT_HEALTH_CHECK_INTERVAL
    max_connections: int = DEFAULT_MAX_CONNECTIONS
    
    # Lock settings
    lock_ttl_ms: int = DEFAULT_LOCK_TTL_MS
    lock_wait_ms: int = DEFAULT_LOCK_WAIT_MS
    lock_retry_ms: int = DEFAULT_LOCK_RETRY_MS
    
    # Grace period
    grace_ttl: int = DEFAULT_GRACE_TTL
    
    # Monitoring
    enable_metrics: bool = True
    verbose: bool = False


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    lock_acquisitions: int = 0
    lock_failures: int = 0
    total_get_time: float = 0.0
    total_set_time: float = 0.0
    start_time: float = field(default_factory=time.time)
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    @property
    def avg_get_time_ms(self) -> float:
        """Average GET time in milliseconds."""
        total = self.hits + self.misses
        return (self.total_get_time / total * 1000) if total > 0 else 0.0
    
    @property
    def avg_set_time_ms(self) -> float:
        """Average SET time in milliseconds."""
        return (self.total_set_time / self.sets * 1000) if self.sets > 0 else 0.0
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.errors = 0
        self.lock_acquisitions = 0
        self.lock_failures = 0
        self.total_get_time = 0.0
        self.total_set_time = 0.0
        self.start_time = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        uptime = time.time() - self.start_time
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 2),
            "sets": self.sets,
            "deletes": self.deletes,
            "errors": self.errors,
            "lock_acquisitions": self.lock_acquisitions,
            "lock_failures": self.lock_failures,
            "avg_get_time_ms": round(self.avg_get_time_ms, 3),
            "avg_set_time_ms": round(self.avg_set_time_ms, 3),
            "uptime_seconds": round(uptime, 1)
        }


# ========================================================================================
# PROTOCOLS
# ========================================================================================

@runtime_checkable
class Cache(Protocol):
    """Cache interface protocol."""
    
    enabled: bool
    namespace: str
    
    def get(self, key: str, default: Any = None) -> Any: ...
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool: ...
    def delete(self, key: str) -> int: ...
    def exists(self, key: str) -> bool: ...
    def expire(self, key: str, ttl: int) -> bool: ...
    def ttl(self, key: str) -> int: ...
    def incr(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int: ...
    def decr(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int: ...
    def flush(self, pattern: Optional[str] = None) -> int: ...
    def lock(self, name: str, ttl_ms: Optional[int] = None, wait_ms: Optional[int] = None) -> "BaseLock": ...
    def status(self) -> str: ...
    def get_metrics(self) -> Dict[str, Any]: ...
    def close(self) -> None: ...


@runtime_checkable  
class BaseLock(Protocol):
    """Lock interface protocol."""
    
    @property
    def acquired(self) -> bool: ...
    
    def __enter__(self) -> "BaseLock": ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...


# ========================================================================================
# CONFIGURATION LOADING
# ========================================================================================

def _load_cache_config() -> CacheConfig:
    """
    Load cache configuration from multiple sources (priority order):
    1. Environment variables (highest priority)
    2. Streamlit secrets
    3. config.yaml
    4. Defaults (lowest priority)
    
    Returns:
        CacheConfig with merged settings
    """
    config = CacheConfig()
    
    # ============================================================================
    # 1. Load from config.yaml
    # ============================================================================
    
    if HAS_YAML and yaml is not None:
        try:
            from pathlib import Path
            
            config_path = Path("config.yaml")
            
            if config_path.exists():
                LOGGER.debug("Loading cache config from config.yaml")
                
                data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                
                if isinstance(data, dict) and "cache" in data:
                    cache_cfg = data["cache"]
                    
                    if isinstance(cache_cfg, dict):
                        config.enabled = bool(cache_cfg.get("enabled", config.enabled))
                        config.redis_url = cache_cfg.get("redis_url", config.redis_url)
                        config.namespace = cache_cfg.get("namespace", config.namespace)
                        config.default_ttl = int(cache_cfg.get("default_ttl", config.default_ttl))
                        config.compression = cache_cfg.get("compression", config.compression)
                        config.compress_threshold = int(cache_cfg.get("compress_threshold", config.compress_threshold))
                        config.compress_level = int(cache_cfg.get("compress_level", config.compress_level))
                        config.grace_ttl = int(cache_cfg.get("grace_ttl", config.grace_ttl))
                        config.enable_metrics = bool(cache_cfg.get("enable_metrics", config.enable_metrics))
                        config.verbose = bool(cache_cfg.get("verbose", config.verbose))
                        
                        LOGGER.info("Cache config loaded from config.yaml")
        except Exception as e:
            LOGGER.warning(f"Failed to load config.yaml: {e}")
    
    # ============================================================================
    # 2. Load from Streamlit secrets
    # ============================================================================
    
    if HAS_STREAMLIT and st is not None:
        try:
            secrets = st.secrets.get("cache", {})
            
            if secrets:
                LOGGER.debug("Loading cache config from Streamlit secrets")
                
                config.enabled = bool(secrets.get("enabled", config.enabled))
                config.redis_url = secrets.get("redis_url", config.redis_url)
                config.namespace = secrets.get("namespace", config.namespace)
                config.default_ttl = int(secrets.get("default_ttl", config.default_ttl))
                config.compression = secrets.get("compression", config.compression)
                
                LOGGER.info("Cache config loaded from Streamlit secrets")
        except Exception as e:
            LOGGER.debug(f"No Streamlit secrets found: {e}")
    
    # ============================================================================
    # 3. Environment variables (highest priority)
    # ============================================================================
    
    # Enabled flag
    enabled_env = os.getenv("CACHE_ENABLED")
    if enabled_env is not None:
        config.enabled = enabled_env.lower() in ("1", "true", "yes", "on")
        LOGGER.debug(f"Cache enabled from env: {config.enabled}")
    
    # Redis URL
    redis_url_env = os.getenv("REDIS_URL") or os.getenv("CACHE_REDIS_URL")
    if redis_url_env:
        config.redis_url = redis_url_env
        LOGGER.debug("Redis URL loaded from environment")
    
    # Other settings
    if os.getenv("CACHE_NAMESPACE"):
        config.namespace = os.getenv("CACHE_NAMESPACE", config.namespace)
    
    if os.getenv("CACHE_DEFAULT_TTL"):
        try:
            config.default_ttl = int(os.getenv("CACHE_DEFAULT_TTL", config.default_ttl))
        except ValueError:
            pass
    
    if os.getenv("CACHE_COMPRESSION"):
        config.compression = os.getenv("CACHE_COMPRESSION", config.compression).lower()
    
    if os.getenv("CACHE_VERBOSE"):
        config.verbose = os.getenv("CACHE_VERBOSE", "").lower() in ("1", "true", "yes")
    
    # ============================================================================
    # Validation
    # ============================================================================
    
    # Validate compression mode
    if config.compression not in ("none", "zlib", "gzip", "auto"):
        LOGGER.warning(f"Invalid compression mode: {config.compression}, using 'auto'")
        config.compression = "auto"
    
    # Validate TTLs
    if config.default_ttl <= 0:
        LOGGER.warning(f"Invalid default_ttl: {config.default_ttl}, using {DEFAULT_TTL}")
        config.default_ttl = DEFAULT_TTL
    
    if config.grace_ttl < config.default_ttl:
        LOGGER.warning(f"grace_ttl ({config.grace_ttl}) < default_ttl ({config.default_ttl}), adjusting")
        config.grace_ttl = config.default_ttl * 2
    
    # Validate compression settings
    if not (0 <= config.compress_level <= 9):
        LOGGER.warning(f"Invalid compress_level: {config.compress_level}, using {COMPRESS_LEVEL}")
        config.compress_level = COMPRESS_LEVEL
    
    # Set verbose logging
    if config.verbose:
        LOGGER.setLevel(logging.DEBUG)
        LOGGER.debug("Verbose logging enabled")
    
    return config


# ========================================================================================
# SERIALIZATION
# ========================================================================================

def _try_json_serialize(obj: Any) -> Optional[bytes]:
    """
    Try to serialize object as JSON.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON bytes or None if not serializable
    """
    try:
        return json.dumps(
            obj,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None


def _pickle_serialize(obj: Any) -> bytes:
    """
    Serialize object using pickle.
    
    Args:
        obj: Object to serialize
        
    Returns:
        Pickle bytes
    """
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def _should_compress(data: bytes, mode: str, threshold: int) -> bool:
    """
    Determine if data should be compressed.
    
    Args:
        data: Data to check
        mode: Compression mode
        threshold: Size threshold for auto mode
        
    Returns:
        True if should compress
    """
    if mode == "none":
        return False
    
    if mode in ("zlib", "gzip"):
        return True
    
    # Auto mode
    return len(data) >= threshold


def _compress_data(data: bytes, mode: str, level: int) -> bytes:
    """
    Compress data.
    
    Args:
        data: Data to compress
        mode: Compression mode (zlib or gzip)
        level: Compression level (0-9)
        
    Returns:
        Compressed data
        
    Raises:
        ValueError: If compression fails
    """
    if len(data) > MAX_COMPRESS_SIZE:
        raise ValueError(f"Data too large for compression: {len(data)} bytes")
    
    try:
        if mode == "gzip":
            import gzip
            return gzip.compress(data, compresslevel=level)
        else:
            return zlib.compress(data, level)
    except Exception as e:
        raise ValueError(f"Compression failed: {e}") from e


def _decompress_data(data: bytes, mode: str) -> bytes:
    """
    Decompress data.
    
    Args:
        data: Compressed data
        mode: Compression mode (zlib or gzip)
        
    Returns:
        Decompressed data
        
    Raises:
        ValueError: If decompression fails
    """
    try:
        if mode == "gzip":
            import gzip
            return gzip.decompress(data)
        else:
            return zlib.decompress(data)
    except Exception as e:
        raise ValueError(f"Decompression failed: {e}") from e


def serialize_value(obj: Any, config: CacheConfig) -> bytes:
    """
    Serialize value with optional compression.
    
    Format: PROTOCOL|VERSION|FORMAT|COMPRESSION|payload
    Example: IP|v2|json|zlib|<compressed_json_data>
    
    Args:
        obj: Object to serialize
        config: Cache configuration
        
    Returns:
        Serialized bytes
        
    Raises:
        ValueError: If serialization fails
    """
    # Try JSON first (faster and more portable)
    payload = _try_json_serialize(obj)
    
    if payload is not None:
        fmt = "json"
    else:
        # Fallback to pickle
        payload = _pickle_serialize(obj)
        fmt = "pickle"
    
    # Compression
    compression = "none"
    
    if _should_compress(payload, config.compression, config.compress_threshold):
        try:
            compressed = _compress_data(payload, config.compression, config.compress_level)
            
            # Only use compression if it actually reduces size
            if len(compressed) < len(payload):
                payload = compressed
                compression = config.compression if config.compression != "auto" else "zlib"
        except Exception as e:
            LOGGER.warning(f"Compression failed, storing uncompressed: {e}")
    
    # Build header
    header = f"{PROTOCOL_VERSION}|{CACHE_VERSION}|{fmt}|{compression}|".encode("utf-8")
    
    return header + payload


def deserialize_value(data: Optional[bytes]) -> Any:
    """
    Deserialize value with automatic format detection.
    
    Args:
        data: Serialized data
        
    Returns:
        Deserialized object or None
    """
    if data is None or len(data) == 0:
        return None
    
    try:
        # Try to parse header
        header_end = data.find(b"|", 20)  # Search for 4th pipe within first 20 bytes
        
        if header_end > 0:
            header_part = data[:header_end + 1]
            parts = header_part.split(b"|")
            
            if len(parts) >= 5 and parts[0] == PROTOCOL_VERSION.encode():
                # Valid header found
                protocol, version, fmt, compression, _ = parts[:5]
                
                # Extract payload (after 4th pipe)
                payload_start = len(b"|".join(parts[:4])) + 1
                payload = data[payload_start:]
                
                fmt_str = fmt.decode("utf-8", errors="ignore")
                compression_str = compression.decode("utf-8", errors="ignore")
                
                # Decompress if needed
                if compression_str in ("zlib", "gzip"):
                    try:
                        payload = _decompress_data(payload, compression_str)
                    except Exception as e:
                        LOGGER.warning(f"Decompression failed: {e}")
                        return None
                
                # Deserialize
                if fmt_str == "json":
                    try:
                        return json.loads(payload.decode("utf-8"))
                    except Exception:
                        # Try pickle fallback
                        try:
                            return pickle.loads(payload)
                        except Exception:
                            return None
                else:
                    # Pickle
                    try:
                        return pickle.loads(payload)
                    except Exception:
                        return None
        
        # No valid header - try heuristic deserialization
        LOGGER.debug("No valid header found, trying heuristic deserialization")
        
        # Try JSON
        try:
            return json.loads(data.decode("utf-8"))
        except Exception:
            pass
        
        # Try pickle
        try:
            return pickle.loads(data)
        except Exception:
            pass
        
        # Try compressed pickle
        try:
            return pickle.loads(zlib.decompress(data))
        except Exception:
            pass
        
        # Try compressed JSON
        try:
            return json.loads(zlib.decompress(data).decode("utf-8"))
        except Exception:
            pass
        
        LOGGER.warning("Failed to deserialize data with all methods")
        return None
        
    except Exception as e:
        LOGGER.error(f"Deserialization error: {e}", exc_info=True)
        return None


# ========================================================================================
# KEY MANAGEMENT
# ========================================================================================

def hash_object(*parts: Any) -> str:
    """
    Create stable hash from objects.
    
    Args:
        *parts: Objects to hash
        
    Returns:
        48-character hex hash
    """
    hasher = hashlib.sha256()
    
    for part in parts:
        try:
            # Try JSON first (deterministic)
            serialized = json.dumps(
                part,
                sort_keys=True,
                default=str,
                ensure_ascii=False
            ).encode("utf-8")
        except (TypeError, ValueError):
            # Fallback to pickle
            serialized = pickle.dumps(part, protocol=pickle.HIGHEST_PROTOCOL)
        
        hasher.update(serialized)
    
    return hasher.hexdigest()[:48]


def sanitize_key(key: str) -> str:
    """
    Sanitize cache key.
    
    Args:
        key: Raw key
        
    Returns:
        Sanitized key
    """
    # Replace problematic characters
    key = key.replace(" ", "_").replace("\n", "_").replace("\r", "_").replace("\t", "_")
    
    # Limit length
    if len(key) > MAX_KEY_LENGTH:
        # Keep start and hash of full key
        hash_suffix = hashlib.md5(key.encode()).hexdigest()[:8]
        key = key[:MAX_KEY_LENGTH - 9] + "_" + hash_suffix
    
    return key


def build_namespaced_key(namespace: str, key: str, version: str = CACHE_VERSION) -> str:
    """
    Build fully qualified cache key.
    
    Args:
        namespace: Cache namespace
        key: Base key
        version: Cache version
        
    Returns:
        Namespaced key
    """
    sanitized = sanitize_key(key)
    return f"{namespace}:{version}:{sanitized}"


# ========================================================================================
# LOCK IMPLEMENTATIONS
# ========================================================================================

class NoopLock(BaseLock):
    """No-op lock for disabled cache."""
    
    @property
    def acquired(self) -> bool:
        return False
    
    def __enter__(self) -> "NoopLock":
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


class RedisLock(BaseLock):
    """
    Distributed lock using Redis SET NX PX.
    
    Uses Lua script for atomic unlock to prevent releasing someone else's lock.
    """
    
    # Lua script for atomic unlock
    UNLOCK_SCRIPT = """
    if redis.call('get', KEYS[1]) == ARGV[1] then
        return redis.call('del', KEYS[1])
    else
        return 0
    end
    """
    
    def __init__(
        self,
        client: Any,  # redis.Redis
        name: str,
        ttl_ms: int,
        wait_ms: int,
        retry_ms: int
    ):
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
                # Try to acquire lock
                acquired = self.client.set(
                    self.name,
                    self.token,
                    nx=True,
                    px=self.ttl_ms
                )
                
                if acquired:
                    self._acquired = True
                    LOGGER.debug(f"Lock acquired: {self.name}")
                    break
            except Exception as e:
                LOGGER.warning(f"Lock acquisition failed: {e}")
                break
            
            # Wait before retry
            time.sleep(self.retry_ms / 1000.0)
        
        if not self._acquired:
            LOGGER.warning(f"Failed to acquire lock: {self.name}")
        
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self._acquired:
            return
        
        try:
            # Use Lua script for atomic unlock
            self.client.eval(
                self.UNLOCK_SCRIPT,
                1,
                self.name,
                self.token
            )
            LOGGER.debug(f"Lock released: {self.name}")
        except Exception as e:
            LOGGER.warning(f"Lock release failed: {e}")
            
            # Fallback: try simple delete
            try:
                self.client.delete(self.name)
            except Exception:
                pass
        finally:
            self._acquired = False


# ========================================================================================
# CACHE IMPLEMENTATIONS
# ========================================================================================

class NoopCache:
    """No-op cache implementation (disabled state)."""
    
    def __init__(self, namespace: str = DEFAULT_NAMESPACE):
        self.enabled = False
        self.namespace = namespace
        self.metrics = CacheMetrics()
    
    def get(self, key: str, default: Any = None) -> Any:
        return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        return False
    
    def delete(self, key: str) -> int:
        return 0
    
    def exists(self, key: str) -> bool:
        return False
    
    def expire(self, key: str, ttl: int) -> bool:
        return False
    
    def ttl(self, key: str) -> int:
        return -2
    
    def incr(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        return 0
    
    def decr(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        return 0
    
    def flush(self, pattern: Optional[str] = None) -> int:
        return 0
    
    def lock(
        self,
        name: str,
        ttl_ms: Optional[int] = None,
        wait_ms: Optional[int] = None
    ) -> BaseLock:
        return NoopLock()
    
    def status(self) -> str:
        return (
            "❌ Cache disabled\n"
            "To enable: set REDIS_URL environment variable or cache.enabled=true in config.yaml"
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "enabled": False,
            "message": "Cache is disabled"
        }
    
    def close(self) -> None:
        pass


class RedisCache:
    """
    Production-ready Redis cache implementation.
    
    Features:
    - Connection pooling
    - Automatic reconnection
    - Metrics tracking
    - Error handling with graceful degradation
    - Thread-safe operations
    """
    
    def __init__(self, config: CacheConfig):
        if not HAS_REDIS or redis is None:
            raise RuntimeError("redis-py is not installed")
        
        if not config.redis_url:
            raise ValueError("Redis URL is required")
        
        self.config = config
        self.enabled = True
        self.namespace = config.namespace
        self.metrics = CacheMetrics() if config.enable_metrics else None
        self._lock = threading.Lock()
        
        # Create Redis client with connection pool
        try:
            self.client = redis.Redis.from_url(
                config.redis_url,
                decode_responses=False,
                socket_timeout=config.socket_timeout,
                socket_connect_timeout=config.socket_connect_timeout,
                health_check_interval=config.health_check_interval,
                max_connections=config.max_connections,
                retry_on_timeout=True,
                retry_on_error=[redis.exceptions.ConnectionError, redis.exceptions.TimeoutError]
            )
            
            # Test connection
            self.client.ping()
            
            LOGGER.info(f"✅ Redis cache initialized: {config.namespace}")
            
        except Exception as e:
            LOGGER.error(f"Failed to initialize Redis: {e}")
            raise
    
    def _record_metric(self, operation: str, duration: float = 0.0) -> None:
        """Record operation metric."""
        if self.metrics is None:
            return
        
        with self._lock:
            if operation == "hit":
                self.metrics.hits += 1
                self.metrics.total_get_time += duration
            elif operation == "miss":
                self.metrics.misses += 1
                self.metrics.total_get_time += duration
            elif operation == "set":
                self.metrics.sets += 1
                self.metrics.total_set_time += duration
            elif operation == "delete":
                self.metrics.deletes += 1
            elif operation == "error":
                self.metrics.errors += 1
            elif operation == "lock_acquired":
                self.metrics.lock_acquisitions += 1
            elif operation == "lock_failed":
                self.metrics.lock_failures += 1
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        qualified_key = build_namespaced_key(self.namespace, key)
        
        start = time.time()
        
        try:
            data = self.client.get(qualified_key)
            
            if data is None:
                self._record_metric("miss", time.time() - start)
                return default
            
            value = deserialize_value(data)
            
            if value is None:
                self._record_metric("miss", time.time() - start)
                return default
            
            self._record_metric("hit", time.time() -start)
            return value
            
        except Exception as e:
            LOGGER.error(f"Cache GET error for key '{key}': {e}")
            self._record_metric("error")
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        qualified_key = build_namespaced_key(self.namespace, key)
        
        start = time.time()
        
        try:
            # Serialize value
            serialized = serialize_value(value, self.config)
            
            # Check size limit
            if len(serialized) > MAX_VALUE_SIZE:
                LOGGER.warning(
                    f"Value too large for cache: {len(serialized)} bytes "
                    f"(max: {MAX_VALUE_SIZE})"
                )
                return False
            
            # Set with TTL
            expiration = ttl if ttl is not None else self.config.default_ttl
            
            result = self.client.set(qualified_key, serialized, ex=expiration)
            
            self._record_metric("set", time.time() - start)
            
            return bool(result)
            
        except Exception as e:
            LOGGER.error(f"Cache SET error for key '{key}': {e}")
            self._record_metric("error")
            return False
    
    def delete(self, key: str) -> int:
        """Delete key from cache."""
        qualified_key = build_namespaced_key(self.namespace, key)
        
        try:
            result = self.client.delete(qualified_key)
            self._record_metric("delete")
            return int(result)
            
        except Exception as e:
            LOGGER.error(f"Cache DELETE error for key '{key}': {e}")
            self._record_metric("error")
            return 0
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        qualified_key = build_namespaced_key(self.namespace, key)
        
        try:
            return bool(self.client.exists(qualified_key))
        except Exception as e:
            LOGGER.error(f"Cache EXISTS error for key '{key}': {e}")
            self._record_metric("error")
            return False
    
    def expire(self, key: str, ttl: int) -> bool:
        """Set expiration on key."""
        qualified_key = build_namespaced_key(self.namespace, key)
        
        try:
            return bool(self.client.expire(qualified_key, ttl))
        except Exception as e:
            LOGGER.error(f"Cache EXPIRE error for key '{key}': {e}")
            self._record_metric("error")
            return False
    
    def ttl(self, key: str) -> int:
        """Get remaining TTL for key."""
        qualified_key = build_namespaced_key(self.namespace, key)
        
        try:
            return int(self.client.ttl(qualified_key))
        except Exception as e:
            LOGGER.error(f"Cache TTL error for key '{key}': {e}")
            self._record_metric("error")
            return -2
    
    def incr(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """Increment counter."""
        qualified_key = build_namespaced_key(self.namespace, key)
        
        try:
            value = int(self.client.incrby(qualified_key, amount))
            
            if ttl is not None:
                self.client.expire(qualified_key, ttl)
            
            return value
            
        except Exception as e:
            LOGGER.error(f"Cache INCR error for key '{key}': {e}")
            self._record_metric("error")
            return 0
    
    def decr(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """Decrement counter."""
        qualified_key = build_namespaced_key(self.namespace, key)
        
        try:
            value = int(self.client.decrby(qualified_key, amount))
            
            if ttl is not None:
                self.client.expire(qualified_key, ttl)
            
            return value
            
        except Exception as e:
            LOGGER.error(f"Cache DECR error for key '{key}': {e}")
            self._record_metric("error")
            return 0
    
    def flush(self, pattern: Optional[str] = None) -> int:
        """
        Flush keys matching pattern in namespace.
        
        Args:
            pattern: Optional pattern (wildcards allowed)
            
        Returns:
            Number of keys deleted
        """
        try:
            prefix = build_namespaced_key(self.namespace, "")
            match_pattern = f"{prefix}{pattern or '*'}"
            
            deleted = 0
            cursor = 0
            
            while True:
                cursor, keys = self.client.scan(
                    cursor=cursor,
                    match=match_pattern,
                    count=100
                )
                
                if keys:
                    deleted += self.client.delete(*keys)
                
                if cursor == 0:
                    break
            
            LOGGER.info(f"Flushed {deleted} keys matching '{match_pattern}'")
            return deleted
            
        except Exception as e:
            LOGGER.error(f"Cache FLUSH error: {e}")
            self._record_metric("error")
            return 0
    
    def lock(
        self,
        name: str,
        ttl_ms: Optional[int] = None,
        wait_ms: Optional[int] = None
    ) -> BaseLock:
        """
        Acquire distributed lock.
        
        Args:
            name: Lock name
            ttl_ms: Lock TTL in milliseconds
            wait_ms: Max wait time in milliseconds
            
        Returns:
            Lock context manager
        """
        lock_key = build_namespaced_key(self.namespace, f"lock:{name}")
        
        lock_ttl = ttl_ms if ttl_ms is not None else self.config.lock_ttl_ms
        lock_wait = wait_ms if wait_ms is not None else self.config.lock_wait_ms
        
        return RedisLock(
            client=self.client,
            name=lock_key,
            ttl_ms=lock_ttl,
            wait_ms=lock_wait,
            retry_ms=self.config.lock_retry_ms
        )
    
    def status(self) -> str:
        """Get cache status."""
        try:
            start = time.time()
            pong = self.client.ping()
            ping_ms = (time.time() - start) * 1000
            
            # Get server info
            info = self.client.info(section="memory")
            used_memory = info.get("used_memory_human", "N/A")
            max_memory = info.get("maxmemory_human", "N/A")
            
            # Build status string
            status_lines = [
                f"✅ Cache enabled",
                f"Namespace: {self.namespace}",
                f"Ping: {ping_ms:.1f}ms",
                f"Memory: {used_memory}",
            ]
            
            if max_memory and max_memory != "N/A" and max_memory != "0B":
                status_lines.append(f"Max memory: {max_memory}")
            
            # Add metrics if enabled
            if self.metrics:
                metrics = self.metrics.to_dict()
                status_lines.extend([
                    f"Hit rate: {metrics['hit_rate']:.1f}%",
                    f"Operations: {metrics['hits'] + metrics['misses']} gets, {metrics['sets']} sets",
                ])
            
            return "\n".join(status_lines)
            
        except Exception as e:
            return f"❌ Cache error: {e}"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        if self.metrics is None:
            return {"enabled": True, "metrics_disabled": True}
        
        return {
            "enabled": True,
            "namespace": self.namespace,
            **self.metrics.to_dict()
        }
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        if self.metrics:
            with self._lock:
                self.metrics.reset()
            LOGGER.info("Cache metrics reset")
    
    def close(self) -> None:
        """Close Redis connection."""
        try:
            self.client.close()
            LOGGER.info("Redis connection closed")
        except Exception as e:
            LOGGER.warning(f"Error closing Redis connection: {e}")


# ========================================================================================
# SINGLETON INSTANCE
# ========================================================================================

_CACHE: Optional[Cache] = None
_CACHE_LOCK = threading.Lock()


def _initialize_cache(force: bool = False) -> Cache:
    """
    Initialize cache singleton.
    
    Args:
        force: Force reinitialization
        
    Returns:
        Cache instance
    """
    global _CACHE
    
    with _CACHE_LOCK:
        if _CACHE is not None and not force:
            return _CACHE
        
        config = _load_cache_config()
        
        # Check if cache is disabled
        if not config.enabled:
            LOGGER.info("Cache disabled by configuration")
            _CACHE = NoopCache(namespace=config.namespace)
            return _CACHE
        
        # Check if Redis URL is provided
        if not config.redis_url:
            LOGGER.warning("Cache enabled but no REDIS_URL provided, cache disabled")
            _CACHE = NoopCache(namespace=config.namespace)
            return _CACHE
        
        # Try to initialize Redis cache
        try:
            _CACHE = RedisCache(config)
            LOGGER.info(f"✅ Cache initialized successfully: {config.namespace}")
            return _CACHE
            
        except Exception as e:
            LOGGER.error(f"Failed to initialize Redis cache: {e}")
            LOGGER.info("Falling back to NoopCache")
            _CACHE = NoopCache(namespace=config.namespace)
            return _CACHE


def get_cache(force_reload: bool = False) -> Cache:
    """
    Get cache singleton instance.
    
    Args:
        force_reload: Force cache reinitialization
        
    Returns:
        Cache instance (Redis or Noop)
        
    Examples:
        >>> cache = get_cache()
        >>> cache.set("key", "value", ttl=3600)
        >>> cache.get("key")
        'value'
    """
    global _CACHE
    
    if _CACHE is None or force_reload:
        return _initialize_cache(force=force_reload)
    
    return _CACHE


# ========================================================================================
# DECORATOR
# ========================================================================================

def _default_key_builder(
    func: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any]
) -> str:
    """
    Default key builder for cacheable decorator.
    
    Args:
        func: Function being cached
        args: Positional arguments
        kwargs: Keyword arguments
        
    Returns:
        Cache key
    """
    base = f"{func.__module__}.{func.__qualname__}"
    args_hash = hash_object(args, kwargs)
    return f"{base}:{args_hash}"


def cacheable(
    namespace: Optional[str] = None,
    ttl: Optional[int] = None,
    key_builder: Optional[Callable[[Callable, Tuple, Dict], str]] = None,
    *,
    lock_ms: int = 0,
    grace_ttl: int = 0,
    condition: Optional[Callable[..., bool]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for caching function results.
    
    Args:
        namespace: Sub-namespace for cache keys
        ttl: Time-to-live in seconds
        key_builder: Custom key builder function
        lock_ms: Lock timeout for anti-stampede protection (0 = disabled)
        grace_ttl: Grace period for stale data (0 = disabled)
        condition: Optional condition function (args, kwargs) -> bool
        
    Returns:
        Decorated function
        
    Examples:
        >>> @cacheable(ttl=3600)
        ... def expensive_computation(x, y):
        ...     return x + y
        
        >>> @cacheable(namespace="api", lock_ms=5000, grace_ttl=86400)
        ... def fetch_api_data(endpoint):
        ...     return requests.get(endpoint).json()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Check condition
            if condition is not None:
                try:
                    if not condition(*args, **kwargs):
                        # Condition not met, skip cache
                        return func(*args, **kwargs)
                except Exception as e:
                    LOGGER.warning(f"Cache condition check failed: {e}")
                    return func(*args, **kwargs)
            
            cache = get_cache()
            
            # Build cache key
            kb = key_builder or _default_key_builder
            local_key = kb(func, args, kwargs)
            
            # Build full key with namespace
            if namespace:
                full_ns = f"{cache.namespace}:{namespace}"
            else:
                full_ns = cache.namespace
            
            cache_key = f"{full_ns}:{local_key}"
            stale_key = f"{cache_key}:stale" if grace_ttl > 0 else None
            
            # Fast path: cache hit
            if cache.enabled:
                cached = cache.get(cache_key)
                
                if cached is not None:
                    LOGGER.debug(f"Cache HIT: {local_key[:50]}")
                    return cached  # type: ignore
                
                # Try stale data if available
                if stale_key:
                    stale = cache.get(stale_key)
                    
                    if stale is not None:
                        LOGGER.debug(f"Cache STALE HIT: {local_key[:50]}")
                        # Return stale data immediately, could refresh in background
                        return stale  # type: ignore
            
            # Cache miss - compute result
            LOGGER.debug(f"Cache MISS: {local_key[:50]}")
            
            # Anti-stampede protection with lock
            if cache.enabled and lock_ms > 0:
                lock_name = f"func:{local_key}"
                
                with cache.lock(name=lock_name, ttl_ms=lock_ms, wait_ms=lock_ms) as lock:
                    # Double-check after acquiring lock
                    if lock.acquired and cache.enabled:
                        cached = cache.get(cache_key)
                        
                        if cached is not None:
                            LOGGER.debug(f"Cache HIT after lock: {local_key[:50]}")
                            return cached  # type: ignore
                    
                    # Compute result
                    result = func(*args, **kwargs)
                    
                    # Store in cache
                    if cache.enabled:
                        cache.set(cache_key, result, ttl=ttl)
                        
                        # Store stale copy
                        if stale_key:
                            cache.set(stale_key, result, ttl=grace_ttl)
                    
                    return result
            
            # No lock - direct computation
            result = func(*args, **kwargs)
            
            # Store in cache
            if cache.enabled:
                cache.set(cache_key, result, ttl=ttl)
                
                # Store stale copy
                if stale_key:
                    cache.set(stale_key, result, ttl=grace_ttl)
            
            return result
        
        # Add cache management methods to wrapper
        wrapper.__cache_clear__ = lambda: get_cache().flush(pattern=f"{func.__qualname__}:*")  # type: ignore
        wrapper.__cache_info__ = lambda: get_cache().get_metrics()  # type: ignore
        
        return wrapper
    
    return decorator


# ========================================================================================
# CONVENIENCE FUNCTIONS
# ========================================================================================

def status() -> str:
    """
    Get cache status string.
    
    Returns:
        Status string
        
    Examples:
        >>> print(status())
        ✅ Cache enabled
        Namespace: intelligent-predictor
        ...
    """
    return get_cache().status()


def get_metrics() -> Dict[str, Any]:
    """
    Get cache performance metrics.
    
    Returns:
        Metrics dictionary
        
    Examples:
        >>> metrics = get_metrics()
        >>> print(f"Hit rate: {metrics['hit_rate']:.1f}%")
    """
    return get_cache().get_metrics()


def flush_all(pattern: Optional[str] = None) -> int:
    """
    Flush cache keys matching pattern.
    
    Args:
        pattern: Optional pattern (wildcards allowed)
        
    Returns:
        Number of keys deleted
        
    Examples:
        >>> flush_all("user:*")  # Delete all user keys
        >>> flush_all()  # Delete all keys in namespace
    """
    return get_cache().flush(pattern)


def reset_metrics() -> None:
    """
    Reset performance metrics.
    
    Examples:
        >>> reset_metrics()
    """
    cache = get_cache()
    
    if isinstance(cache, RedisCache):
        cache.reset_metrics()


def close_cache() -> None:
    """
    Close cache connection.
    
    Examples:
        >>> close_cache()
    """
    global _CACHE
    
    if _CACHE is not None:
        _CACHE.close()
        _CACHE = None


@contextmanager
def temporary_cache(config: Optional[CacheConfig] = None):
    """
    Context manager for temporary cache instance.
    
    Args:
        config: Optional cache configuration
        
    Yields:
        Temporary cache instance
        
    Examples:
        >>> with temporary_cache() as cache:
        ...     cache.set("key", "value")
        ...     print(cache.get("key"))
    """
    if config is None:
        config = _load_cache_config()
    
    temp_cache: Cache
    
    if config.enabled and config.redis_url:
        try:
            temp_cache = RedisCache(config)
        except Exception as e:
            LOGGER.warning(f"Failed to create temporary cache: {e}")
            temp_cache = NoopCache(namespace=config.namespace)
    else:
        temp_cache = NoopCache(namespace=config.namespace)
    
    try:
        yield temp_cache
    finally:
        temp_cache.close()


# ========================================================================================
# TESTING & DIAGNOSTICS
# ========================================================================================

def test_cache(verbose: bool = True) -> Dict[str, bool]:
    """
    Test cache functionality.
    
    Args:
        verbose: Print detailed results
        
    Returns:
        Test results
        
    Examples:
        >>> results = test_cache(verbose=False)
        >>> all(results.values())
        True
    """
    results = {}
    cache = get_cache()
    
    # Test 1: Basic set/get
    try:
        test_key = f"test:{uuid.uuid4().hex}"
        test_value = {"data": "test", "number": 42}
        
        cache.set(test_key, test_value, ttl=60)
        retrieved = cache.get(test_key)
        
        results["set_get"] = retrieved == test_value
        
        if verbose:
            print(f"✅ Set/Get: {results['set_get']}")
        
        # Cleanup
        cache.delete(test_key)
        
    except Exception as e:
        results["set_get"] = False
        if verbose:
            print(f"❌ Set/Get: {e}")
    
    # Test 2: TTL
    try:
        test_key = f"test:{uuid.uuid4().hex}"
        
        cache.set(test_key, "value", ttl=2)
        ttl_value = cache.ttl(test_key)
        
        results["ttl"] = 0 < ttl_value <= 2
        
        if verbose:
            print(f"✅ TTL: {results['ttl']} (ttl={ttl_value}s)")
        
        # Cleanup
        cache.delete(test_key)
        
    except Exception as e:
        results["ttl"] = False
        if verbose:
            print(f"❌ TTL: {e}")
    
    # Test 3: Delete
    try:
        test_key = f"test:{uuid.uuid4().hex}"
        
        cache.set(test_key, "value", ttl=60)
        deleted = cache.delete(test_key)
        exists = cache.exists(test_key)
        
        results["delete"] = deleted > 0 and not exists
        
        if verbose:
            print(f"✅ Delete: {results['delete']}")
        
    except Exception as e:
        results["delete"] = False
        if verbose:
            print(f"❌ Delete: {e}")
    
    # Test 4: Increment
    try:
        test_key = f"test:{uuid.uuid4().hex}"
        
        val1 = cache.incr(test_key, amount=5, ttl=60)
        val2 = cache.incr(test_key, amount=3)
        
        results["incr"] = val1 == 5 and val2 == 8
        
        if verbose:
            print(f"✅ Increment: {results['incr']} ({val1}, {val2})")
        
        # Cleanup
        cache.delete(test_key)
        
    except Exception as e:
        results["incr"] = False
        if verbose:
            print(f"❌ Increment: {e}")
    
    # Test 5: Lock
    try:
        lock_name = f"test_lock:{uuid.uuid4().hex}"
        
        with cache.lock(lock_name, ttl_ms=5000, wait_ms=1000) as lock:
            results["lock"] = lock.acquired
        
        if verbose:
            print(f"✅ Lock: {results['lock']}")
        
    except Exception as e:
        results["lock"] = False
        if verbose:
            print(f"❌ Lock: {e}")
    
    return results


def print_diagnostics() -> None:
    """
    Print comprehensive cache diagnostics.
    
    Examples:
        >>> print_diagnostics()
        📊 Cache Diagnostics
        ...
    """
    cache = get_cache()
    
    print("📊 Cache Diagnostics\n")
    print("="*60)
    
    # Status
    print("\n🔧 Status:")
    print(cache.status())
    
    # Configuration
    print("\n⚙️ Configuration:")
    config = _load_cache_config()
    print(f"  Enabled: {config.enabled}")
    print(f"  Namespace: {config.namespace}")
    print(f"  Default TTL: {config.default_ttl}s")
    print(f"  Compression: {config.compression}")
    print(f"  Grace TTL: {config.grace_ttl}s")
    
    # Metrics
    print("\n📈 Metrics:")
    metrics = cache.get_metrics()
    
    if "enabled" in metrics and not metrics.get("metrics_disabled"):
        for key, value in metrics.items():
            if key not in ("enabled", "namespace"):
                print(f"  {key}: {value}")
    else:
        print("  Metrics disabled or unavailable")
    
    # Dependencies
    print("\n📦 Dependencies:")
    print(f"  redis-py: {'✅' if HAS_REDIS else '❌'}")
    print(f"  PyYAML: {'✅' if HAS_YAML else '❌'}")
    print(f"  Streamlit: {'✅' if HAS_STREAMLIT else '❌'}")
    
    print("\n" + "="*60)


# ========================================================================================
# MAIN
# ========================================================================================

if __name__ == "__main__":
    print_diagnostics()
    print("\n🧪 Running tests...\n")
    test_results = test_cache(verbose=True)
    print(f"\n✅ Passed: {sum(test_results.values())}/{len(test_results)}")