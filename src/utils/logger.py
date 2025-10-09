"""
Advanced Logging System - Centralizowane logowanie z Loguru.

Funkcjonalności:
- Loguru jako backend (lepsze od stdlib logging)
- Multiple sinks: console, file, JSON, memory buffer
- Configuration z YAML/environment variables
- Automatic log rotation i retention
- Intercept stdlib logging (uvicorn, prophet, etc.)
- Memory buffer dla quick access (Streamlit)
- Context management
- Exception logging decorator
- Colored console output
- Structured JSON logging
"""

from __future__ import annotations

import os
import sys
import json
import pathlib
import logging
import warnings
from typing import Any, Optional, Iterable, Dict, Callable
from dataclasses import dataclass, field
from collections import deque
from functools import wraps

# Optional YAML support
try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None
    HAS_YAML = False

from loguru import logger as _loguru_logger

# ========================================================================================
# KONFIGURACJA
# ========================================================================================

# Paths
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    # Levels
    "level": "INFO",
    "console_level": None,  # If None, use 'level'
    "file_level": None,     # If None, use 'level'
    
    # File settings
    "rotation": "10 MB",
    "retention": "14 days",
    "compression": None,  # Can be "zip", "gz", "bz2", "xz"
    
    # Filenames
    "log_filename": "app.log",
    "json_filename": "app.jsonl",
    
    # JSON logging
    "serialize_json": True,
    
    # App identification
    "app_name": "intelligent-predictor",
    
    # Advanced
    "backtrace": False,
    "diagnose": False,
    "enqueue": True,
    
    # Memory buffer
    "memory_buffer": 2000,
    "memory_format": "detailed",  # "simple" or "detailed"
}

# Environment variable mapping
ENV_VAR_MAPPING = {
    "level": "LOG_LEVEL",
    "console_level": "LOG_CONSOLE_LEVEL",
    "file_level": "LOG_FILE_LEVEL",
    "rotation": "LOG_ROTATION",
    "retention": "LOG_RETENTION",
    "compression": "LOG_COMPRESSION",
    "log_filename": "LOG_FILE",
    "json_filename": "LOG_JSON_FILENAME",
    "serialize_json": "LOG_JSON",
    "app_name": "APP_NAME",
    "backtrace": "LOG_BACKTRACE",
    "diagnose": "LOG_DIAGNOSE",
    "enqueue": "LOG_ENQUEUE",
    "memory_buffer": "LOG_MEMORY_BUFFER",
    "memory_format": "LOG_MEMORY_FORMAT",
}

# Loggers to silence by default
NOISY_LOGGERS = (
    "uvicorn", "uvicorn.error", "uvicorn.access",
    "asyncio", "matplotlib", "prophet", "fbprophet",
    "numexpr", "PIL", "urllib3", "botocore", "s3transfer",
    "chromadb", "pinecone", "lightgbm", "xgboost",
    "cmdstanpy", "httpx", "charset_normalizer"
)

# Console format (compact for better readability)
CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <5}</level> | "
    "<cyan>{extra[mod]}</cyan> | "
    "{message}"
)

# File format (detailed for debugging)
FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "{extra[mod]} | "
    "{message}"
)

# Simple console format (no module)
CONSOLE_FORMAT_SIMPLE = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <5}</level> | "
    "{message}"
)


# ========================================================================================
# CONFIGURATION LOADER
# ========================================================================================

def _load_yaml_config() -> Dict[str, Any]:
    """
    Wczytuje konfigurację z YAML.
    
    Returns:
        Słownik z konfiguracją lub pusty dict
    """
    if not HAS_YAML:
        return {}
    
    config_path = PROJECT_ROOT / "config.yaml"
    
    if not config_path.exists():
        return {}
    
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        logging_section = raw.get("logging", {})
        
        if not isinstance(logging_section, dict):
            return {}
        
        return logging_section
        
    except Exception as e:
        print(f"Warning: Failed to load YAML config: {e}", file=sys.stderr)
        return {}


def _load_env_config() -> Dict[str, Any]:
    """
    Wczytuje konfigurację z environment variables.
    
    Returns:
        Słownik z konfiguracją
    """
    config = {}
    
    for key, env_var in ENV_VAR_MAPPING.items():
        value = os.getenv(env_var)
        
        if value is None:
            continue
        
        # Type conversion based on default
        default_value = DEFAULT_CONFIG.get(key)
        
        if isinstance(default_value, bool):
            config[key] = value.lower() in ("1", "true", "yes", "on")
        elif isinstance(default_value, int):
            try:
                config[key] = int(value)
            except ValueError:
                pass
        else:
            config[key] = value
    
    return config


def _merge_configs() -> Dict[str, Any]:
    """
    Łączy konfiguracje (defaults → YAML → env).
    
    Returns:
        Finalna konfiguracja
    """
    config = dict(DEFAULT_CONFIG)
    
    # YAML override
    yaml_config = _load_yaml_config()
    config.update(yaml_config)
    
    # Environment override
    env_config = _load_env_config()
    config.update(env_config)
    
    # Post-processing
    if not config["console_level"]:
        config["console_level"] = config["level"]
    
    if not config["file_level"]:
        config["file_level"] = config["level"]
    
    return config


# ========================================================================================
# MEMORY SINK
# ========================================================================================

@dataclass
class MemorySink:
    """
    In-memory log buffer dla quick access.
    
    Używane w Streamlit do pokazywania ostatnich logów.
    """
    maxlen: int = DEFAULT_CONFIG["memory_buffer"]
    buffer: deque = field(default_factory=deque)
    
    def __post_init__(self):
        """Initialize buffer with maxlen."""
        self.buffer = deque(maxlen=self.maxlen)
    
    def write(self, message: str) -> None:
        """
        Zapisuje message do bufora.
        
        Args:
            message: Log message
        """
        # Zapewnij że message kończy się newline
        msg = message if message.endswith("\n") else message + "\n"
        self.buffer.append(msg)
    
    def dump(self, n: Optional[int] = None) -> str:
        """
        Zwraca logi jako string.
        
        Args:
            n: Liczba ostatnich logów (None = wszystkie)
            
        Returns:
            Logi jako string
        """
        if not self.buffer:
            return ""
        
        if n is None or n >= len(self.buffer):
            return "".join(list(self.buffer))
        
        return "".join(list(self.buffer)[-n:])
    
    def lines(self, n: Optional[int] = None) -> list[str]:
        """
        Zwraca logi jako listę linii.
        
        Args:
            n: Liczba ostatnich logów (None = wszystkie)
            
        Returns:
            Lista logów
        """
        if not self.buffer:
            return []
        
        if n is None or n >= len(self.buffer):
            return list(self.buffer)
        
        return list(self.buffer)[-n:]
    
    def clear(self) -> None:
        """Czyści buffer."""
        self.buffer.clear()
    
    def size(self) -> int:
        """Zwraca liczbę logów w buforze."""
        return len(self.buffer)


# Global memory sink instance
_MEMORY_SINK = MemorySink(DEFAULT_CONFIG["memory_buffer"])


# ========================================================================================
# STDLIB LOGGING INTERCEPT
# ========================================================================================

class InterceptHandler(logging.Handler):
    """
    Handler interceptujący stdlib logging i przekierowujący do Loguru.
    
    Pozwala na jednolite logowanie z bibliotek używających stdlib logging.
    """
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Emituje log record do Loguru.
        
        Args:
            record: Stdlib log record
        """
        # Get corresponding Loguru level
        try:
            level = _loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        # Find caller frame
        frame = logging.currentframe()
        depth = 2
        
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        # Log to Loguru
        _loguru_logger.opt(
            depth=depth,
            exception=record.exc_info
        ).log(level, record.getMessage())


def patch_stdlib_logging(level: str = "INFO") -> None:
    """
    Patchuje stdlib logging żeby używał Loguru.
    
    Args:
        level: Minimum log level
    """
    # Replace root handler
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(level)
    
    # Patch known noisy loggers
    for logger_name in NOISY_LOGGERS:
        stdlib_logger = logging.getLogger(logger_name)
        stdlib_logger.handlers = [InterceptHandler()]
        stdlib_logger.setLevel(level)
    
    # Capture warnings
    warnings.simplefilter("default")
    logging.captureWarnings(True)


def silence_loggers(
    logger_names: Iterable[str],
    level: str = "WARNING"
) -> None:
    """
    Wycisza specificzne loggery.
    
    Args:
        logger_names: Lista nazw loggerów
        level: Minimalny poziom dla tych loggerów
    """
    for name in logger_names:
        # Disable w Loguru
        try:
            _loguru_logger.disable(name)
        except Exception:
            pass
        
        # Set level w stdlib
        logging.getLogger(name).setLevel(level)


# ========================================================================================
# LOGGER CONFIGURATION
# ========================================================================================

_CONFIGURED = False


def _norm_level(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().upper()
    return v if v in _VALID_LEVELS else None

def configure_logger(
    *,
    force: bool = False,
    level: Optional[str] = None,
    console_level: Optional[str] = None,
    file_level: Optional[str] = None,
) -> Any:
    """
    Konfiguruje system logowania (loguru) z opcjonalnym nadpisaniem poziomów.
    
    Args:
        force:     Wymuś rekonfigurację nawet jeśli już skonfigurowano.
        level:     Globalne nadpisanie poziomu (konsola+plik+patch stdlib).
        console_level: Nadpisanie tylko poziomu konsoli.
        file_level:    Nadpisanie tylko poziomu pliku/JSON.
    
    Returns:
        loguru.logger
    """
    global _CONFIGURED

    if _CONFIGURED and not force:
        return _loguru_logger

    # 1) Bazowa konfiguracja z plików/env
    config = _merge_configs()

    # 2) Normalizacja i nadpisania z parametrów
    lvl_all   = _norm_level(level)
    lvl_cons  = _norm_level(console_level)
    lvl_file  = _norm_level(file_level)

    if lvl_all:
        # Jednym parametrem ustawiamy spójnie wszystko
        config["level"] = lvl_all
        config["console_level"] = lvl_all
        config["file_level"] = lvl_all

    if lvl_cons:
        config["console_level"] = lvl_cons
    if lvl_file:
        config["file_level"] = lvl_file

    # 3) (Re)inicjalizacja handlerów
    _loguru_logger.remove()

    _loguru_logger.add(
        sys.stderr,
        level=config["console_level"],
        format=CONSOLE_FORMAT,
        colorize=True,
        backtrace=config["backtrace"],
        diagnose=config["diagnose"],
        enqueue=config["enqueue"],
        catch=True,
    )

    log_file = LOG_DIR / config["log_filename"]
    _loguru_logger.add(
        log_file,
        level=config["file_level"],
        format=FILE_FORMAT,
        rotation=config["rotation"],
        retention=config["retention"],
        compression=config["compression"],
        backtrace=False,
        diagnose=False,
        enqueue=config["enqueue"],
        catch=True,
    )

    if config.get("serialize_json", False):
        json_file = LOG_DIR / config["json_filename"]
        _loguru_logger.add(
            json_file,
            level=config["file_level"],
            serialize=True,
            rotation=config["rotation"],
            retention=config["retention"],
            compression=config["compression"],
            backtrace=False,
            diagnose=False,
            enqueue=config["enqueue"],
            catch=True,
        )

    _loguru_logger.add(
        _MEMORY_SINK.write,
        level=config["console_level"],
        format=FILE_FORMAT if config["memory_format"] == "detailed" else CONSOLE_FORMAT_SIMPLE,
        backtrace=False,
        diagnose=False,
        enqueue=False,
        catch=True,
    )

    # 4) Patch stdlib logging (użyj globalnego pola 'level' po nadpisaniach)
    patch_stdlib_logging(config["level"])

    # 5) Wyciszenia i kontekst
    silence_loggers(NOISY_LOGGERS, level="WARNING")
    _loguru_logger.configure(extra={"mod": config["app_name"]})

    _CONFIGURED = True
    return _loguru_logger


# Auto-configure on import
configure_logger()

# Main logger instance
logger = _loguru_logger


# ========================================================================================
# PUBLIC API
# ========================================================================================

def get_logger(module: Optional[str] = None, **context: Any):
    """
    Zwraca logger z kontekstem.
    
    Args:
        module: Nazwa modułu (dla "mod" field)
        **context: Dodatkowy kontekst
        
    Returns:
        Bound logger
        
    Example:
        >>> log = get_logger("my_module")
        >>> log.info("Hello from my module")
    """
    if module:
        context = {"mod": module, **context}
    else:
        # Default mod if not provided
        if "mod" not in context:
            context["mod"] = "app"
    
    return logger.bind(**context)


def set_level(level: str) -> None:
    """
    Zmienia globalny poziom logowania.
    
    Args:
        level: Nowy poziom (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Example:
        >>> set_level("DEBUG")
    """
    os.environ["LOG_LEVEL"] = level.upper()
    configure_logger(force=True)


def get_memory_logs(n: Optional[int] = None) -> list[str]:
    """
    Zwraca logi z memory buffer.
    
    Args:
        n: Liczba ostatnich logów (None = wszystkie)
        
    Returns:
        Lista logów
        
    Example:
        >>> logs = get_memory_logs(100)
        >>> for log in logs:
        ...     print(log)
    """
    return _MEMORY_SINK.lines(n)


def clear_memory_logs() -> None:
    """
    Czyści memory buffer.
    
    Example:
        >>> clear_memory_logs()
    """
    _MEMORY_SINK.clear()


def get_memory_logs_text(n: Optional[int] = None) -> str:
    """
    Zwraca logi z memory buffer jako string.
    
    Args:
        n: Liczba ostatnich logów (None = wszystkie)
        
    Returns:
        Logi jako string
    """
    return _MEMORY_SINK.dump(n)


def memory_buffer_size() -> int:
    """
    Zwraca liczbę logów w memory buffer.
    
    Returns:
        Liczba logów
    """
    return _MEMORY_SINK.size()


# ========================================================================================
# CONTEXT MANAGERS
# ========================================================================================

class LogContext:
    """Context manager dla temporary context w logach."""
    
    def __init__(self, **context: Any):
        """
        Initialize context manager.
        
        Args:
            **context: Kontekst do dodania
        """
        self.context = context
        self._bound_logger = None
        self._original_logger = None
    
    def __enter__(self):
        """Enter context."""
        self._bound_logger = logger.bind(**self.context)
        self._original_logger = globals().get("logger")
        globals()["logger"] = self._bound_logger
        return self._bound_logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if self._original_logger is not None:
            globals()["logger"] = self._original_logger
        return False


def with_context(**context: Any) -> LogContext:
    """
    Creates context manager dla temporary logging context.
    
    Args:
        **context: Kontekst do dodania
        
    Returns:
        LogContext instance
        
    Example:
        >>> with with_context(user_id="123", action="login"):
        ...     logger.info("User action")
    """
    return LogContext(**context)


# ========================================================================================
# DECORATORS
# ========================================================================================

def log_exception(
    message: str = "Unhandled exception",
    level: str = "ERROR",
    reraise: bool = True,
    **extra_context: Any
) -> Callable:
    """
    Decorator do logowania wyjątków.
    
    Args:
        message: Message do zalogowania
        level: Log level (default: ERROR)
        reraise: Czy re-raise exception (default: True)
        **extra_context: Dodatkowy kontekst
        
    Returns:
        Decorated function
        
    Example:
        >>> @log_exception("Failed to process data", user_id="123")
        ... def process_data(data):
        ...     return data / 0  # Will log and re-raise
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log with context
                bound_logger = logger.bind(**extra_context) if extra_context else logger
                
                # Log exception with traceback
                bound_logger.opt(exception=True).log(level, f"{message}: {e}")
                
                # Re-raise if requested
                if reraise:
                    raise
                
                return None
        
        return wrapper
    return decorator


def log_call(
    level: str = "DEBUG",
    log_args: bool = False,
    log_result: bool = False,
    **extra_context: Any
) -> Callable:
    """
    Decorator do logowania wywołań funkcji.
    
    Args:
        level: Log level (default: DEBUG)
        log_args: Czy logować argumenty
        log_result: Czy logować wynik
        **extra_context: Dodatkowy kontekst
        
    Returns:
        Decorated function
        
    Example:
        >>> @log_call(level="INFO", log_args=True, log_result=True)
        ... def add(a, b):
        ...     return a + b
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            bound_logger = logger.bind(**extra_context) if extra_context else logger
            
            # Log call
            if log_args:
                bound_logger.log(level, f"Calling {func_name} with args={args}, kwargs={kwargs}")
            else:
                bound_logger.log(level, f"Calling {func_name}")
            
            # Execute
            result = func(*args, **kwargs)
            
            # Log result
            if log_result:
                bound_logger.log(level, f"{func_name} returned: {result}")
            else:
                bound_logger.log(level, f"{func_name} completed")
            
            return result
        
        return wrapper
    return decorator


# ========================================================================================
# UTILITIES
# ========================================================================================

def get_log_files() -> Dict[str, pathlib.Path]:
    """
    Zwraca ścieżki do plików logów.
    
    Returns:
        Dict z ścieżkami
    """
    config = _merge_configs()
    
    return {
        "log_dir": LOG_DIR,
        "log_file": LOG_DIR / config["log_filename"],
        "json_file": LOG_DIR / config["json_filename"] if config["serialize_json"] else None,
    }


def tail_log_file(n: int = 50) -> str:
    """
    Zwraca ostatnie N linii z pliku logów.
    
    Args:
        n: Liczba linii
        
    Returns:
        Logi jako string
    """
    config = _merge_configs()
    log_file = LOG_DIR / config["log_filename"]
    
    if not log_file.exists():
        return "Log file not found"
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return "".join(lines[-n:])
    except Exception as e:
        return f"Error reading log file: {e}"


def export_logs(
    output_path: str,
    format: str = "text",
    n: Optional[int] = None
) -> None:
    """
    Eksportuje logi do pliku.
    
    Args:
        output_path: Ścieżka do pliku wyjściowego
        format: Format ("text" lub "json")
        n: Liczba logów (None = wszystkie z memory buffer)
    """
    output = pathlib.Path(output_path)
    
    logs = get_memory_logs(n)
    
    if format == "json":
        # Convert to JSON list
        json_logs = [{"log": log.strip()} for log in logs]
        output.write_text(json.dumps(json_logs, indent=2), encoding="utf-8")
    else:
        # Plain text
        output.write_text("".join(logs), encoding="utf-8")