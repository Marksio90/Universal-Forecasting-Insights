# src/utils/logger.py
from __future__ import annotations
import os
import sys
import json
import pathlib
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Iterable
from collections import deque

try:
    import yaml  # pyyaml
except Exception:
    yaml = None  # pragma: no cover

from loguru import logger as _logger

# ======================================================
# ŚCIEŻKI I DOMYŚLNE USTAWIENIA
# ======================================================
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULTS = {
    "level": "INFO",
    "console_level": None,   # jak None → jak level
    "file_level": None,      # jak None → jak level
    "rotation": "10 MB",
    "retention": "14 days",
    "serialize_json": True,
    "json_filename": "app.jsonl",
    "log_filename": "app.log",
    "app_name": "intelligent-predictor",
    "backtrace": False,
    "diagnose": False,
    "enqueue": True,
    "memory_buffer": 2000,
}

def _load_config() -> dict:
    cfg = dict(DEFAULTS)
    cfg_path = PROJECT_ROOT / "config.yaml"
    if yaml and cfg_path.exists():
        try:
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            sec = raw.get("logging") or {}
            if isinstance(sec, dict):
                cfg.update({k: sec.get(k, cfg[k]) for k in DEFAULTS})
        except Exception:
            pass
    env_map = {
        "level": "LOG_LEVEL",
        "console_level": "LOG_CONSOLE_LEVEL",
        "file_level": "LOG_FILE_LEVEL",
        "rotation": "LOG_ROTATION",
        "retention": "LOG_RETENTION",
        "serialize_json": "LOG_JSON",
        "json_filename": "LOG_JSON_FILENAME",
        "log_filename": "LOG_FILE",
        "app_name": "APP_NAME",
        "backtrace": "LOG_BACKTRACE",
        "diagnose": "LOG_DIAGNOSE",
        "enqueue": "LOG_ENQUEUE",
        "memory_buffer": "LOG_MEMORY_BUFFER",
    }
    for k, env in env_map.items():
        v = os.getenv(env)
        if v is None:
            continue
        if isinstance(DEFAULTS[k], bool):
            cfg[k] = v.lower() in ("1", "true", "yes", "on")
        elif isinstance(DEFAULTS[k], int):
            try:
                cfg[k] = int(v)
            except Exception:
                pass
        else:
            cfg[k] = v
    if not cfg["console_level"]:
        cfg["console_level"] = cfg["level"]
    if not cfg["file_level"]:
        cfg["file_level"] = cfg["level"]
    return cfg

@dataclass
class _MemorySink:
    maxlen: int = 2000
    def __post_init__(self):
        self.buffer: deque[str] = deque(maxlen=self.maxlen)
    def write(self, message: str) -> None:
        msg = message if message.endswith("\n") else message + "\n"
        self.buffer.append(msg)
    def dump(self, n: Optional[int] = None) -> str:
        if not self.buffer:
            return ""
        if n is None or n >= len(self.buffer):
            return "".join(list(self.buffer))
        return "".join(list(self.buffer)[-n:])
    def lines(self, n: Optional[int] = None) -> list[str]:
        if not self.buffer:
            return []
        if n is None or n >= len(self.buffer):
            return list(self.buffer)
        return list(self.buffer)[-n:]

_MEM_SINK = _MemorySink(DEFAULTS["memory_buffer"])

def get_memory_logs(n: Optional[int] = None) -> list[str]:
    return _MEM_SINK.lines(n)

class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = _logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[attr-defined]
            depth += 1
        _logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def patch_std_logging(level: str = "INFO") -> None:
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "asyncio", "matplotlib", "prophet", "fbprophet",
                 "numexpr", "PIL", "urllib3", "botocore", "s3transfer", "chromadb", "pinecone"):
        logging.getLogger(name).handlers = [InterceptHandler()]
        logging.getLogger(name).setLevel(level)
    warnings.simplefilter("default")
    logging.captureWarnings(True)

def silence(names: Iterable[str], level: str = "WARNING") -> None:
    for n in names:
        try:
            _logger.disable(n)
        except Exception:
            pass
        logging.getLogger(n).setLevel(level)

# >>> Węższy, czytelniejszy format konsoli <<<
_CONSOLE_FMT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <5}</level> | "
    "{message}"
)

_FILE_FMT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"

_configured = False

def configure_logger(force: bool = False) -> "_logger.__class__":
    global _configured
    if _configured and not force:
        return _logger

    cfg = _load_config()
    _logger.remove()

    _logger.add(
        sys.stderr,
        level=cfg["console_level"],
        format=_CONSOLE_FMT,
        colorize=True,
        backtrace=cfg["backtrace"],
        diagnose=cfg["diagnose"],
        enqueue=cfg["enqueue"],
    )

    logfile = LOG_DIR / cfg["log_filename"]
    _logger.add(
        logfile,
        level=cfg["file_level"],
        format=_FILE_FMT,
        rotation=cfg["rotation"],
        retention=cfg["retention"],
        backtrace=False,
        diagnose=False,
        enqueue=cfg["enqueue"],
    )

    if cfg["serialize_json"]:
        jsonfile = LOG_DIR / cfg["json_filename"]
        _logger.add(
            jsonfile,
            level=cfg["file_level"],
            serialize=True,
            rotation=cfg["rotation"],
            retention=cfg["retention"],
            backtrace=False,
            diagnose=False,
            enqueue=cfg["enqueue"],
        )

    _logger.add(
        _MEM_SINK.write,
        level=cfg["console_level"],
        format=_FILE_FMT,  # w buforze zostawiamy info o module/linie
        backtrace=False,
        diagnose=False,
        enqueue=False,
    )

    patch_std_logging(cfg["level"])
    silence(["matplotlib", "prophet", "fbprophet", "urllib3", "PIL", "numexpr", "chromadb", "pinecone"], level="WARNING")
    _logger.bind(app=cfg["app_name"])
    _configured = True
    return _logger

configure_logger()
logger = _logger

def get_logger(module: Optional[str] = None, **context: Any):
    if module:
        context = {"mod": module, **context}
    return logger.bind(**context)

def set_level(level: str) -> None:
    os.environ["LOG_LEVEL"] = level
    configure_logger(force=True)

def with_context(**ctx: Any):
    class _Ctx:
        def __enter__(self):
            self._bound = logger.bind(**ctx)
            self._old = logger
            globals()["logger"] = self._bound
            return self._bound
        def __exit__(self, exc_type, exc, tb):
            globals()["logger"] = _logger
    return _Ctx()

def log_exception(msg: str = "Unhandled exception", **extra: Any):
    def _decorator(func):
        def _wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                logger.bind(**extra).exception(msg)
                raise
        return _wrapped
    return _decorator
