from __future__ import annotations
import os
import pathlib
import threading
from contextlib import contextmanager
from typing import Optional, Dict, Any, Iterable

import yaml  # pyyaml
import pandas as pd
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url, URL
from sqlalchemy.orm import sessionmaker, scoped_session, Session
from sqlalchemy.pool import StaticPool

# Optional: Streamlit secrets (nie jest wymagane do działania)
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None  # pragma: no cover

# ===========================
# Konfiguracja (config.yaml / secrets / env)
# ===========================

def _load_db_config() -> Dict[str, Any]:
    """
    Ładuje konfigurację bazy z:
      1) config.yaml -> sekcja `database`
      2) st.secrets["database"]
      3) zmienne środowiskowe
    Obsługiwane klucze:
      url, echo, pool_size, max_overflow, pool_recycle, pool_pre_ping, application_name
    """
    cfg: Dict[str, Any] = {
        "url": None,
        "echo": False,
        "pool_size": 5,
        "max_overflow": 10,
        "pool_recycle": 1800,    # sekundy
        "pool_pre_ping": True,
        "application_name": "intelligent-predictor",
    }

    # 1) config.yaml
    try:
        cfg_path = pathlib.Path("config.yaml")
        if cfg_path.exists():
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            dbs = raw.get("database") or {}
            if isinstance(dbs, dict):
                cfg.update({
                    "url": dbs.get("url", cfg["url"]),
                    "echo": bool(dbs.get("echo", cfg["echo"])),
                    "pool_size": int(dbs.get("pool_size", cfg["pool_size"])),
                    "max_overflow": int(dbs.get("max_overflow", cfg["max_overflow"])),
                    "pool_recycle": int(dbs.get("pool_recycle", cfg["pool_recycle"])),
                    "pool_pre_ping": bool(dbs.get("pool_pre_ping", cfg["pool_pre_ping"])),
                    "application_name": dbs.get("application_name", cfg["application_name"]),
                })
    except Exception:
        pass

    # 2) st.secrets
    if st is not None:
        try:
            ds = st.secrets.get("database", {})  # type: ignore[attr-defined]
            if ds:
                cfg.update({
                    "url": ds.get("url", cfg["url"]),
                    "echo": bool(ds.get("echo", cfg["echo"])),
                    "pool_size": int(ds.get("pool_size", cfg["pool_size"])),
                    "max_overflow": int(ds.get("max_overflow", cfg["max_overflow"])),
                    "pool_recycle": int(ds.get("pool_recycle", cfg["pool_recycle"])),
                    "pool_pre_ping": bool(ds.get("pool_pre_ping", cfg["pool_pre_ping"])),
                    "application_name": ds.get("application_name", cfg["application_name"]),
                })
        except Exception:
            pass

    # 3) ENV
    cfg["url"] = os.getenv("DATABASE_URL", cfg["url"])
    if os.getenv("DB_ECHO") is not None:
        cfg["echo"] = os.getenv("DB_ECHO", "0").lower() in ("1", "true", "yes")
    cfg["pool_size"] = int(os.getenv("DB_POOL_SIZE", cfg["pool_size"]))
    cfg["max_overflow"] = int(os.getenv("DB_MAX_OVERFLOW", cfg["max_overflow"]))
    cfg["pool_recycle"] = int(os.getenv("DB_POOL_RECYCLE", cfg["pool_recycle"]))
    if os.getenv("DB_PRE_PING") is not None:
        cfg["pool_pre_ping"] = os.getenv("DB_PRE_PING", "1").lower() in ("1", "true", "yes")
    cfg["application_name"] = os.getenv("DB_APP_NAME", cfg["application_name"])

    # Domyślny fallback na SQLite w katalogu projektu
    if not cfg["url"]:
        cfg["url"] = "sqlite:///data/app.db"

    return cfg

# ===========================
# Engine cache i Session factory
# ===========================

_ENGINE: Optional[Engine] = None
_SESSION_FACTORY: Optional[scoped_session] = None
_DB_URL: Optional[URL] = None
_CFG_FINGERPRINT: Optional[str] = None
_LOCK = threading.RLock()

def _ensure_sqlite_dir(url: URL) -> None:
    """Tworzy katalog dla pliku sqlite, jeśli trzeba (pomija :memory:)."""
    if url.get_backend_name() != "sqlite":
        return
    database = (url.database or "").strip() if url.database else ""
    if database and database != ":memory:":
        p = pathlib.Path(database)
        if not p.is_absolute():
            p = pathlib.Path.cwd() / p
        p.parent.mkdir(parents=True, exist_ok=True)

def _configure_sqlite_events(engine: Engine) -> None:
    """Ustawia sensowne PRAGMA dla SQLite."""
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, _):
        try:
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute("PRAGMA cache_size=-65536;")  # ~64MB cache (ujemna = w KB)
            cur.execute("PRAGMA foreign_keys=ON;")
            cur.close()
        except Exception:
            pass  # łagodne

def _configure_postgres_events(engine: Engine, app_name: str) -> None:
    """Ustawia parametry sesji dla Postgresa."""
    @event.listens_for(engine, "connect")
    def _set_pg_session(dbapi_conn, _):
        try:
            with dbapi_conn.cursor() as cur:
                cur.execute("SET statement_timeout = '600000'")  # 10 min
                cur.execute("SET idle_in_transaction_session_timeout = '900000'")  # 15 min
                try:
                    cur.execute("SET application_name = %s", (app_name,))
                except Exception:
                    pass
        except Exception:
            pass  # łagodne

def _build_engine(url_str: str, *, echo: bool, pool_size: int, max_overflow: int,
                  pool_recycle: int, pool_pre_ping: bool, app_name: str) -> Engine:
    """
    Buduje Engine z sensownymi opcjami dla SQLite i Postgres.
    """
    url = make_url(url_str)
    _ensure_sqlite_dir(url)

    engine_kwargs: Dict[str, Any] = {
        "echo": echo,
        "pool_pre_ping": pool_pre_ping,
        "future": True,
    }

    if url.get_backend_name() == "sqlite":
        # In-memory → StaticPool żeby sesje współdzieliły tę samą pamięć
        if (url.database or "") == ":memory:" or url_str.endswith("://"):
            engine_kwargs.update({
                "poolclass": StaticPool,
                "connect_args": {"check_same_thread": False},
            })
        else:
            engine_kwargs.update({
                "connect_args": {"check_same_thread": False},
            })
    else:
        # Postgres / inne dialekty z pulą połączeń
        engine_kwargs.update({
            "pool_size": pool_size,
            "max_overflow": max_overflow,
            "pool_recycle": pool_recycle,
        })

    engine = create_engine(url, **engine_kwargs)

    # Eventy dla backendów
    backend = url.get_backend_name()
    if backend == "sqlite":
        _configure_sqlite_events(engine)
    elif backend in ("postgresql", "postgresql+psycopg", "postgresql+psycopg2"):
        _configure_postgres_events(engine, app_name)

    return engine

def _fingerprint(cfg: Dict[str, Any]) -> str:
    """Prosty odcisk konfigu (bezpieczny do porównań)."""
    keys = ("url", "echo", "pool_size", "max_overflow", "pool_recycle", "pool_pre_ping", "application_name")
    values = tuple(cfg.get(k) for k in keys)
    return str(hash(values))

def _init_or_reuse_engine(cfg: Dict[str, Any]) -> Engine:
    global _ENGINE, _DB_URL, _SESSION_FACTORY, _CFG_FINGERPRINT
    with _LOCK:
        fp = _fingerprint(cfg)
        if _ENGINE is not None and _CFG_FINGERPRINT == fp:
            return _ENGINE
        # jeśli engine istnieje, zamknij go przed rekreacją
        if _SESSION_FACTORY is not None:
            try:
                _SESSION_FACTORY.remove()
            except Exception:
                pass
            _SESSION_FACTORY = None
        if _ENGINE is not None:
            try:
                _ENGINE.dispose()
            except Exception:
                pass
            _ENGINE = None
        # zbuduj nowy
        engine = _build_engine(
            cfg.get("url"),
            echo=bool(cfg.get("echo", False)),
            pool_size=int(cfg.get("pool_size", 5)),
            max_overflow=int(cfg.get("max_overflow", 10)),
            pool_recycle=int(cfg.get("pool_recycle", 1800)),
            pool_pre_ping=bool(cfg.get("pool_pre_ping", True)),
            app_name=str(cfg.get("application_name", "intelligent-predictor")),
        )
        _ENGINE = engine
        _DB_URL = make_url(cfg.get("url"))
        _CFG_FINGERPRINT = fp
        return engine

def get_engine(path: str = "sqlite:///data/app.db") -> Engine:
    """
    Zwraca globalny Engine (cache). Parametr `path` służy jako
    *domyślny fallback*, ale faktyczny URL jest pobierany z konfiguracji.
    Jeśli konfiguracja uległa zmianie od poprzedniego wywołania — engine zostanie
    bezpiecznie przeinstancjonowany.
    """
    cfg = _load_db_config()
    if not cfg.get("url"):
        cfg["url"] = path
    return _init_or_reuse_engine(cfg)

def get_url() -> str:
    """Zwraca aktualny URL połączenia (string)."""
    global _DB_URL
    if _DB_URL is None:
        _ = get_engine()
    return str(_DB_URL) if _DB_URL is not None else "sqlite:///data/app.db"

def safe_url(mask_password: bool = True) -> str:
    """
    Zwraca URL bez hasła (np. dla logów/UI).
    """
    try:
        url = make_url(get_url())
        if mask_password and url.password:
            url = url.set(password="***")
        return str(url)
    except Exception:
        return get_url()

# ===========================
# Session factory + scope
# ===========================

def _get_session_factory() -> scoped_session:
    global _SESSION_FACTORY
    if _SESSION_FACTORY is not None:
        return _SESSION_FACTORY
    engine = get_engine()
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)
    _SESSION_FACTORY = scoped_session(factory)
    return _SESSION_FACTORY

def get_session() -> Session:
    """Zwraca nową sesję z globalnej fabryki (scoped_session)."""
    return _get_session_factory()()

@contextmanager
def session_scope() -> Session:
    """
    Kontekst zarządzania sesją:
        with session_scope() as s:
            s.add(obj)
    Zapewnia commit/rollback/close.
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# ===========================
# Helpery operacyjne
# ===========================

def health_check() -> bool:
    """Prosta weryfikacja połączenia (SELECT 1)."""
    eng = get_engine()
    try:
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False

def dispose_engine() -> None:
    """Zamyka i czyści Engine + Session factory (np. po zmianie URL)."""
    global _ENGINE, _SESSION_FACTORY, _DB_URL, _CFG_FINGERPRINT
    with _LOCK:
        try:
            if _SESSION_FACTORY is not None:
                _SESSION_FACTORY.remove()
        except Exception:
            pass
        try:
            if _ENGINE is not None:
                _ENGINE.dispose()
        except Exception:
            pass
        _ENGINE = None
        _SESSION_FACTORY = None
        _DB_URL = None
        _CFG_FINGERPRINT = None

def ensure_schema(base: Optional[Any] = None) -> None:
    """
    Tworzy tabele na podstawie deklaratywnej bazy (SQLAlchemy Base).
    Użycie:
        from .models import Base
        ensure_schema(Base)
    """
    if base is None:
        return
    eng = get_engine()
    try:
        base.metadata.create_all(eng)  # type: ignore[attr-defined]
    except Exception:
        # Nie przerywaj działania UI — schema można utworzyć ręcznie
        pass

def exec_sql(sql: str, params: Optional[dict] = None) -> None:
    """Szybkie wykonanie arbitralnego SQL (DDL/DML)."""
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text(sql), params or {})

# ===========================
# Dodatkowe (opcjonalne) helpery PRO
# ===========================

def query_df(sql: str, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Wykonuje zapytanie SELECT i zwraca DataFrame.
    """
    eng = get_engine()
    with eng.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})

def execute_many(sql: str, rows: Iterable[dict]) -> int:
    """
    Szybkie wykonanie wielu DML (INSERT/UPDATE) z parametrami.
    Zwraca liczbę wierszy objętych operacją (best-effort).
    """
    eng = get_engine()
    with eng.begin() as conn:
        res = conn.execute(text(sql), list(rows))
        try:
            return int(res.rowcount or 0)
        except Exception:
            return 0

def set_url_and_reinit(url: str) -> None:
    """
    Ustawia URL bazy w locie i bezpiecznie przeinstancjonuje engine + session factory.
    Przydatne w UI lub testach.
    """
    # Nadpisz env i zainicjuj od nowa
    os.environ["DATABASE_URL"] = url
    dispose_engine()
    get_engine()
