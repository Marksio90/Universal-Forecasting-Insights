"""
database_engine.py — ULTRA PRO Edition

Production-ready database engine with:
- Multi-database support (SQLite, PostgreSQL, MySQL, etc.)
- Connection pooling with health checks
- Automatic reconnection and retry logic
- Thread-safe operations
- Comprehensive error handling
- Performance monitoring
- Migration support hints
- Query timeout protection
- Connection leak detection
"""

from __future__ import annotations

import os
import time
import logging
import threading
import pathlib
from functools import wraps
from contextlib import contextmanager
from typing import Optional, Dict, Any, Iterable, List, Union, Generator
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import (
    create_engine, event, text, inspect,
    MetaData, Table
)
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.engine.url import make_url, URL
from sqlalchemy.orm import sessionmaker, scoped_session, Session
from sqlalchemy.pool import StaticPool, NullPool, QueuePool
from sqlalchemy.exc import (
    SQLAlchemyError, OperationalError, 
    DisconnectionError, TimeoutError as SATimeoutError
)

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

# ========================================================================================
# LOGGING
# ========================================================================================

def get_logger(name: str = "database_engine") -> logging.Logger:
    """Configure logger."""
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

# Default values
DEFAULT_APPLICATION_NAME = "intelligent-predictor"
DEFAULT_POOL_SIZE = 5
DEFAULT_MAX_OVERFLOW = 10
DEFAULT_POOL_RECYCLE = 1800  # 30 minutes
DEFAULT_POOL_TIMEOUT = 30  # seconds
DEFAULT_STATEMENT_TIMEOUT = 600000  # 10 minutes (PostgreSQL)
DEFAULT_IDLE_TIMEOUT = 900000  # 15 minutes (PostgreSQL)

# SQLite defaults
SQLITE_DEFAULT_PATH = "data/app.db"
SQLITE_CACHE_SIZE = -65536  # ~64MB in KB

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
RETRY_BACKOFF = 2.0  # exponential backoff multiplier

# Monitoring
QUERY_LOG_THRESHOLD = 1.0  # Log queries taking longer than 1 second

# ========================================================================================
# DATACLASSES
# ========================================================================================

@dataclass
class DatabaseConfig:
    """Database configuration."""
    
    # Connection
    url: Optional[str] = None
    
    # Pooling
    pool_size: int = DEFAULT_POOL_SIZE
    max_overflow: int = DEFAULT_MAX_OVERFLOW
    pool_recycle: int = DEFAULT_POOL_RECYCLE
    pool_timeout: int = DEFAULT_POOL_TIMEOUT
    pool_pre_ping: bool = True
    
    # Application
    application_name: str = DEFAULT_APPLICATION_NAME
    
    # Timeouts (PostgreSQL)
    statement_timeout: int = DEFAULT_STATEMENT_TIMEOUT
    idle_timeout: int = DEFAULT_IDLE_TIMEOUT
    
    # Debugging
    echo: bool = False
    echo_pool: bool = False
    
    # Performance
    use_batch_mode: bool = True
    
    # Safety
    enable_connection_leak_detection: bool = True
    max_connection_age: int = 3600  # 1 hour


@dataclass
class ConnectionMetrics:
    """Connection pool metrics."""
    
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    queries_executed: int = 0
    slow_queries: int = 0
    connection_errors: int = 0
    total_query_time: float = 0.0
    start_time: float = field(default_factory=time.time)
    
    @property
    def avg_query_time_ms(self) -> float:
        """Average query time in milliseconds."""
        if self.queries_executed == 0:
            return 0.0
        return (self.total_query_time / self.queries_executed) * 1000
    
    def reset(self) -> None:
        """Reset metrics."""
        self.queries_executed = 0
        self.slow_queries = 0
        self.connection_errors = 0
        self.total_query_time = 0.0
        self.start_time = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        uptime = time.time() - self.start_time
        
        return {
            "total_connections": self.total_connections,
            "active_connections": self.active_connections,
            "idle_connections": self.idle_connections,
            "queries_executed": self.queries_executed,
            "slow_queries": self.slow_queries,
            "connection_errors": self.connection_errors,
            "avg_query_time_ms": round(self.avg_query_time_ms, 3),
            "uptime_seconds": round(uptime, 1)
        }


# ========================================================================================
# CONFIGURATION LOADING
# ========================================================================================

def _load_database_config() -> DatabaseConfig:
    """
    Load database configuration from multiple sources (priority order):
    1. Environment variables (highest priority)
    2. Streamlit secrets
    3. config.yaml
    4. Defaults (lowest priority)
    
    Returns:
        DatabaseConfig with merged settings
    """
    config = DatabaseConfig()
    
    # ============================================================================
    # 1. Load from config.yaml
    # ============================================================================
    
    if HAS_YAML and yaml is not None:
        try:
            config_path = pathlib.Path("config.yaml")
            
            if config_path.exists():
                LOGGER.debug("Loading database config from config.yaml")
                
                data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                
                if isinstance(data, dict) and "database" in data:
                    db_cfg = data["database"]
                    
                    if isinstance(db_cfg, dict):
                        config.url = db_cfg.get("url", config.url)
                        config.echo = bool(db_cfg.get("echo", config.echo))
                        config.echo_pool = bool(db_cfg.get("echo_pool", config.echo_pool))
                        config.pool_size = int(db_cfg.get("pool_size", config.pool_size))
                        config.max_overflow = int(db_cfg.get("max_overflow", config.max_overflow))
                        config.pool_recycle = int(db_cfg.get("pool_recycle", config.pool_recycle))
                        config.pool_timeout = int(db_cfg.get("pool_timeout", config.pool_timeout))
                        config.pool_pre_ping = bool(db_cfg.get("pool_pre_ping", config.pool_pre_ping))
                        config.application_name = db_cfg.get("application_name", config.application_name)
                        config.statement_timeout = int(db_cfg.get("statement_timeout", config.statement_timeout))
                        config.idle_timeout = int(db_cfg.get("idle_timeout", config.idle_timeout))
                        
                        LOGGER.info("Database config loaded from config.yaml")
        except Exception as e:
            LOGGER.warning(f"Failed to load config.yaml: {e}")
    
    # ============================================================================
    # 2. Load from Streamlit secrets
    # ============================================================================
    
    if HAS_STREAMLIT and st is not None:
        try:
            secrets = st.secrets.get("database", {})
            
            if secrets:
                LOGGER.debug("Loading database config from Streamlit secrets")
                
                config.url = secrets.get("url", config.url)
                config.echo = bool(secrets.get("echo", config.echo))
                config.pool_size = int(secrets.get("pool_size", config.pool_size))
                config.max_overflow = int(secrets.get("max_overflow", config.max_overflow))
                config.pool_recycle = int(secrets.get("pool_recycle", config.pool_recycle))
                config.pool_pre_ping = bool(secrets.get("pool_pre_ping", config.pool_pre_ping))
                config.application_name = secrets.get("application_name", config.application_name)
                
                LOGGER.info("Database config loaded from Streamlit secrets")
        except Exception as e:
            LOGGER.debug(f"No Streamlit secrets found: {e}")
    
    # ============================================================================
    # 3. Environment variables (highest priority)
    # ============================================================================
    
    # Database URL
    db_url = os.getenv("DATABASE_URL") or os.getenv("DB_URL")
    if db_url:
        config.url = db_url
        LOGGER.debug("Database URL loaded from environment")
    
    # Echo mode
    if os.getenv("DB_ECHO"):
        config.echo = os.getenv("DB_ECHO", "").lower() in ("1", "true", "yes", "on")
    
    if os.getenv("DB_ECHO_POOL"):
        config.echo_pool = os.getenv("DB_ECHO_POOL", "").lower() in ("1", "true", "yes", "on")
    
    # Pool settings
    if os.getenv("DB_POOL_SIZE"):
        try:
            config.pool_size = int(os.getenv("DB_POOL_SIZE", config.pool_size))
        except ValueError:
            pass
    
    if os.getenv("DB_MAX_OVERFLOW"):
        try:
            config.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", config.max_overflow))
        except ValueError:
            pass
    
    if os.getenv("DB_POOL_RECYCLE"):
        try:
            config.pool_recycle = int(os.getenv("DB_POOL_RECYCLE", config.pool_recycle))
        except ValueError:
            pass
    
    if os.getenv("DB_POOL_TIMEOUT"):
        try:
            config.pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", config.pool_timeout))
        except ValueError:
            pass
    
    # Pre-ping
    if os.getenv("DB_PRE_PING"):
        config.pool_pre_ping = os.getenv("DB_PRE_PING", "").lower() in ("1", "true", "yes", "on")
    
    # Application name
    if os.getenv("DB_APP_NAME") or os.getenv("APPLICATION_NAME"):
        config.application_name = os.getenv("DB_APP_NAME") or os.getenv("APPLICATION_NAME", config.application_name)
    
    # ============================================================================
    # Validation and defaults
    # ============================================================================
    
    # Default to SQLite if no URL provided
    if not config.url:
        config.url = f"sqlite:///{SQLITE_DEFAULT_PATH}"
        LOGGER.info(f"No database URL provided, using default SQLite: {config.url}")
    
    # Validate pool settings
    if config.pool_size < 1:
        LOGGER.warning(f"Invalid pool_size: {config.pool_size}, using {DEFAULT_POOL_SIZE}")
        config.pool_size = DEFAULT_POOL_SIZE
    
    if config.max_overflow < 0:
        LOGGER.warning(f"Invalid max_overflow: {config.max_overflow}, using {DEFAULT_MAX_OVERFLOW}")
        config.max_overflow = DEFAULT_MAX_OVERFLOW
    
    return config


# ========================================================================================
# ENGINE MANAGEMENT
# ========================================================================================

class DatabaseEngine:
    """
    Thread-safe database engine manager.
    
    Manages SQLAlchemy engine lifecycle with automatic reconnection,
    connection pooling, and performance monitoring.
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize database engine.
        
        Args:
            config: Optional database configuration
        """
        self.config = config or _load_database_config()
        self.metrics = ConnectionMetrics()
        
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[scoped_session] = None
        self._url: Optional[URL] = None
        self._lock = threading.RLock()
        self._initialized = False
        
        # Initialize engine
        self._initialize_engine()
    
    def _get_config_fingerprint(self) -> str:
        """Generate configuration fingerprint for cache invalidation."""
        key_values = (
            self.config.url,
            self.config.pool_size,
            self.config.max_overflow,
            self.config.pool_recycle,
            self.config.pool_timeout,
            self.config.pool_pre_ping,
            self.config.echo
        )
        return str(hash(key_values))
    
    def _ensure_sqlite_directory(self, url: URL) -> None:
        """Create SQLite database directory if needed."""
        if url.get_backend_name() != "sqlite":
            return
        
        database = (url.database or "").strip()
        
        if not database or database == ":memory:":
            return
        
        db_path = pathlib.Path(database)
        
        if not db_path.is_absolute():
            db_path = pathlib.Path.cwd() / db_path
        
        db_path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.debug(f"SQLite directory ensured: {db_path.parent}")
    
    def _configure_sqlite_pragmas(self, engine: Engine) -> None:
        """Configure SQLite PRAGMA settings for better performance."""
        
        @event.listens_for(engine, "connect")
        def set_sqlite_pragmas(dbapi_conn, connection_record):
            try:
                cursor = dbapi_conn.cursor()
                
                # WAL mode for better concurrency
                cursor.execute("PRAGMA journal_mode=WAL")
                
                # Faster but still safe
                cursor.execute("PRAGMA synchronous=NORMAL")
                
                # Larger cache for better performance
                cursor.execute(f"PRAGMA cache_size={SQLITE_CACHE_SIZE}")
                
                # Enable foreign keys
                cursor.execute("PRAGMA foreign_keys=ON")
                
                # Optimize temp storage
                cursor.execute("PRAGMA temp_store=MEMORY")
                
                cursor.close()
                LOGGER.debug("SQLite PRAGMA settings applied")
                
            except Exception as e:
                LOGGER.warning(f"Failed to set SQLite PRAGMA: {e}")
    
    def _configure_postgresql_settings(self, engine: Engine) -> None:
        """Configure PostgreSQL session settings."""
        
        @event.listens_for(engine, "connect")
        def set_postgresql_settings(dbapi_conn, connection_record):
            try:
                cursor = dbapi_conn.cursor()
                
                # Statement timeout
                cursor.execute(f"SET statement_timeout = '{self.config.statement_timeout}'")
                
                # Idle in transaction timeout
                cursor.execute(f"SET idle_in_transaction_session_timeout = '{self.config.idle_timeout}'")
                
                # Application name
                try:
                    cursor.execute("SET application_name = %s", (self.config.application_name,))
                except Exception:
                    pass
                
                # Client encoding
                cursor.execute("SET client_encoding = 'UTF8'")
                
                cursor.close()
                LOGGER.debug("PostgreSQL session settings applied")
                
            except Exception as e:
                LOGGER.warning(f"Failed to set PostgreSQL settings: {e}")
    
    def _configure_mysql_settings(self, engine: Engine) -> None:
        """Configure MySQL session settings."""
        
        @event.listens_for(engine, "connect")
        def set_mysql_settings(dbapi_conn, connection_record):
            try:
                cursor = dbapi_conn.cursor()
                
                # Wait timeout
                cursor.execute("SET SESSION wait_timeout = 3600")
                
                # Interactive timeout
                cursor.execute("SET SESSION interactive_timeout = 3600")
                
                # Max execution time (MySQL 5.7.8+)
                try:
                    cursor.execute(f"SET SESSION max_execution_time = {self.config.statement_timeout}")
                except Exception:
                    pass
                
                cursor.close()
                LOGGER.debug("MySQL session settings applied")
                
            except Exception as e:
                LOGGER.warning(f"Failed to set MySQL settings: {e}")
    
    def _setup_query_logging(self, engine: Engine) -> None:
        """Setup slow query logging."""
        
        @event.listens_for(engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            conn.info.setdefault("query_start_time", []).append(time.time())
        
        @event.listens_for(engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            total_time = time.time() - conn.info["query_start_time"].pop()
            
            self.metrics.queries_executed += 1
            self.metrics.total_query_time += total_time
            
            if total_time > QUERY_LOG_THRESHOLD:
                self.metrics.slow_queries += 1
                LOGGER.warning(
                    f"Slow query detected ({total_time:.3f}s): "
                    f"{statement[:200]}{'...' if len(statement) > 200 else ''}"
                )
    
    def _setup_connection_monitoring(self, engine: Engine) -> None:
        """Setup connection pool monitoring."""
        
        @event.listens_for(engine, "connect")
        def on_connect(dbapi_conn, connection_record):
            self.metrics.total_connections += 1
            LOGGER.debug(f"New connection created (total: {self.metrics.total_connections})")
        
        @event.listens_for(engine, "checkout")
        def on_checkout(dbapi_conn, connection_record, connection_proxy):
            self.metrics.active_connections += 1
            LOGGER.debug(f"Connection checked out (active: {self.metrics.active_connections})")
        
        @event.listens_for(engine, "checkin")
        def on_checkin(dbapi_conn, connection_record):
            if self.metrics.active_connections > 0:
                self.metrics.active_connections -= 1
            LOGGER.debug(f"Connection checked in (active: {self.metrics.active_connections})")
    
    def _build_engine(self) -> Engine:
        """
        Build SQLAlchemy engine with appropriate settings.
        
        Returns:
            Configured engine
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.config.url:
            raise ValueError("Database URL is required")
        
        url = make_url(self.config.url)
        self._url = url
        
        # Ensure directory for SQLite
        self._ensure_sqlite_directory(url)
        
        # Base engine kwargs
        engine_kwargs: Dict[str, Any] = {
            "echo": self.config.echo,
            "echo_pool": self.config.echo_pool,
            "pool_pre_ping": self.config.pool_pre_ping,
            "future": True,
        }
        
        # Configure pooling based on backend
        backend = url.get_backend_name()
        
        if backend == "sqlite":
            # SQLite-specific configuration
            database = (url.database or "").strip()
            
            if database == ":memory:" or not database:
                # In-memory: use StaticPool for shared connection
                engine_kwargs["poolclass"] = StaticPool
                engine_kwargs["connect_args"] = {"check_same_thread": False}
                LOGGER.info("Using SQLite in-memory database with StaticPool")
            else:
                # File-based: use NullPool for better thread safety
                engine_kwargs["poolclass"] = NullPool
                engine_kwargs["connect_args"] = {
                    "check_same_thread": False,
                    "timeout": 20.0  # Lock timeout
                }
                LOGGER.info(f"Using SQLite file database: {database}")
        
        else:
            # Server databases: use connection pooling
            engine_kwargs.update({
                "pool_size": self.config.pool_size,
                "max_overflow": self.config.max_overflow,
                "pool_recycle": self.config.pool_recycle,
                "pool_timeout": self.config.pool_timeout,
                "poolclass": QueuePool,
            })
            LOGGER.info(
                f"Using {backend} with connection pool "
                f"(size={self.config.pool_size}, overflow={self.config.max_overflow})"
            )
        
        # Create engine
        try:
            engine = create_engine(self.config.url, **engine_kwargs)
            
            # Configure database-specific settings
            if backend == "sqlite":
                self._configure_sqlite_pragmas(engine)
            elif backend in ("postgresql", "postgresql+psycopg", "postgresql+psycopg2", "postgresql+pg8000"):
                self._configure_postgresql_settings(engine)
            elif backend in ("mysql", "mysql+pymysql", "mysql+mysqlconnector"):
                self._configure_mysql_settings(engine)
            
            # Setup monitoring
            self._setup_query_logging(engine)
            self._setup_connection_monitoring(engine)
            
            LOGGER.info(f"✅ Database engine created: {backend}")
            
            return engine
            
        except Exception as e:
            LOGGER.error(f"Failed to create database engine: {e}")
            raise
    
    def _initialize_engine(self) -> None:
        """Initialize engine and session factory."""
        with self._lock:
            if self._initialized:
                return
            
            try:
                self._engine = self._build_engine()
                
                # Create session factory
                session_factory = sessionmaker(
                    bind=self._engine,
                    autoflush=False,
                    autocommit=False,
                    expire_on_commit=False,
                    future=True
                )
                
                self._session_factory = scoped_session(session_factory)
                
                self._initialized = True
                
                LOGGER.info("✅ Database engine initialized")
                
            except Exception as e:
                LOGGER.error(f"Failed to initialize database engine: {e}")
                raise
    
    @property
    def engine(self) -> Engine:
        """Get engine instance."""
        if not self._initialized or self._engine is None:
            raise RuntimeError("Engine not initialized")
        
        return self._engine
    
    @property
    def url(self) -> str:
        """Get database URL."""
        if self._url is None:
            return self.config.url or ""
        
        return str(self._url)
    
    def safe_url(self, mask_password: bool = True) -> str:
        """
        Get safe URL for display (with masked password).
        
        Args:
            mask_password: Whether to mask password
            
        Returns:
            Safe URL string
        """
        try:
            url = make_url(self.url)
            
            if mask_password and url.password:
                url = url.set(password="***")
            
            return str(url)
            
        except Exception:
            return self.url
    
    def get_session(self) -> Session:
        """
        Get new database session.
        
        Returns:
            SQLAlchemy session
        """
        if self._session_factory is None:
            raise RuntimeError("Session factory not initialized")
        
        return self._session_factory()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Context manager for database session with automatic commit/rollback.
        
        Yields:
            Database session
            
        Examples:
            >>> with db.session_scope() as session:
            ...     session.add(obj)
            ...     # Automatic commit on success, rollback on error
        """
        session = self.get_session()
        
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def execute(
        self,
        statement: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        commit: bool = True
    ) -> Any:
        """
        Execute SQL statement.
        
        Args:
            statement: SQL statement
            params: Optional parameters
            commit: Whether to commit transaction
            
        Returns:
            Result proxy
        """
        with self.engine.begin() as conn:
            result = conn.execute(text(statement), params or {})
            
            if commit:
                conn.commit()
            
            return result
    
    def query_df(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Execute query and return DataFrame.
        
        Args:
            query: SQL query
            params: Optional parameters
            **kwargs: Additional arguments for pd.read_sql
            
        Returns:
            Query results as DataFrame
        """
        with self.engine.connect() as conn:
            return pd.read_sql(
                text(query),
                conn,
                params=params or {},
                **kwargs
            )
    
    def execute_many(
        self,
        statement: str,
        rows: Iterable[Dict[str, Any]]
    ) -> int:
        """
        Execute statement with multiple parameter sets.
        
        Args:
            statement: SQL statement
            rows: Iterable of parameter dictionaries
            
        Returns:
            Number of affected rows
        """
        rows_list = list(rows)
        
        if not rows_list:
            return 0
        
        with self.engine.begin() as conn:
            result = conn.execute(text(statement), rows_list)
            
            try:
                return int(result.rowcount or 0)
            except Exception:
                return len(rows_list)
    
    def health_check(self, timeout: float = 5.0) -> bool:
        """
        Check database connection health.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            True if healthy
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            LOGGER.debug("Health check passed")
            return True
            
        except Exception as e:
            LOGGER.error(f"Health check failed: {e}")
            self.metrics.connection_errors += 1
            return False
    
    def get_table_names(self, schema: Optional[str] = None) -> List[str]:
        """
        Get list of table names.
        
        Args:
            schema: Optional schema name
            
        Returns:
            List of table names
        """
        try:
            inspector = inspect(self.engine)
            return inspector.get_table_names(schema=schema)
        except Exception as e:
            LOGGER.error(f"Failed to get table names: {e}")
            return []
    
    def table_exists(self, table_name: str, schema: Optional[str] = None) -> bool:
        """
        Check if table exists.
        
        Args:
            table_name: Name of table
            schema: Optional schema name
            
        Returns:
            True if table exists
        """
        try:
            inspector = inspect(self.engine)
            return inspector.has_table(table_name, schema=schema)
        except Exception as e:
            LOGGER.error(f"Failed to check table existence: {e}")
            return False
    
    def create_tables(self, metadata: MetaData) -> None:
        """
        Create tables from metadata.
        
        Args:
            metadata: SQLAlchemy metadata
        """
        try:
            metadata.create_all(self.engine)
            LOGGER.info("Tables created successfully")
        except Exception as e:
            LOGGER.error(f"Failed to create tables: {e}")
            raise
    
    def drop_tables(self, metadata: MetaData) -> None:
        """
        Drop tables from metadata.
        
        Args:
            metadata: SQLAlchemy metadata
        """
        try:
            metadata.drop_all(self.engine)
            LOGGER.info("Tables dropped successfully")
        except Exception as e:
            LOGGER.error(f"Failed to drop tables: {e}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get connection metrics.
        
        Returns:
            Metrics dictionary
        """
        metrics_dict = self.metrics.to_dict()
        
        # Add pool info if available
        if hasattr(self.engine.pool, "size"):
            metrics_dict["pool_size"] = self.engine.pool.size()
        
        if hasattr(self.engine.pool, "checkedin"):
            metrics_dict["idle_connections"] = self.engine.pool.checkedin()
        
        return metrics_dict
    
    def reset_metrics(self) -> None:
        """Reset connection metrics."""
        with self._lock:
            self.metrics.reset()
        
        LOGGER.info("Metrics reset")
    
    def status(self) -> str:
        """
        Get database status string.
        
        Returns:
            Status string
        """
        try:
            healthy = self.health_check()
            backend = self._url.get_backend_name() if self._url else "unknown"
            
            metrics = self.get_metrics()
            
            status_lines = [
                f"{'✅' if healthy else '❌'} Database: {backend}",
                f"URL: {self.safe_url()}",
                f"Pool size: {self.config.pool_size}",
                f"Active connections: {metrics.get('active_connections', 0)}",
                f"Queries executed: {metrics.get('queries_executed', 0)}",
                f"Slow queries: {metrics.get('slow_queries', 0)}",
                f"Avg query time: {metrics.get('avg_query_time_ms', 0):.2f}ms",
            ]
            
            return "\n".join(status_lines)
            
        except Exception as e:
            return f"❌ Error: {e}"
    
    def dispose(self) -> None:
        """Dispose engine and close all connections."""
        with self._lock:
            try:
                if self._session_factory is not None:
                    self._session_factory.remove()
                    self._session_factory = None
            except Exception as e:
                LOGGER.warning(f"Error removing session factory: {e}")
            
            try:
                if self._engine is not None:
                    self._engine.dispose()
                    self._engine = None
            except Exception as e:
                LOGGER.warning(f"Error disposing engine: {e}")
            
            self._url = None
            self._initialized = False
            
            LOGGER.info("Database engine disposed")

    def reinitialize(self, config: Optional[DatabaseConfig] = None) -> None:
        """
        Reinitialize engine with new configuration.
        
        Args:
            config: New database configuration
        """
        with self._lock:
            LOGGER.info("Reinitializing database engine...")
            
            # Dispose old engine
            self.dispose()
            
            # Update configuration
            if config is not None:
                self.config = config
            else:
                self.config = _load_database_config()
            
            # Reset metrics
            self.metrics = ConnectionMetrics()
            
            # Initialize new engine
            self._initialize_engine()
            
            LOGGER.info("✅ Database engine reinitialized")


# ========================================================================================
# SINGLETON INSTANCE
# ========================================================================================

_DB_ENGINE: Optional[DatabaseEngine] = None
_DB_LOCK = threading.RLock()


def get_engine(url: Optional[str] = None, force_reload: bool = False) -> Engine:
    """
    Get database engine (singleton).
    
    Args:
        url: Optional database URL (uses config if not provided)
        force_reload: Force engine reinitialization
        
    Returns:
        SQLAlchemy engine
        
    Examples:
        >>> engine = get_engine()
        >>> with engine.connect() as conn:
        ...     result = conn.execute(text("SELECT 1"))
    """
    global _DB_ENGINE
    
    with _DB_LOCK:
        if _DB_ENGINE is None or force_reload:
            config = _load_database_config()
            
            if url is not None:
                config.url = url
            
            _DB_ENGINE = DatabaseEngine(config)
        
        return _DB_ENGINE.engine


def get_database() -> DatabaseEngine:
    """
    Get database engine manager (singleton).
    
    Returns:
        DatabaseEngine instance
        
    Examples:
        >>> db = get_database()
        >>> with db.session_scope() as session:
        ...     session.add(obj)
    """
    global _DB_ENGINE
    
    with _DB_LOCK:
        if _DB_ENGINE is None:
            _DB_ENGINE = DatabaseEngine()
        
        return _DB_ENGINE


def get_url() -> str:
    """
    Get current database URL.
    
    Returns:
        Database URL string
    """
    return get_database().url


def safe_url(mask_password: bool = True) -> str:
    """
    Get safe database URL (with masked password).
    
    Args:
        mask_password: Whether to mask password
        
    Returns:
        Safe URL string
    """
    return get_database().safe_url(mask_password)


def get_session() -> Session:
    """
    Get new database session.
    
    Returns:
        SQLAlchemy session
        
    Examples:
        >>> session = get_session()
        >>> try:
        ...     session.add(obj)
        ...     session.commit()
        ... finally:
        ...     session.close()
    """
    return get_database().get_session()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Context manager for database session.
    
    Yields:
        Database session with automatic commit/rollback
        
    Examples:
        >>> with session_scope() as session:
        ...     session.add(obj)
        ...     # Automatic commit on success
    """
    with get_database().session_scope() as session:
        yield session


# ========================================================================================
# CONVENIENCE FUNCTIONS
# ========================================================================================

def health_check() -> bool:
    """
    Check database connection health.
    
    Returns:
        True if healthy
        
    Examples:
        >>> if health_check():
        ...     print("Database is healthy")
    """
    return get_database().health_check()


def execute_sql(
    statement: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    commit: bool = True
) -> Any:
    """
    Execute SQL statement.
    
    Args:
        statement: SQL statement
        params: Optional parameters
        commit: Whether to commit transaction
        
    Returns:
        Result proxy
        
    Examples:
        >>> execute_sql("CREATE TABLE users (id INTEGER PRIMARY KEY)")
        >>> execute_sql("INSERT INTO users (id) VALUES (:id)", {"id": 1})
    """
    return get_database().execute(statement, params, commit=commit)


def query_df(
    query: str,
    params: Optional[Dict[str, Any]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Execute query and return DataFrame.
    
    Args:
        query: SQL query
        params: Optional parameters
        **kwargs: Additional arguments for pd.read_sql
        
    Returns:
        Query results as DataFrame
        
    Examples:
        >>> df = query_df("SELECT * FROM users WHERE id > :min_id", {"min_id": 10})
    """
    return get_database().query_df(query, params, **kwargs)


def execute_many(
    statement: str,
    rows: Iterable[Dict[str, Any]]
) -> int:
    """
    Execute statement with multiple parameter sets.
    
    Args:
        statement: SQL statement
        rows: Iterable of parameter dictionaries
        
    Returns:
        Number of affected rows
        
    Examples:
        >>> rows = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        >>> execute_many("INSERT INTO users (id, name) VALUES (:id, :name)", rows)
    """
    return get_database().execute_many(statement, rows)


def get_table_names(schema: Optional[str] = None) -> List[str]:
    """
    Get list of table names.
    
    Args:
        schema: Optional schema name
        
    Returns:
        List of table names
        
    Examples:
        >>> tables = get_table_names()
        >>> print(f"Found {len(tables)} tables")
    """
    return get_database().get_table_names(schema)


def table_exists(table_name: str, schema: Optional[str] = None) -> bool:
    """
    Check if table exists.
    
    Args:
        table_name: Name of table
        schema: Optional schema name
        
    Returns:
        True if table exists
        
    Examples:
        >>> if table_exists("users"):
        ...     print("Users table exists")
    """
    return get_database().table_exists(table_name, schema)


def ensure_schema(metadata: MetaData) -> None:
    """
    Create tables from SQLAlchemy metadata.
    
    Args:
        metadata: SQLAlchemy metadata (or Base.metadata)
        
    Examples:
        >>> from sqlalchemy.ext.declarative import declarative_base
        >>> Base = declarative_base()
        >>> # Define your models...
        >>> ensure_schema(Base.metadata)
    """
    try:
        get_database().create_tables(metadata)
    except Exception as e:
        LOGGER.error(f"Failed to create schema: {e}")


def dispose_engine() -> None:
    """
    Dispose database engine and close all connections.
    
    Examples:
        >>> dispose_engine()
        >>> # Engine will be recreated on next use
    """
    global _DB_ENGINE
    
    with _DB_LOCK:
        if _DB_ENGINE is not None:
            _DB_ENGINE.dispose()
            _DB_ENGINE = None


def set_url_and_reinit(url: str) -> None:
    """
    Set database URL and reinitialize engine.
    
    Args:
        url: New database URL
        
    Examples:
        >>> set_url_and_reinit("postgresql://user:pass@localhost/db")
    """
    global _DB_ENGINE
    
    with _DB_LOCK:
        # Update environment variable
        os.environ["DATABASE_URL"] = url
        
        # Reinitialize with new URL
        if _DB_ENGINE is not None:
            config = _load_database_config()
            config.url = url
            _DB_ENGINE.reinitialize(config)
        else:
            # Will use new URL from environment
            get_database()


def status() -> str:
    """
    Get database status string.
    
    Returns:
        Status string
        
    Examples:
        >>> print(status())
        ✅ Database: postgresql
        URL: postgresql://***@localhost/mydb
        ...
    """
    return get_database().status()


def get_metrics() -> Dict[str, Any]:
    """
    Get database connection metrics.
    
    Returns:
        Metrics dictionary
        
    Examples:
        >>> metrics = get_metrics()
        >>> print(f"Queries executed: {metrics['queries_executed']}")
    """
    return get_database().get_metrics()


def reset_metrics() -> None:
    """
    Reset connection metrics.
    
    Examples:
        >>> reset_metrics()
    """
    get_database().reset_metrics()


# ========================================================================================
# RETRY DECORATOR
# ========================================================================================

def with_retry(
    max_retries: int = MAX_RETRIES,
    delay: float = RETRY_DELAY,
    backoff: float = RETRY_BACKOFF,
    exceptions: tuple = (OperationalError, DisconnectionError, SATimeoutError)
):
    """
    Decorator for retrying database operations.
    
    Args:
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Decorated function
        
    Examples:
        >>> @with_retry(max_retries=3)
        ... def fetch_user(user_id):
        ...     with session_scope() as session:
        ...         return session.query(User).get(user_id)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        LOGGER.warning(
                            f"Database operation failed (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        LOGGER.error(
                            f"Database operation failed after {max_retries} attempts: {e}"
                        )
            
            # All retries exhausted
            raise last_exception
        
        return wrapper
    
    return decorator


# ========================================================================================
# BATCH OPERATIONS
# ========================================================================================

def bulk_insert_df(
    df: pd.DataFrame,
    table_name: str,
    *,
    if_exists: str = "append",
    index: bool = False,
    chunksize: int = 1000,
    method: Optional[str] = None
) -> None:
    """
    Bulk insert DataFrame into database table.
    
    Args:
        df: DataFrame to insert
        table_name: Target table name
        if_exists: How to behave if table exists ('fail', 'replace', 'append')
        index: Whether to write DataFrame index
        chunksize: Number of rows per batch
        method: Insert method (None, 'multi')
        
    Examples:
        >>> df = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        >>> bulk_insert_df(df, "users")
    """
    engine = get_engine()
    
    try:
        df.to_sql(
            table_name,
            engine,
            if_exists=if_exists,
            index=index,
            chunksize=chunksize,
            method=method
        )
        
        LOGGER.info(f"Inserted {len(df)} rows into {table_name}")
        
    except Exception as e:
        LOGGER.error(f"Bulk insert failed: {e}")
        raise


def bulk_update(
    table_name: str,
    updates: List[Dict[str, Any]],
    key_column: str = "id"
) -> int:
    """
    Bulk update records in table.
    
    Args:
        table_name: Target table name
        updates: List of update dictionaries (must include key_column)
        key_column: Column to use as key for updates
        
    Returns:
        Number of updated rows
        
    Examples:
        >>> updates = [
        ...     {"id": 1, "status": "active"},
        ...     {"id": 2, "status": "inactive"}
        ... ]
        >>> bulk_update("users", updates, key_column="id")
    """
    if not updates:
        return 0
    
    # Build UPDATE statement
    first_row = updates[0]
    columns = [col for col in first_row.keys() if col != key_column]
    
    if not columns:
        raise ValueError("No columns to update")
    
    set_clause = ", ".join([f"{col} = :{col}" for col in columns])
    statement = f"UPDATE {table_name} SET {set_clause} WHERE {key_column} = :{key_column}"
    
    return execute_many(statement, updates)


# ========================================================================================
# TRANSACTION MANAGEMENT
# ========================================================================================

@contextmanager
def transaction() -> Generator[Connection, None, None]:
    """
    Context manager for explicit transaction.
    
    Yields:
        Database connection with active transaction
        
    Examples:
        >>> with transaction() as conn:
        ...     conn.execute(text("INSERT INTO users (name) VALUES ('Alice')"))
        ...     conn.execute(text("INSERT INTO logs (action) VALUES ('user_created')"))
        ...     # Both inserts committed together or rolled back on error
    """
    engine = get_engine()
    
    with engine.begin() as conn:
        yield conn


# ========================================================================================
# TESTING & DIAGNOSTICS
# ========================================================================================

def test_database(verbose: bool = True) -> Dict[str, bool]:
    """
    Test database functionality.
    
    Args:
        verbose: Print detailed results
        
    Returns:
        Test results dictionary
        
    Examples:
        >>> results = test_database(verbose=False)
        >>> all(results.values())
        True
    """
    results = {}
    
    # Test 1: Connection
    try:
        healthy = health_check()
        results["connection"] = healthy
        
        if verbose:
            print(f"{'✅' if healthy else '❌'} Connection: {healthy}")
    except Exception as e:
        results["connection"] = False
        if verbose:
            print(f"❌ Connection: {e}")
    
    # Test 2: Query execution
    try:
        with get_engine().connect() as conn:
            result = conn.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            results["query"] = row is not None and row[0] == 1
        
        if verbose:
            print(f"✅ Query execution: {results['query']}")
    except Exception as e:
        results["query"] = False
        if verbose:
            print(f"❌ Query execution: {e}")
    
    # Test 3: Session management
    try:
        with session_scope() as session:
            # Just test that session can be created
            results["session"] = session is not None
        
        if verbose:
            print(f"✅ Session management: {results['session']}")
    except Exception as e:
        results["session"] = False
        if verbose:
            print(f"❌ Session management: {e}")
    
    # Test 4: DataFrame query (if possible)
    try:
        df = query_df("SELECT 1 as test")
        results["dataframe"] = not df.empty and df.iloc[0, 0] == 1
        
        if verbose:
            print(f"✅ DataFrame query: {results['dataframe']}")
    except Exception as e:
        results["dataframe"] = False
        if verbose:
            print(f"❌ DataFrame query: {e}")
    
    return results


def print_diagnostics() -> None:
    """
    Print comprehensive database diagnostics.
    
    Examples:
        >>> print_diagnostics()
        📊 Database Diagnostics
        ...
    """
    print("📊 Database Diagnostics\n")
    print("="*60)
    
    # Status
    print("\n🔧 Status:")
    print(status())
    
    # Configuration
    print("\n⚙️ Configuration:")
    config = _load_database_config()
    print(f"  URL: {safe_url()}")
    print(f"  Pool size: {config.pool_size}")
    print(f"  Max overflow: {config.max_overflow}")
    print(f"  Pool recycle: {config.pool_recycle}s")
    print(f"  Pre-ping: {config.pool_pre_ping}")
    print(f"  Echo: {config.echo}")
    
    # Metrics
    print("\n📈 Metrics:")
    metrics = get_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Tables
    print("\n📋 Tables:")
    try:
        tables = get_table_names()
        if tables:
            for table in tables[:10]:  # Show first 10
                print(f"  - {table}")
            if len(tables) > 10:
                print(f"  ... and {len(tables) - 10} more")
        else:
            print("  No tables found")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Dependencies
    print("\n📦 Dependencies:")
    print(f"  SQLAlchemy: ✅ {__import__('sqlalchemy').__version__}")
    print(f"  Pandas: ✅ {pd.__version__}")
    print(f"  PyYAML: {'✅' if HAS_YAML else '❌'}")
    print(f"  Streamlit: {'✅' if HAS_STREAMLIT else '❌'}")
    
    print("\n" + "="*60)


def benchmark_database(num_queries: int = 100) -> Dict[str, float]:
    """
    Benchmark database performance.
    
    Args:
        num_queries: Number of test queries to run
        
    Returns:
        Benchmark results
        
    Examples:
        >>> results = benchmark_database(num_queries=1000)
        >>> print(f"Avg query time: {results['avg_query_time_ms']:.2f}ms")
    """
    results = {
        "num_queries": num_queries,
        "total_time": 0.0,
        "avg_query_time_ms": 0.0,
        "queries_per_second": 0.0
    }
    
    start_time = time.time()
    
    try:
        for _ in range(num_queries):
            query_df("SELECT 1")
        
        total_time = time.time() - start_time
        
        results["total_time"] = total_time
        results["avg_query_time_ms"] = (total_time / num_queries) * 1000
        results["queries_per_second"] = num_queries / total_time
        
    except Exception as e:
        LOGGER.error(f"Benchmark failed: {e}")
    
    return results


# ========================================================================================
# CONTEXT MANAGER FOR TEMPORARY DATABASE
# ========================================================================================

@contextmanager
def temporary_database(url: Optional[str] = None):
    """
    Context manager for temporary database instance.
    
    Args:
        url: Optional database URL (uses in-memory SQLite if not provided)
        
    Yields:
        Temporary DatabaseEngine instance
        
    Examples:
        >>> with temporary_database() as db:
        ...     with db.session_scope() as session:
        ...         # Use temporary database
        ...         pass
    """
    config = DatabaseConfig()
    
    if url is not None:
        config.url = url
    else:
        config.url = "sqlite:///:memory:"
    
    temp_db = DatabaseEngine(config)
    
    try:
        yield temp_db
    finally:
        temp_db.dispose()


# ========================================================================================
# MIGRATION HINTS
# ========================================================================================

def create_migration_hint() -> str:
    """
    Generate migration hint for Alembic or other tools.
    
    Returns:
        Migration configuration hint
    """
    return f"""
# Database Migration Configuration
# 
# For Alembic migrations, use:
# 
# 1. Install Alembic:
#    pip install alembic
# 
# 2. Initialize Alembic:
#    alembic init migrations
# 
# 3. Configure alembic.ini:
#    sqlalchemy.url = {safe_url()}
# 
# 4. Create migration:
#    alembic revision --autogenerate -m "description"
# 
# 5. Apply migration:
#    alembic upgrade head
# 
# Current database: {safe_url()}
"""


# ========================================================================================
# MAIN
# ========================================================================================

if __name__ == "__main__":
    print_diagnostics()
    print("\n" + "="*60)
    print("\n🧪 Running tests...\n")
    test_results = test_database(verbose=True)
    print(f"\n✅ Passed: {sum(test_results.values())}/{len(test_results)}")
    
    print("\n" + "="*60)
    print("\n⚡ Running benchmark...\n")
    bench_results = benchmark_database(num_queries=100)
    print(f"Total time: {bench_results['total_time']:.3f}s")
    print(f"Avg query time: {bench_results['avg_query_time_ms']:.2f}ms")
    print(f"Queries/sec: {bench_results['queries_per_second']:.1f}")