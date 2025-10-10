# src/utils/secrets.py
from __future__ import annotations
import os
from typing import Any, Optional

try:
    import streamlit as st
    _SECRETS = getattr(st, "secrets", {})  # MappingProxy (działa lokalnie i w Cloud)
except Exception:
    _SECRETS = {}

def _sget(path: str, default: Optional[Any] = None) -> Optional[Any]:
    """
    Czytaj klucz z st.secrets (sekcja.kod) z bezpiecznym fallbackiem do ENV.
    Przykłady: "OPENAI_API_KEY" lub "mlflow.TRACKING_URI" lub "auth.JWT_SECRET".
    """
    if "." in path:
        section, key = path.split(".", 1)
        val = _SECRETS.get(section, {}).get(key) if isinstance(_SECRETS.get(section, {}), dict) else None
        return val if val not in (None, "") else os.getenv(key.upper(), default)
    val = _SECRETS.get(path)
    return val if val not in (None, "") else os.getenv(path, default)

def prime_env_from_secrets() -> None:
    """
    Wypchnij krytyczne wartości do os.environ — biblioteki (MLflow, boto3 s3fs, itp.)
    tego wymagają. Ustawia TYLKO jeśli nie istnieje w ENV (os.environ.setdefault).
    """
    mapping = {
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
        # Domeny
        "domain.PUBLIC_BASE_URL": "PUBLIC_BASE_URL",
    }
    for src, env_key in mapping.items():
        val = _sget(src)
        if val is not None:
            os.environ.setdefault(env_key, str(val))

def get_secret(path: str, default: Optional[Any] = None) -> Optional[Any]:
    """Publiczny accessor (np. get_secret('alerts.SLACK_WEBHOOK_URL'))."""
    return _sget(path, default)
