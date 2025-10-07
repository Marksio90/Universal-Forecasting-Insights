from __future__ import annotations
import json, pathlib

MODELS_DIR = pathlib.Path(__file__).resolve().parents[2] / "models" / "trained_models"
REGISTRY = MODELS_DIR / "registry.json"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def list_models():
    if not REGISTRY.exists():
        return []
    try:
        data = json.loads(REGISTRY.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []

def clear_registry():
    REGISTRY.write_text("[]", encoding="utf-8")
