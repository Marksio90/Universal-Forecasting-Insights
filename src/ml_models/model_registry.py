"""
Model Registry Engine - Centralne zarządzanie zapisanymi modelami ML.

Funkcjonalności:
- Rejestr wszystkich wytrenowanych modeli z metadanymi
- Wyszukiwanie i filtrowanie modeli
- Automatyczne śledzenie metryk i wersji
- Idempotentne operacje (update vs insert)
- Zarządzanie cyklem życia modelu (register, load, delete)
- Best model selection z różnymi metrykami
- Orphan detection i cleanup
- Schema versioning dla backward compatibility
- Atomic file operations dla bezpieczeństwa
- Comprehensive statistics
"""

from __future__ import annotations

import json
import uuid
import hashlib
import pathlib
import datetime as dt
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Literal, Callable
from dataclasses import dataclass, field, asdict

import joblib
import pandas as pd

# ========================================================================================
# KONFIGURACJA
# ========================================================================================

# Ścieżki
MODELS_DIR = pathlib.Path(__file__).resolve().parents[2] / "models" / "trained_models"
REGISTRY_FILE = MODELS_DIR / "registry.json"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Schema
REGISTRY_SCHEMA_VERSION = 2

# Backup
MAX_BACKUPS = 5

# Supported formats
MODEL_FORMATS = ("joblib", "pickle", "h5", "onnx")

# Default metrics
DEFAULT_REGRESSION_METRIC = "rmse"
DEFAULT_CLASSIFICATION_METRIC = "f1_weighted"

# Metric directions (True = higher is better)
METRIC_DIRECTIONS = {
    "accuracy": True,
    "balanced_accuracy": True,
    "f1_weighted": True,
    "f1_macro": True,
    "f1_micro": True,
    "precision_weighted": True,
    "recall_weighted": True,
    "roc_auc": True,
    "r2": True,
    # Lower is better
    "rmse": False,
    "mae": False,
    "mse": False,
    "mape": False,
    "smape": False,
    "mase": False,
}

# Types
ProblemType = Literal["classification", "regression", "forecasting", "clustering"]
SortOrder = Literal["asc", "desc"]

# ========================================================================================
# LOGGING
# ========================================================================================

def get_logger(name: str = "model_registry", level: int = logging.INFO) -> logging.Logger:
    """
    Konfiguruje i zwraca logger bez duplikatów handlerów.
    
    Args:
        name: Nazwa loggera
        level: Poziom logowania
        
    Returns:
        Skonfigurowany logger
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    
    return logger


LOGGER = get_logger()


# ========================================================================================
# DATACLASSES
# ========================================================================================

@dataclass
class ModelEntry:
    """
    Wpis w rejestrze modelu.
    
    Attributes:
        id: Unikalny identyfikator modelu
        schema_version: Wersja schematu rejestru
        path: Ścieżka do pliku modelu (względna lub absolutna)
        target: Nazwa kolumny celu
        problem_type: Typ problemu (classification/regression/forecasting)
        created_at: Timestamp utworzenia
        updated_at: Timestamp ostatniej aktualizacji
        metrics: Słownik z metrykami
        columns: Lista nazw kolumn cech
        columns_hash: Hash kolumn dla szybkiego porównania
        best_estimator: Nazwa najlepszego estymatora
        model_format: Format zapisu (joblib/pickle/h5/onnx)
        tags: Lista tagów dla organizacji
        extra: Dodatkowe metadane
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    schema_version: int = REGISTRY_SCHEMA_VERSION
    path: str = ""
    target: Optional[str] = None
    problem_type: Optional[str] = None
    created_at: str = field(default_factory=lambda: _get_timestamp())
    updated_at: str = field(default_factory=lambda: _get_timestamp())
    metrics: Dict[str, float] = field(default_factory=dict)
    columns: Optional[List[str]] = None
    columns_hash: Optional[str] = None
    best_estimator: Optional[str] = None
    model_format: str = "joblib"
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje do słownika."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelEntry:
        """Tworzy z słownika."""
        return cls(**{
            k: v for k, v in data.items()
            if k in cls.__dataclass_fields__
        })


@dataclass
class RegistryStats:
    """Statystyki rejestru."""
    total_models: int
    by_problem_type: Dict[str, int]
    by_target: Dict[str, int]
    by_format: Dict[str, int]
    by_tags: Dict[str, int]
    last_created_at: Optional[str]
    last_updated_at: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje do słownika."""
        return asdict(self)


# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

def _get_timestamp() -> str:
    """
    Zwraca aktualny timestamp w formacie ISO.
    
    Returns:
        Timestamp string
    """
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _hash_columns(columns: Optional[List[str]]) -> Optional[str]:
    """
    Generuje hash z listy kolumn dla szybkiego porównania.
    
    Args:
        columns: Lista nazw kolumn
        
    Returns:
        16-znakowy hash hex lub None
    """
    if not columns:
        return None
    
    hasher = hashlib.sha256()
    for col in sorted(columns):  # Sort dla konsystencji
        hasher.update(str(col).encode("utf-8"))
    
    return hasher.hexdigest()[:16]


def _normalize_path(path: Union[str, pathlib.Path]) -> str:
    """
    Normalizuje ścieżkę do formy względnej jeśli możliwe.
    
    Args:
        path: Ścieżka do normalizacji
        
    Returns:
        Znormalizowana ścieżka jako string
    """
    path_obj = pathlib.Path(path)
    
    try:
        # Spróbuj zapisać jako względną do MODELS_DIR
        relative = path_obj.relative_to(MODELS_DIR)
        return str(relative)
    except ValueError:
        # Jeśli nie da się, zwróć absolutną
        return str(path_obj.resolve())


def _resolve_path(path: Union[str, pathlib.Path]) -> pathlib.Path:
    """
    Rozwiązuje ścieżkę do absolutnej formy.
    
    Args:
        path: Ścieżka do rozwiązania
        
    Returns:
        Absolutna ścieżka
    """
    path_obj = pathlib.Path(path)
    
    if path_obj.is_absolute():
        return path_obj
    
    # Względna do MODELS_DIR
    return (MODELS_DIR / path_obj).resolve()


def _normalize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizuje wpis rejestru do aktualnego schematu.
    
    Args:
        entry: Surowy wpis
        
    Returns:
        Znormalizowany wpis
    """
    # Domyślne wartości
    defaults = ModelEntry(path="").to_dict()
    
    # Merge z zachowaniem istniejących wartości
    normalized = {**defaults, **entry}
    
    # Auto-compute columns_hash jeśli brak
    if normalized.get("columns") and not normalized.get("columns_hash"):
        normalized["columns_hash"] = _hash_columns(normalized["columns"])
    
    # Zapewnij schema_version
    if not normalized.get("schema_version"):
        normalized["schema_version"] = REGISTRY_SCHEMA_VERSION
    
    # Zapewnij updated_at
    if not normalized.get("updated_at"):
        normalized["updated_at"] = normalized.get("created_at", _get_timestamp())
    
    return normalized


# ========================================================================================
# FILE I/O
# ========================================================================================

def _atomic_write(path: pathlib.Path, content: str) -> None:
    """
    Atomiczny zapis do pliku z backupem.
    
    Args:
        path: Ścieżka do pliku
        content: Zawartość do zapisu
    """
    # Upewnij się że katalog istnieje
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Backup istniejącego pliku
    if path.exists():
        try:
            _create_backup(path)
        except Exception as e:
            LOGGER.warning(f"Nie udało się utworzyć backupu: {e}")
    
    # Atomiczny zapis przez temp file
    temp_path = path.with_suffix(path.suffix + ".tmp")
    
    try:
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(path)
        LOGGER.debug(f"Zapisano atomowo: {path}")
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Nie udało się zapisać pliku: {e}")


def _create_backup(path: pathlib.Path) -> None:
    """
    Tworzy backup pliku z rotacją.
    
    Args:
        path: Ścieżka do pliku
    """
    if not path.exists():
        return
    
    backup_dir = path.parent / "backups"
    backup_dir.mkdir(exist_ok=True)
    
    # Backup z timestampem
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{path.stem}_{timestamp}{path.suffix}"
    backup_path = backup_dir / backup_name
    
    # Kopiuj
    backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    
    # Rotacja - zachowaj tylko MAX_BACKUPS
    backups = sorted(backup_dir.glob(f"{path.stem}_*{path.suffix}"))
    if len(backups) > MAX_BACKUPS:
        for old_backup in backups[:-MAX_BACKUPS]:
            try:
                old_backup.unlink()
                LOGGER.debug(f"Usunięto stary backup: {old_backup}")
            except Exception as e:
                LOGGER.warning(f"Nie udało się usunąć backupu {old_backup}: {e}")


def _read_registry() -> List[Dict[str, Any]]:
    """
    Wczytuje rejestr z pliku.
    
    Returns:
        Lista wpisów rejestru
    """
    if not REGISTRY_FILE.exists():
        LOGGER.debug("Rejestr nie istnieje, zwracam pustą listę")
        return []
    
    try:
        content = REGISTRY_FILE.read_text(encoding="utf-8")
        data = json.loads(content)
        
        # Obsługa różnych formatów
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Stary format - pojedynczy wpis
            LOGGER.debug("Wykryto stary format rejestru (dict), konwertuję do listy")
            return [data]
        else:
            LOGGER.warning(f"Nieoczekiwany format rejestru: {type(data)}")
            return []
            
    except json.JSONDecodeError as e:
        LOGGER.error(f"Błąd parsowania JSON rejestru: {e}")
        return []
    except Exception as e:
        LOGGER.error(f"Błąd odczytu rejestru: {e}")
        return []


def _write_registry(entries: List[Dict[str, Any]]) -> None:
    """
    Zapisuje rejestr do pliku.
    
    Args:
        entries: Lista wpisów do zapisu
    """
    try:
        # Serialize z pretty printing
        content = json.dumps(entries, indent=2, ensure_ascii=False, default=str)
        _atomic_write(REGISTRY_FILE, content)
        LOGGER.debug(f"Zapisano rejestr: {len(entries)} wpisów")
    except Exception as e:
        LOGGER.error(f"Nie udało się zapisać rejestru: {e}")
        raise


# ========================================================================================
# FILTERING & SORTING
# ========================================================================================

def _matches_filter(entry: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """
    Sprawdza czy wpis spełnia filtry.
    
    Args:
        entry: Wpis do sprawdzenia
        filters: Słownik z filtrami
        
    Returns:
        True jeśli spełnia wszystkie filtry
    """
    for key, value in filters.items():
        # Brak klucza = nie spełnia
        if key not in entry:
            return False
        
        entry_value = entry[key]
        
        # Lista/zbiór wartości (OR logic)
        if isinstance(value, (list, tuple, set)):
            if entry_value not in value:
                return False
        
        # Zagnieżdżone filtry (np. dla metrics)
        elif isinstance(value, dict):
            if not isinstance(entry_value, dict):
                return False
            
            for sub_key, sub_condition in value.items():
                sub_value = entry_value.get(sub_key)
                
                # Operator comparison
                if isinstance(sub_condition, tuple) and len(sub_condition) == 2:
                    operator, threshold = sub_condition
                    
                    if sub_value is None:
                        return False
                    
                    try:
                        if operator == "<" and not (sub_value < threshold):
                            return False
                        elif operator == ">" and not (sub_value > threshold):
                            return False
                        elif operator == "<=" and not (sub_value <= threshold):
                            return False
                        elif operator == ">=" and not (sub_value >= threshold):
                            return False
                        elif operator == "==" and not (sub_value == threshold):
                            return False
                        elif operator == "!=" and not (sub_value != threshold):
                            return False
                    except (TypeError, ValueError):
                        return False
                
                # Exact match
                else:
                    if sub_value != sub_condition:
                        return False
        
        # Exact match
        else:
            if entry_value != value:
                return False
    
    return True


def _sort_entries(
    entries: List[Dict[str, Any]],
    sort_by: str,
    reverse: bool
) -> List[Dict[str, Any]]:
    """
    Sortuje wpisy według klucza.
    
    Args:
        entries: Lista wpisów
        sort_by: Klucz do sortowania
        reverse: Czy odwrócić kolejność
        
    Returns:
        Posortowana lista
    """
    def sort_key(entry: Dict[str, Any]) -> Any:
        value = entry.get(sort_by)
        
        # None na końcu
        if value is None:
            return "" if not reverse else "zzzzzz"
        
        return value
    
    try:
        return sorted(entries, key=sort_key, reverse=reverse)
    except Exception as e:
        LOGGER.warning(f"Nie udało się posortować wpisów: {e}")
        return entries


# ========================================================================================
# MAIN API
# ========================================================================================

def list_models(
    filters: Optional[Dict[str, Any]] = None,
    sort_by: str = "created_at",
    reverse: bool = True,
) -> List[Dict[str, Any]]:
    """
    Zwraca listę modeli z rejestru z opcjonalnym filtrowaniem i sortowaniem.
    
    Args:
        filters: Słownik z filtrami, obsługuje:
            - Proste equality: {"target": "price"}
            - Lista wartości (OR): {"problem_type": ["classification", "regression"]}
            - Zagnieżdżone (metrics): {"metrics": {"rmse": ("<", 1.0)}}
        sort_by: Klucz do sortowania (default: "created_at")
        reverse: Odwrócona kolejność (default: True = newest first)
        
    Returns:
        Lista znormalizowanych wpisów
        
    Example:
        >>> # Wszystkie modele
        >>> models = list_models()
        >>> 
        >>> # Filtrowanie po targecie
        >>> models = list_models(filters={"target": "sales"})
        >>> 
        >>> # Modele klasyfikacyjne z accuracy > 0.9
        >>> models = list_models(filters={
        ...     "problem_type": "classification",
        ...     "metrics": {"accuracy": (">", 0.9)}
        ... })
        >>> 
        >>> # Sortowanie po metryce
        >>> models = list_models(sort_by="metrics.rmse", reverse=False)
    """
    # Wczytaj i normalizuj
    entries = [_normalize_entry(e) for e in _read_registry()]
    
    # Filtrowanie
    if filters:
        entries = [e for e in entries if _matches_filter(e, filters)]
    
    # Sortowanie
    entries = _sort_entries(entries, sort_by, reverse)
    
    LOGGER.debug(f"list_models: {len(entries)} wyników po filtrach i sortowaniu")
    return entries


def register_model(
    *,
    model_path: Union[str, pathlib.Path],
    target: str,
    problem_type: ProblemType,
    metrics: Optional[Dict[str, float]] = None,
    columns: Optional[List[str]] = None,
    best_estimator: Optional[str] = None,
    tags: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
    model_format: str = "joblib"
) -> Dict[str, Any]:
    """
    Rejestruje nowy model lub aktualizuje istniejący.
    
    Operacja jest idempotentna - jeśli model o tej samej ścieżce już istnieje,
    zostanie zaktualizowany z zachowaniem id i created_at.
    
    Args:
        model_path: Ścieżka do pliku modelu
        target: Nazwa kolumny celu
        problem_type: Typ problemu
        metrics: Słownik z metrykami (optional)
        columns: Lista nazw kolumn cech (optional)
        best_estimator: Nazwa najlepszego estymatora (optional)
        tags: Lista tagów (optional)
        extra: Dodatkowe metadane (optional)
        model_format: Format pliku (default: "joblib")
        
    Returns:
        Znormalizowany wpis rejestru
        
    Raises:
        ValueError: Jeśli parametry są nieprawidłowe
        FileNotFoundError: Jeśli plik modelu nie istnieje
        
    Example:
        >>> entry = register_model(
        ...     model_path="model_lgbm_12345.joblib",
        ...     target="price",
        ...     problem_type="regression",
        ...     metrics={"rmse": 123.45, "r2": 0.89},
        ...     columns=["feature1", "feature2"],
        ...     best_estimator="lgbm",
        ...     tags=["production", "v1"]
        ... )
        >>> print(entry["id"])
    """
    # Walidacja
    if not target:
        raise ValueError("Parametr 'target' jest wymagany")
    
    if not problem_type:
        raise ValueError("Parametr 'problem_type' jest wymagany")
    
    if model_format not in MODEL_FORMATS:
        LOGGER.warning(
            f"Nieznany format modelu: {model_format}. "
            f"Obsługiwane: {MODEL_FORMATS}"
        )
    
    # Sprawdź czy plik istnieje
    resolved_path = _resolve_path(model_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Plik modelu nie istnieje: {resolved_path}")
    
    # Normalizuj ścieżkę
    normalized_path = _normalize_path(model_path)
    
    # Utwórz wpis
    entry = ModelEntry(
        path=normalized_path,
        target=target,
        problem_type=problem_type,
        metrics=metrics or {},
        columns=columns,
        columns_hash=_hash_columns(columns),
        best_estimator=best_estimator,
        tags=tags or [],
        extra=extra or {},
        model_format=model_format
    )
    
    # Wczytaj istniejący rejestr
    entries = _read_registry()
    
    # Szukaj istniejącego wpisu (idempotency)
    updated = False
    for i, existing in enumerate(entries):
        if existing.get("path") == normalized_path:
            # Aktualizuj z zachowaniem id i created_at
            entry_dict = entry.to_dict()
            entry_dict["id"] = existing.get("id", entry.id)
            entry_dict["created_at"] = existing.get("created_at", entry.created_at)
            entry_dict["updated_at"] = _get_timestamp()
            
            entries[i] = entry_dict
            updated = True
            LOGGER.info(f"Zaktualizowano model: {normalized_path} (id={entry_dict['id']})")
            break
    
    # Dodaj nowy wpis
    if not updated:
        entry_dict = entry.to_dict()
        entries.append(entry_dict)
        LOGGER.info(f"Zarejestrowano nowy model: {normalized_path} (id={entry.id})")
    
    # Zapisz
    _write_registry(entries)
    
    return _normalize_entry(entry.to_dict())


def load_model(reference: str) -> Any:
    """
    Ładuje model z rejestru.
    
    Wspiera różne typy referencji:
    - ID modelu z rejestru (np. "a1b2c3d4e5f6")
    - Nazwa pliku (np. "model_lgbm_12345.joblib")
    - Pełna ścieżka
    
    Args:
        reference: Referencja do modelu
        
    Returns:
        Załadowany model (jeśli payload ma klucz 'model', zwraca payload['model'])
        
    Raises:
        FileNotFoundError: Jeśli model nie został znaleziony
        
    Example:
        >>> model = load_model("a1b2c3d4e5f6")  # By ID
        >>> model = load_model("model_lgbm.joblib")  # By filename
        >>> predictions = model.predict(X_test)
    """
    # Szukaj po ID
    for entry in _read_registry():
        if entry.get("id") == reference:
            model_path = _resolve_path(entry["path"])
            LOGGER.debug(f"Znaleziono model po ID: {reference} → {model_path}")
            return _load_model_file(model_path)
    
    # Szukaj po ścieżce/nazwie
    resolved = _resolve_path(reference)
    if resolved.exists():
        LOGGER.debug(f"Znaleziono model po ścieżce: {reference} → {resolved}")
        return _load_model_file(resolved)
    
    # Nie znaleziono
    raise FileNotFoundError(
        f"Nie znaleziono modelu dla referencji: '{reference}'. "
        "Sprawdź ID, nazwę pliku lub ścieżkę."
    )


def _load_model_file(path: pathlib.Path) -> Any:
    """
    Ładuje model z pliku.
    
    Args:
        path: Ścieżka do pliku
        
    Returns:
        Załadowany model
    """
    try:
        obj = joblib.load(path)
        
        # Jeśli to payload dict z kluczem 'model', zwróć model
        if isinstance(obj, dict) and "model" in obj:
            return obj["model"]
        
        # W przeciwnym razie zwróć cały obiekt
        return obj
        
    except Exception as e:
        raise RuntimeError(f"Nie udało się wczytać modelu z {path}: {e}")


def delete_model(
    reference: str,
    remove_file: bool = True
) -> bool:
    """
    Usuwa model z rejestru i opcjonalnie plik.
    
    Args:
        reference: Referencja do modelu (ID, ścieżka lub nazwa)
        remove_file: Czy usunąć też plik modelu (default: True)
        
    Returns:
        True jeśli model został usunięty, False jeśli nie znaleziono
        
    Example:
        >>> # Usuń z rejestru i plik
        >>> deleted = delete_model("a1b2c3d4e5f6")
        >>> 
        >>> # Usuń tylko z rejestru
        >>> deleted = delete_model("model.joblib", remove_file=False)
    """
    entries = _read_registry()
    kept: List[Dict[str, Any]] = []
    deleted = False
    deleted_path: Optional[pathlib.Path] = None
    
    # Szukaj po ID lub ścieżce
    for entry in entries:
        if entry.get("id") == reference or entry.get("path") == reference:
            deleted = True
            deleted_path = _resolve_path(entry["path"])
            LOGGER.info(f"Usuwam model: {entry.get('path')} (id={entry.get('id')})")
            continue
        
        kept.append(entry)
    
    # Jeśli nie znaleziono, spróbuj jako ścieżkę
    if not deleted:
        reference_path = _resolve_path(reference)
        kept_final: List[Dict[str, Any]] = []
        
        for entry in kept:
            entry_path = _resolve_path(entry["path"])
            if entry_path == reference_path:
                deleted = True
                deleted_path = reference_path
                LOGGER.info(f"Usuwam model: {entry.get('path')} (id={entry.get('id')})")
                continue
            
            kept_final.append(entry)
        
        kept = kept_final
    
    # Zapisz jeśli coś usunięto
    if deleted:
        _write_registry(kept)
        
        # Usuń plik jeśli requested
        if remove_file and deleted_path and deleted_path.exists():
            try:
                deleted_path.unlink()
                LOGGER.info(f"Usunięto plik modelu: {deleted_path}")
            except Exception as e:
                LOGGER.warning(f"Nie udało się usunąć pliku {deleted_path}: {e}")
    
    return deleted


def get_best_model(
    *,
    target: Optional[str] = None,
    problem_type: Optional[ProblemType] = None,
    metric: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    """
    Zwraca najlepszy model według metryki.
    
    Domyślne metryki:
    - regression: rmse (minimize)
    - classification: f1_weighted (maximize)
    - forecasting: smape (minimize)
    
    Args:
        target: Filtruj po targecie (optional)
        problem_type: Filtruj po typie problemu (optional)
        metric: Nazwa metryki (optional, auto-detect jeśli None)
        tags: Filtruj po tagach - wszystkie muszą być obecne (optional)
        
    Returns:
        Dict z najlepszym modelem lub None jeśli brak
        
    Example:
        >>> # Najlepszy model dla danego targetu
        >>> best = get_best_model(target="sales")
        >>> 
        >>> # Najlepszy model klasyfikacyjny wg accuracy
        >>> best = get_best_model(
        ...     problem_type="classification",
        ...     metric="accuracy"
        ... )
        >>> 
        >>> # Najlepszy production model
        >>> best = get_best_model(tags=["production"])
    """
    # Przygotuj filtry
    filters: Dict[str, Any] = {}
    if target:
        filters["target"] = target
    if problem_type:
        filters["problem_type"] = problem_type
    
    # Lista modeli
    models = list_models(filters=filters if filters else None)
    
    if not models:
        LOGGER.debug("Brak modeli spełniających kryteria")
        return None
    
    # Filtrowanie po tagach
    if tags:
        models = [
            m for m in models
            if all(tag in (m.get("tags") or []) for tag in tags)
        ]
        
        if not models:
            LOGGER.debug(f"Brak modeli z tagami: {tags}")
            return None
    
    # Auto-detect metryki
    if metric is None:
        detected_problem_type = problem_type or models[0].get("problem_type", "")
        
        if "regress" in detected_problem_type.lower():
            metric = DEFAULT_REGRESSION_METRIC
        elif "forecast" in detected_problem_type.lower():
            metric = "smape"
        else:
            metric = DEFAULT_CLASSIFICATION_METRIC
        
        LOGGER.debug(f"Auto-detected metric: {metric}")
    
    # Określ kierunek (maximize czy minimize)
    maximize = METRIC_DIRECTIONS.get(metric, True)
    
    # Zbierz modele z metryką
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for model in models:
        value = (model.get("metrics") or {}).get(metric)
        
        if value is None:
            continue
        
        try:
            scored.append((float(value), model))
        except (TypeError, ValueError):
            LOGGER.warning(f"Nieprawidłowa wartość metryki {metric} dla modelu {model.get('id')}: {value}")
            continue
    
    if not scored:
        LOGGER.debug(f"Żaden model nie ma metryki: {metric}")
        return None
    
    # Wybierz najlepszy
    if maximize:
        best = max(scored, key=lambda x: x[0])
        LOGGER.info(f"Najlepszy model (max {metric}={best[0]:.4f}): {best[1].get('id')}")
    else:
        best = min(scored, key=lambda x: x[0])
        LOGGER.info(f"Najlepszy model (min {metric}={best[0]:.4f}): {best[1].get('id')}")
    
    return best[1]


def get_model_by_id(model_id: str) -> Optional[Dict[str, Any]]:
    """
    Zwraca wpis modelu po ID.
    
    Args:
        model_id: ID modelu
        
    Returns:
        Dict z wpisem lub None jeśli nie znaleziono
    """
    for entry in _read_registry():
        if entry.get("id") == model_id:
            return _normalize_entry(entry)
    
    return None


def update_model(
    model_id: str,
    **updates
) -> Optional[Dict[str, Any]]:
    """
    Aktualizuje metadane modelu.
    
    Args:
        model_id: ID modelu do aktualizacji
        **updates: Pola do aktualizacji (metrics, tags, extra, etc.)
        
    Returns:
        Zaktualizowany wpis lub None jeśli nie znaleziono
        
    Example:
        >>> # Aktualizuj metryki
        >>> update_model("a1b2c3", metrics={"rmse": 100.0})
        >>> 
        >>> # Dodaj tagi
        >>> update_model("a1b2c3", tags=["production", "v2"])
        >>> 
        >>> # Aktualizuj extra
        >>> update_model("a1b2c3", extra={"note": "best model so far"})
    """
    entries = _read_registry()
    updated = False
    result = None
    
    for i, entry in enumerate(entries):
        if entry.get("id") == model_id:
            # Aktualizuj pola
            for key, value in updates.items():
                if key in ("id", "created_at", "schema_version"):
                    # Nie pozwalaj na zmianę tych pól
                    LOGGER.warning(f"Pomijam próbę aktualizacji pola {key}")
                    continue
                
                entry[key] = value
            
            # Ustaw updated_at
            entry["updated_at"] = _get_timestamp()
            
            # Auto-compute columns_hash jeśli zaktualizowano columns
            if "columns" in updates:
                entry["columns_hash"] = _hash_columns(updates["columns"])
            
            entries[i] = entry
            updated = True
            result = _normalize_entry(entry)
            LOGGER.info(f"Zaktualizowano model: {model_id}")
            break
    
    if updated:
        _write_registry(entries)
    
    return result


# ========================================================================================
# STATISTICS
# ========================================================================================

def get_stats() -> RegistryStats:
    """
    Zwraca statystyki rejestru.
    
    Returns:
        RegistryStats object
        
    Example:
        >>> stats = get_stats()
        >>> print(f"Total models: {stats.total_models}")
        >>> print(f"By type: {stats.by_problem_type}")
    """
    entries = _read_registry()
    
    by_problem_type: Dict[str, int] = {}
    by_target: Dict[str, int] = {}
    by_format: Dict[str, int] = {}
    by_tags: Dict[str, int] = {}
    
    last_created = None
    last_updated = None
    
    for entry in entries:
        # Problem type
        ptype = entry.get("problem_type", "unknown")
        by_problem_type[ptype] = by_problem_type.get(ptype, 0) + 1
        
        # Target
        target = entry.get("target", "unknown")
        by_target[target] = by_target.get(target, 0) + 1
        
        # Format
        fmt = entry.get("model_format", "unknown")
        by_format[fmt] = by_format.get(fmt, 0) + 1
        
        # Tags
        for tag in entry.get("tags", []):
            by_tags[tag] = by_tags.get(tag, 0) + 1
        
        # Timestamps
        created = entry.get("created_at")
        if created and (last_created is None or created > last_created):
            last_created = created
        
        updated = entry.get("updated_at")
        if updated and (last_updated is None or updated > last_updated):
            last_updated = updated
    
    stats = RegistryStats(
        total_models=len(entries),
        by_problem_type=by_problem_type,
        by_target=by_target,
        by_format=by_format,
        by_tags=by_tags,
        last_created_at=last_created,
        last_updated_at=last_updated
    )
    
    LOGGER.debug(f"Stats: {stats.total_models} modeli w rejestrze")
    return stats


# ========================================================================================
# MAINTENANCE
# ========================================================================================

def prune_orphans() -> Dict[str, Any]:
    """
    Porządkuje rejestr - usuwa wpisy bez plików i identyfikuje pliki bez wpisów.
    
    Returns:
        Słownik z wynikami:
        - removed_entries: Liczba usuniętych wpisów
        - orphan_files: Lista plików bez wpisów
        - kept_entries: Liczba zachowanych wpisów
        
    Example:
        >>> result = prune_orphans()
        >>> print(f"Removed {result['removed_entries']} broken entries")
        >>> print(f"Found {len(result['orphan_files'])} orphan files")
    """
    entries = _read_registry()
    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    
    # Sprawdź każdy wpis
    for entry in entries:
        path = _resolve_path(entry.get("path", ""))
        
        if path.exists():
            kept.append(entry)
        else:
            removed.append(entry)
            LOGGER.warning(f"Usuwam wpis bez pliku: {entry.get('path')} (id={entry.get('id')})")
    
    # Zapisz oczyszczony rejestr
    if removed:
        _write_registry(kept)
        LOGGER.info(f"Usunięto {len(removed)} wpisów bez plików")
    
    # Znajdź orphan files (pliki bez wpisów)
    all_files = set(MODELS_DIR.glob("*.joblib"))
    registered_files = {_resolve_path(e["path"]) for e in kept}
    orphan_files = sorted([f.name for f in all_files - registered_files])
    
    if orphan_files:
        LOGGER.info(f"Znaleziono {len(orphan_files)} plików bez wpisów: {orphan_files}")
    
    return {
        "removed_entries": len(removed),
        "orphan_files": orphan_files,
        "kept_entries": len(kept),
    }


def clear_registry() -> None:
    """
    Czyści cały rejestr (nie usuwa plików).
    
    Użyj ostrożnie - usuwa wszystkie wpisy!
    """
    _write_registry([])
    LOGGER.warning("Rejestr wyczyszczony!")


def vacuum_registry() -> Dict[str, Any]:
    """
    Kompletny cleanup: usuwa orphan entries i orphan files.
    
    Returns:
        Słownik z wynikami operacji
        
    Example:
        >>> result = vacuum_registry()
        >>> print(f"Cleaned {result['files_deleted']} orphan files")
    """
    # Najpierw prune orphans
    prune_result = prune_orphans()
    
    # Usuń orphan files
    orphan_files = prune_result["orphan_files"]
    deleted_files: List[str] = []
    failed_files: List[str] = []
    
    for filename in orphan_files:
        filepath = MODELS_DIR / filename
        try:
            filepath.unlink()
            deleted_files.append(filename)
            LOGGER.info(f"Usunięto orphan file: {filename}")
        except Exception as e:
            failed_files.append(filename)
            LOGGER.error(f"Nie udało się usunąć {filename}: {e}")
    
    return {
        "removed_entries": prune_result["removed_entries"],
        "files_deleted": len(deleted_files),
        "files_failed": len(failed_files),
        "kept_entries": prune_result["kept_entries"],
    }


# ========================================================================================
# EXPORT / IMPORT
# ========================================================================================

def export_registry(output_path: Union[str, pathlib.Path]) -> None:
    """
    Eksportuje rejestr do pliku JSON.
    
    Args:
        output_path: Ścieżka do pliku wyjściowego
    """
    entries = _read_registry()
    output = pathlib.Path(output_path)
    
    content = json.dumps(entries, indent=2, ensure_ascii=False, default=str)
    output.write_text(content, encoding="utf-8")
    
    LOGGER.info(f"Wyeksportowano {len(entries)} wpisów do {output}")


def import_registry(
    input_path: Union[str, pathlib.Path],
    merge: bool = False
) -> int:
    """
    Importuje rejestr z pliku JSON.
    
    Args:
        input_path: Ścieżka do pliku wejściowego
        merge: Czy scalić z istniejącym rejestrem (default: False = replace)
        
    Returns:
        Liczba zaimportowanych wpisów
    """
    input_file = pathlib.Path(input_path)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Plik nie istnieje: {input_file}")
    
    # Wczytaj nowe wpisy
    content = input_file.read_text(encoding="utf-8")
    new_entries = json.loads(content)
    
    if not isinstance(new_entries, list):
        raise ValueError("Import file musi zawierać listę wpisów")
    
    # Normalizuj
    normalized = [_normalize_entry(e) for e in new_entries]
    
    if merge:
        # Scalanie - update istniejących, dodaj nowe
        existing = _read_registry()
        existing_by_id = {e.get("id"): e for e in existing}
        
        for entry in normalized:
            existing_by_id[entry["id"]] = entry
        
        final = list(existing_by_id.values())
    else:
        # Replace
        final = normalized
    
    _write_registry(final)
    
    LOGGER.info(f"Zaimportowano {len(normalized)} wpisów (merge={merge})")
    return len(normalized)


# ========================================================================================
# SEARCH & QUERY
# ========================================================================================

def search_models(
    query: str,
    fields: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Wyszukuje modele po tekście w wybranych polach.
    
    Args:
        query: Tekst do wyszukania (case-insensitive)
        fields: Lista pól do przeszukania (default: ["target", "best_estimator", "tags"])
        
    Returns:
        Lista modeli zawierających query
        
    Example:
        >>> # Szukaj "sales" w targecie, estimatorze i tagach
        >>> models = search_models("sales")
        >>> 
        >>> # Szukaj tylko w tagach
        >>> models = search_models("production", fields=["tags"])
    """
    if fields is None:
        fields = ["target", "best_estimator", "tags"]
    
    query_lower = query.lower()
    entries = _read_registry()
    results: List[Dict[str, Any]] = []
    
    for entry in entries:
        for field in fields:
            value = entry.get(field)
            
            if value is None:
                continue
            
            # String field
            if isinstance(value, str) and query_lower in value.lower():
                results.append(_normalize_entry(entry))
                break
            
            # List field (np. tags)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and query_lower in item.lower():
                        results.append(_normalize_entry(entry))
                        break
    
    LOGGER.debug(f"Wyszukiwanie '{query}': {len(results)} wyników")
    return results


def compare_models(
    model_ids: List[str],
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Porównuje metryki wielu modeli.
    
    Args:
        model_ids: Lista ID modeli do porównania
        metrics: Lista metryk do porównania (optional, default: wszystkie)
        
    Returns:
        DataFrame z porównaniem
        
    Raises:
        ImportError: Jeśli pandas nie jest zainstalowany
        
    Example:
        >>> comparison = compare_models(
        ...     ["model1_id", "model2_id", "model3_id"],
        ...     metrics=["rmse", "mae", "r2"]
        ... )
        >>> print(comparison)
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("compare_models wymaga pandas")
    
    entries = _read_registry()
    models_data: List[Dict[str, Any]] = []
    
    for model_id in model_ids:
        for entry in entries:
            if entry.get("id") == model_id:
                model_metrics = entry.get("metrics", {})
                
                data = {
                    "id": model_id,
                    "target": entry.get("target"),
                    "problem_type": entry.get("problem_type"),
                    "best_estimator": entry.get("best_estimator"),
                }
                
                # Dodaj metryki
                if metrics:
                    for metric in metrics:
                        data[metric] = model_metrics.get(metric)
                else:
                    data.update(model_metrics)
                
                models_data.append(data)
                break
    
    if not models_data:
        return pd.DataFrame()
    
    return pd.DataFrame(models_data)


# ========================================================================================
# BACKWARD COMPATIBILITY
# ========================================================================================

def stats() -> Dict[str, Any]:
    """Backward compatible stats function."""
    return get_stats().to_dict()


def list_models_legacy() -> List[Dict[str, Any]]:
    """Backward compatible list function."""
    return list_models()


def clear_registry_legacy() -> None:
    """Backward compatible clear function."""
    clear_registry()