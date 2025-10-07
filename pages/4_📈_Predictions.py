"""
Moduł Predictions (AutoML) PRO - Zaawansowane trenowanie modeli ML.

Funkcjonalności:
- AutoML pipeline z automatycznym wyborem algorytmu
- Pełna ewaluacja (classification/regression)
- Wizualizacje (confusion matrix, ROC, parity plot)
- Feature importance + SHAP values
- Multi-format export (model, metrics, predictions)
- Historia treningu
- Progress tracking
"""

from __future__ import annotations

import io
import json
import time
import logging
import hashlib
from typing import Optional, Literal, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from src.ml_models.automl_pipeline import train_automl
from src.utils.helpers import infer_problem_type

# ========================================================================================
# KONFIGURACJA
# ========================================================================================

logger = logging.getLogger(__name__)

# Limity bezpieczeństwa
MAX_TRAIN_ROWS = 1_000_000
MAX_FEATURES = 1000
MIN_SAMPLES = 10
MAX_SHAP_SAMPLES = 5000
DEFAULT_SHAP_SAMPLES = 2000

# Hinty dla target detection
TARGET_NAME_HINTS = (
    "target", "y", "label", "sales", "sprzeda", "zuzy", "consum",
    "amount", "profit", "revenue", "price", "cena", "wartość", "value"
)

ProblemType = Literal["classification", "regression", "timeseries"]


# ========================================================================================
# DATACLASSES
# ========================================================================================

@dataclass
class TrainingConfig:
    """Konfiguracja treningu modelu."""
    target: str
    test_size: float
    random_state: int
    enable_shap: bool
    shap_max_samples: int


@dataclass
class ModelMetrics:
    """Metryki modelu."""
    problem_type: ProblemType
    train_samples: int
    test_samples: int
    n_features: int
    training_time: float
    metrics: dict
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def to_dict(self) -> dict:
        """Konwertuje do słownika."""
        return asdict(self)


@dataclass
class TrainingResult:
    """Wynik treningu."""
    model: Any
    metrics: ModelMetrics
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    feature_names: list[str]


# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

def _validate_dataframe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Waliduje DataFrame z session state.
    
    Args:
        df: DataFrame do walidacji
        
    Returns:
        Zwalidowany DataFrame
        
    Raises:
        ValueError: Jeśli DataFrame jest nieprawidłowy
    """
    if df is None:
        raise ValueError("Brak danych w session state")
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Oczekiwano DataFrame, otrzymano {type(df)}")
    
    if df.empty:
        raise ValueError("DataFrame jest pusty")
    
    return df


def _validate_training_data(
    df: pd.DataFrame,
    target: str
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Waliduje dane treningowe.
    
    Args:
        df: DataFrame z danymi
        target: Nazwa kolumny celu
        
    Returns:
        Tuple (X, y) - features i target
        
    Raises:
        ValueError: Jeśli dane są nieprawidłowe
    """
    # Sprawdź czy target istnieje
    if target not in df.columns:
        raise ValueError(f"Kolumna celu '{target}' nie istnieje w danych")
    
    # Sprawdź rozmiar
    if len(df) < MIN_SAMPLES:
        raise ValueError(
            f"Za mało próbek do treningu ({len(df)}). "
            f"Minimum: {MIN_SAMPLES}"
        )
    
    if len(df) > MAX_TRAIN_ROWS:
        raise ValueError(
            f"Za dużo próbek ({len(df):,}). "
            f"Maksimum: {MAX_TRAIN_ROWS:,}"
        )
    
    # Przygotuj X i y
    y = df[target]
    X = df.drop(columns=[target])
    
    # Walidacja X
    if X.shape[1] == 0:
        raise ValueError("Brak kolumn cech (features)")
    
    if X.shape[1] > MAX_FEATURES:
        raise ValueError(
            f"Za dużo cech ({X.shape[1]}). "
            f"Maksimum: {MAX_FEATURES}"
        )
    
    # Walidacja y
    if y.isna().all():
        raise ValueError("Wszystkie wartości celu są NaN")
    
    if y.isna().any():
        logger.warning(f"Target ma {y.isna().sum()} wartości NaN - zostaną usunięte")
        # Usuń NaN z y i odpowiednie wiersze z X
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
    
    return X, y


def _detect_target_candidates(df: pd.DataFrame) -> list[str]:
    """
    Znajduje potencjalne kolumny docelowe.
    
    Args:
        df: DataFrame do przeszukania
        
    Returns:
        Lista nazw kolumn - kandydatów
    """
    candidates = []
    
    for col in df.columns:
        col_lower = col.lower()
        if any(hint in col_lower for hint in TARGET_NAME_HINTS):
            candidates.append(col)
    
    return list(dict.fromkeys(candidates))  # Usuń duplikaty


def _safe_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    problem_type: Optional[str] = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Bezpieczny split z walidacją stratyfikacji.
    
    Args:
        X: Features
        y: Target
        test_size: Rozmiar zbioru testowego
        random_state: Seed dla reproducibility
        problem_type: Typ problemu (dla stratyfikacji)
        
    Returns:
        Tuple (X_train, X_test, y_train, y_test)
    """
    # Stratyfikacja dla klasyfikacji
    stratify = None
    if problem_type == "classification":
        # Sprawdź czy stratyfikacja możliwa
        try:
            class_counts = y.value_counts()
            min_class_count = class_counts.min()
            
            # Musi być minimum 2 próbki w najmniejszej klasie
            if min_class_count >= 2:
                stratify = y
            else:
                logger.warning(
                    f"Stratyfikacja niemożliwa - najmniejsza klasa ma "
                    f"{min_class_count} próbek"
                )
        except Exception as e:
            logger.warning(f"Błąd sprawdzania stratyfikacji: {e}")
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Błąd split: {e}", exc_info=True)
        
        # Retry bez stratyfikacji
        logger.info("Retry bez stratyfikacji...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=None
        )
        return X_train, X_test, y_train, y_test


def _compute_classification_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model: Any
) -> dict:
    """
    Oblicza metryki dla klasyfikacji.
    
    Args:
        y_test: Prawdziwe wartości
        y_pred: Predykcje
        model: Wytrenowany model
        
    Returns:
        Słownik z metrykami
    """
    metrics = {}
    
    # Classification report
    try:
        report = classification_report(
            y_test, y_pred,
            output_dict=True,
            zero_division=0
        )
        metrics["classification_report"] = report
        
        # Extract key metrics
        if "accuracy" in report:
            metrics["accuracy"] = report["accuracy"]
        
        if "weighted avg" in report:
            metrics["f1_weighted"] = report["weighted avg"]["f1-score"]
            metrics["precision_weighted"] = report["weighted avg"]["precision"]
            metrics["recall_weighted"] = report["weighted avg"]["recall"]
            
    except Exception as e:
        logger.error(f"Błąd classification report: {e}")
        metrics["classification_report_error"] = str(e)
    
    # Confusion matrix
    try:
        unique_labels = sorted(np.unique(y_test))
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["labels"] = unique_labels.tolist()
    except Exception as e:
        logger.error(f"Błąd confusion matrix: {e}")
    
    # ROC AUC (jeśli binary)
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(y_test.index)  # Use indices if available
            
            if proba.shape[1] == 2:  # Binary classification
                auc = roc_auc_score(y_test, proba[:, 1])
                metrics["roc_auc"] = float(auc)
                
        except Exception as e:
            logger.warning(f"ROC AUC pominięty: {e}")
    
    return metrics


def _compute_regression_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """
    Oblicza metryki dla regresji.
    
    Args:
        y_test: Prawdziwe wartości
        y_pred: Predykcje
        
    Returns:
        Słownik z metrykami
    """
    metrics = {}
    
    try:
        metrics["rmse"] = float(mean_squared_error(y_test, y_pred, squared=False))
        metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
        metrics["r2"] = float(r2_score(y_test, y_pred))
        
        # MSE
        metrics["mse"] = float(mean_squared_error(y_test, y_pred, squared=True))
        
        # MAPE (jeśli nie ma zer)
        if not np.any(y_test == 0):
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            metrics["mape"] = float(mape)
            
    except Exception as e:
        logger.error(f"Błąd metryk regresji: {e}", exc_info=True)
        metrics["error"] = str(e)
    
    return metrics


def _extract_feature_importance(
    model: Any,
    feature_names: list[str],
    top_k: int = 30
) -> Optional[pd.Series]:
    """
    Ekstrahuje feature importance z modelu.
    
    Args:
        model: Wytrenowany model
        feature_names: Nazwy cech
        top_k: Liczba top cech do zwrócenia
        
    Returns:
        Series z importance lub None
    """
    try:
        if hasattr(model, "feature_importances_"):
            importance = pd.Series(
                model.feature_importances_,
                index=feature_names
            ).sort_values(ascending=False).head(top_k)
            
            return importance
            
    except Exception as e:
        logger.warning(f"Nie udało się wyekstrahować feature importance: {e}")
    
    return None


def _compute_shap_values(
    model: Any,
    X_test: pd.DataFrame,
    max_samples: int,
    random_state: int
) -> Optional[tuple]:
    """
    Oblicza SHAP values z bezpiecznymi limitami.
    
    Args:
        model: Wytrenowany model
        X_test: Dane testowe
        max_samples: Maksymalna liczba próbek
        random_state: Seed
        
    Returns:
        Tuple (explainer, shap_values, X_sample) lub None
    """
    try:
        import shap
        
        # Próbkowanie
        n_samples = min(len(X_test), max_samples)
        X_sample = X_test.sample(n=n_samples, random_state=random_state)
        
        logger.info(f"Obliczam SHAP dla {n_samples} próbek...")
        
        # Wybór explainera na podstawie typu modelu
        model_name = model.__class__.__name__.lower()
        
        if any(tree_type in model_name for tree_type in [
            "lgbm", "xgb", "randomforest", "gradientboosting",
            "extratrees", "catboost"
        ]):
            # Tree explainer - szybki
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
        else:
            # Kernel explainer - wolniejszy, użyj mniejszej próbki
            kernel_samples = min(n_samples, 500)
            X_kernel = X_sample.sample(n=kernel_samples, random_state=random_state)
            
            logger.warning(
                f"Model nieobsługiwany przez TreeExplainer. "
                f"Używam KernelExplainer z {kernel_samples} próbkami (może być wolny)"
            )
            
            explainer = shap.KernelExplainer(model.predict, X_kernel)
            shap_values = explainer.shap_values(X_kernel)
            X_sample = X_kernel
        
        return explainer, shap_values, X_sample
        
    except ImportError:
        logger.error("SHAP library nie jest zainstalowana")
        return None
    except Exception as e:
        logger.error(f"Błąd obliczania SHAP: {e}", exc_info=True)
        return None


def _add_to_training_history(result: TrainingResult, config: TrainingConfig) -> None:
    """
    Dodaje wynik treningu do historii.
    
    Args:
        result: Wynik treningu
        config: Konfiguracja treningu
    """
    if "training_history" not in st.session_state:
        st.session_state["training_history"] = []
    
    # Metadane historii
    history_entry = {
        "timestamp": result.metrics.timestamp,
        "target": config.target,
        "problem_type": result.metrics.problem_type,
        "train_samples": result.metrics.train_samples,
        "test_samples": result.metrics.test_samples,
        "metrics": result.metrics.metrics,
        "test_size": config.test_size,
        "training_time": result.metrics.training_time
    }
    
    # Dodaj na początek
    st.session_state["training_history"].insert(0, history_entry)
    
    # Ogranicz do 10
    st.session_state["training_history"] = st.session_state["training_history"][:10]


# ========================================================================================
# STREAMLIT UI
# ========================================================================================

st.title("📈 Predictions (AutoML) — PRO")

# ========================================================================================
# WALIDACJA DANYCH
# ========================================================================================

try:
    df_raw = st.session_state.get("df")
    df_main = _validate_dataframe(df_raw)
except ValueError as e:
    st.warning(f"⚠️ {e}")
    st.info(
        "Przejdź do **📤 Upload Data** i użyj "
        "**🧹 Szybkie czyszczenie + FE**"
    )
    st.stop()

# ========================================================================================
# SIDEBAR: KONFIGURACJA
# ========================================================================================

with st.sidebar:
    st.subheader("⚙️ Ustawienia treningu")
    
    # Target selection
    target_candidates = _detect_target_candidates(df_main)
    
    all_columns = list(df_main.columns)
    
    # Domyślny index
    if target_candidates:
        default_idx = all_columns.index(target_candidates[0])
    else:
        default_idx = 0
    
    target = st.selectbox(
        "Kolumna celu",
        options=all_columns,
        index=default_idx,
        help="Wybierz zmienną, którą model ma przewidywać"
    )
    
    st.caption(
        f"💡 Wykryto {len(target_candidates)} kandydatów na cel"
        if target_candidates else
        "💡 Nie wykryto oczywistych kandydatów"
    )
    
    st.divider()
    
    # Parametry splitu
    test_size = st.slider(
        "Wielkość testu",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05,
        help="Procent danych do walidacji"
    )
    
    random_state = st.number_input(
        "Random state",
        min_value=0,
        value=42,
        step=1,
        help="Seed dla reproducibility"
    )
    
    st.divider()
    
    # SHAP
    enable_shap = st.checkbox(
        "🧠 Oblicz SHAP",
        value=True,
        help="Wyjaśnialność modelu (może zająć chwilę)"
    )
    
    if enable_shap:
        shap_max_n = st.number_input(
            "Maks. próbek SHAP",
            min_value=200,
            max_value=MAX_SHAP_SAMPLES,
            value=DEFAULT_SHAP_SAMPLES,
            step=100,
            help=f"Maksimum: {MAX_SHAP_SAMPLES:,}"
        )
    else:
        shap_max_n = DEFAULT_SHAP_SAMPLES
    
    st.divider()
    
    # Historia
    history_count = len(st.session_state.get("training_history", []))
    st.caption(f"📚 Historia: {history_count} treningów")

# Konfiguracja
config = TrainingConfig(
    target=target,
    test_size=test_size,
    random_state=random_state,
    enable_shap=enable_shap,
    shap_max_samples=shap_max_n
)

# Zapisz target do session
st.session_state["target"] = target

# ========================================================================================
# WYKRYWANIE TYPU PROBLEMU
# ========================================================================================

st.subheader("🎯 Detekcja typu problemu")

try:
    problem_type_hint = infer_problem_type(df_main, target)
    
    if problem_type_hint:
        # Info o typie
        type_emoji = {
            "classification": "🏷️",
            "regression": "📊",
            "timeseries": "📈"
        }
        
        emoji = type_emoji.get(problem_type_hint, "❓")
        
        st.info(
            f"{emoji} Wykryto typ: **{problem_type_hint.upper()}**\n\n"
            f"Target: `{target}` • "
            f"Unikalne: {df_main[target].nunique()} • "
            f"Typ: {df_main[target].dtype}"
        )
    else:
        st.warning("⚠️ Nie udało się określić typu problemu automatycznie")
        
except Exception as e:
    st.error(f"❌ Błąd wykrywania typu: {e}")
    logger.error(f"Błąd detekcji typu: {e}", exc_info=True)

# ========================================================================================
# PRZYCISK TRENINGU
# ========================================================================================

st.divider()

train_col1, train_col2 = st.columns([3, 1])

with train_col1:
    train_button = st.button(
        "🚀 Trenuj AutoML",
        type="primary",
        use_container_width=True,
        help="Rozpocznij trening modelu"
    )

with train_col2:
    if st.button("🗑️ Wyczyść historię", use_container_width=True):
        st.session_state["training_history"] = []
        st.success("✅ Historia wyczyszczona")
        st.rerun()

# ========================================================================================
# GŁÓWNY PIPELINE TRENINGU
# ========================================================================================

if train_button:
    start_time = time.time()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # ============================================================================
        # ETAP 1: WALIDACJA DANYCH (0-20%)
        # ============================================================================
        
        status_text.text("🔍 Walidacja danych...")
        progress_bar.progress(10)
        
        X, y = _validate_training_data(df_main, target)
        
        st.success(
            f"✅ Dane zwalidowane: {len(X):,} próbek × {X.shape[1]} cech"
        )
        
        progress_bar.progress(20)
        
        # ============================================================================
        # ETAP 2: SPLIT DANYCH (20-30%)
        # ============================================================================
        
        status_text.text("✂️ Podział train/test...")
        progress_bar.progress(25)
        
        X_train, X_test, y_train, y_test = _safe_train_test_split(
            X, y,
            config.test_size,
            config.random_state,
            problem_type_hint
        )
        
        st.info(
            f"📊 Split: Train={len(X_train):,} • Test={len(X_test):,} "
            f"({config.test_size*100:.0f}%)"
        )
        
        progress_bar.progress(30)
        
        # ============================================================================
        # ETAP 3: TRENING MODELU (30-70%)
        # ============================================================================
        
        status_text.text("🤖 Trening modelu AutoML...")
        progress_bar.progress(40)
        
        # Przygotuj DataFrame dla train_automl
        df_train = X_train.copy()
        df_train[target] = y_train.values
        
        model, automl_metrics, detected_ptype = train_automl(df_train, target)
        
        training_time = time.time() - start_time
        
        st.success(
            f"✅ Model wytrenowany! "
            f"Typ: **{detected_ptype.upper()}** • "
            f"Czas: **{training_time:.2f}s**"
        )
        
        # Zapisz do session
        st.session_state["model"] = model
        st.session_state["problem_type"] = detected_ptype
        
        progress_bar.progress(70)
        
        # ============================================================================
        # ETAP 4: PREDYKCJE (70-80%)
        # ============================================================================
        
        status_text.text("🔮 Generuję predykcje...")
        progress_bar.progress(75)
        
        y_pred = model.predict(X_test)
        
        progress_bar.progress(80)
        
        # ============================================================================
        # ETAP 5: METRYKI (80-90%)
        # ============================================================================
        
        status_text.text("📊 Obliczam metryki...")
        progress_bar.progress(85)
        
        # Oblicz metryki specyficzne dla typu
        if detected_ptype == "classification":
            eval_metrics = _compute_classification_metrics(y_test, y_pred, model)
        elif detected_ptype == "regression":
            eval_metrics = _compute_regression_metrics(y_test, y_pred)
        else:
            eval_metrics = {"warning": "Nieznany typ problemu"}
        
        # Połącz z metrykami AutoML
        eval_metrics.update(automl_metrics)
        
        # Stwórz obiekt metryk
        model_metrics = ModelMetrics(
            problem_type=detected_ptype,
            train_samples=len(X_train),
            test_samples=len(X_test),
            n_features=X.shape[1],
            training_time=training_time,
            metrics=eval_metrics
        )
        
        progress_bar.progress(90)
        
        # ============================================================================
        # ETAP 6: FINALIZACJA (90-100%)
        # ============================================================================
        
        status_text.text("💾 Zapisuję wyniki...")
        
        # Wynik treningu
        result = TrainingResult(
            model=model,
            metrics=model_metrics,
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            feature_names=list(X.columns)
        )
        
        # Dodaj do historii
        _add_to_training_history(result, config)
        
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        st.success(f"🎉 Trening zakończony pomyślnie w {training_time:.2f}s!")
        
        # ============================================================================
        # PREZENTACJA WYNIKÓW
        # ============================================================================
        
        st.divider()
        
        # Tab layout
        tabs = st.tabs([
            "📊 Metryki",
            "📈 Wizualizacje",
            "🏷️ Feature Importance",
            "🧠 SHAP",
            "💾 Export"
        ])
        
        # ========================================================================
        # TAB 1: METRYKI
        # ========================================================================
        
        with tabs[0]:
            st.subheader("📋 Kluczowe metryki")
            
            if detected_ptype == "classification":
                # Metryki classification
                if "accuracy" in eval_metrics:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{eval_metrics['accuracy']:.4f}")
                    
                    if "f1_weighted" in eval_metrics:
                        col2.metric("F1 (weighted)", f"{eval_metrics['f1_weighted']:.4f}")
                    if "precision_weighted" in eval_metrics:
                        col3.metric("Precision", f"{eval_metrics['precision_weighted']:.4f}")
                    if "recall_weighted" in eval_metrics:
                        col4.metric("Recall", f"{eval_metrics['recall_weighted']:.4f}")
                
                # ROC AUC jeśli jest
                if "roc_auc" in eval_metrics:
                    st.metric("ROC AUC", f"{eval_metrics['roc_auc']:.4f}")
                
                # Pełny raport
                with st.expander("📄 Pełny raport klasyfikacji", expanded=False):
                    if "classification_report" in eval_metrics:
                        st.json(eval_metrics["classification_report"])
                
            elif detected_ptype == "regression":
                # Metryki regression
                col1, col2, col3, col4 = st.columns(4)
                
                if "rmse" in eval_metrics:
                    col1.metric("RMSE", f"{eval_metrics['rmse']:.4f}")
                if "mae" in eval_metrics:
                    col2.metric("MAE", f"{eval_metrics['mae']:.4f}")
                if "r2" in eval_metrics:
                    col3.metric("R²", f"{eval_metrics['r2']:.4f}")
                if "mape" in eval_metrics:
                    col4.metric("MAPE", f"{eval_metrics['mape']:.2f}%")
            
            # AutoML metrics
            with st.expander("🤖 AutoML Metrics", expanded=False):
                st.json(automl_metrics)
        
        # ========================================================================
        # TAB 2: WIZUALIZACJE
        # ========================================================================
        
        with tabs[1]:
            st.subheader("📈 Wizualizacje predykcji")
            
            if detected_ptype == "classification":
                # Confusion Matrix
                if "confusion_matrix" in eval_metrics and "labels" in eval_metrics:
                    cm = np.array(eval_metrics["confusion_matrix"])
                    labels = eval_metrics["labels"]
                    
                    fig_cm = px.imshow(
                        cm,
                        x=labels,
                        y=labels,
                        text_auto=True,
                        aspect="auto",
                        title="Confusion Matrix (Test Set)",
                        labels=dict(x="Predicted", y="True")
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                
                # ROC Curve (jeśli binary)
                if "roc_auc" in eval_metrics and hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(X_test)
                        if proba.shape[1] == 2:
                            fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
                            
                            fig_roc = go.Figure()
                            fig_roc.add_trace(go.Scatter(
                                x=fpr, y=tpr,
                                mode="lines",
                                name=f"ROC (AUC={eval_metrics['roc_auc']:.4f})"
                            ))
                            fig_roc.add_trace(go.Scatter(
                                x=[0, 1], y=[0, 1],
                                mode="lines",
                                line=dict(dash="dash", color="gray"),
                                name="Random"
                            ))
                            fig_roc.update_layout(
                                title="ROC Curve",
                                xaxis_title="False Positive Rate",
                                yaxis_title="True Positive Rate"
                            )
                            st.plotly_chart(fig_roc, use_container_width=True)
                    except Exception as e:
                        st.info(f"ROC curve pominięty: {e}")
            
            elif detected_ptype == "regression":
                # Parity Plot
                df_parity = pd.DataFrame({
                    "y_true": y_test.values,
                    "y_pred": y_pred
                })
                
                fig_parity = px.scatter(
                    df_parity,
                    x="y_true",
                    y="y_pred",
                    title="Parity Plot (True vs Predicted)",
                    trendline="ols",
                    opacity=0.6
                )
                fig_parity.add_trace(go.Scatter(
                    x=[y_test.min(), y_test.max()],
                    y=[y_test.min(), y_test.max()],
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    name="Perfect"
                ))
                st.plotly_chart(fig_parity, use_container_width=True)
                
                # Residuals
                residuals = y_test.values - y_pred
                fig_resid = px.histogram(
                    x=residuals,
                    nbins=40,
                    title="Rozkład reszt (Residuals)",
                    labels={"x": "Residual"}
                )
                st.plotly_chart(fig_resid, use_container_width=True)
                
                # Residuals vs Predicted
                fig_resid_scatter = px.scatter(
                    x=y_pred,
                    y=residuals,
                    title="Residuals vs Predicted",
                    labels={"x": "Predicted", "y": "Residual"},
                    opacity=0.6
                )
                fig_resid_scatter.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_resid_scatter, use_container_width=True)
        
        # ========================================================================
        # TAB 3: FEATURE IMPORTANCE
        # ========================================================================
        
        with tabs[2]:
            st.subheader("🏷️ Ważność cech")
            
            importance = _extract_feature_importance(model, result.feature_names)
            
            if importance is not None:
                # Bar chart
                st.bar_chart(importance)
                
                # Table
                with st.expander("📋 Tabela ważności", expanded=False):
                    importance_df = importance.to_frame(name="importance")
                    st.dataframe(importance_df, use_container_width=True)
            else:
                st.info(
                    "ℹ️ Model nie udostępnia feature importance.\n\n"
                    "Niektóre modele (np. linear) nie mają atrybutu "
                    "`feature_importances_`. Sprawdź SHAP dla wyjaśnialności."
                )
        
        # ========================================================================
        # TAB 4: SHAP
        # ========================================================================
        
        with tabs[3]:
            st.subheader("🧠 SHAP Analysis")
            
            if config.enable_shap:
                shap_button = st.button(
                    "🔮 Oblicz SHAP values",
                    use_container_width=True,
                    help="Może zająć kilka minut dla dużych danych"
                )
                
                if shap_button:
                    with st.spinner(f"Obliczam SHAP dla {config.shap_max_samples} próbek..."):
                        shap_result = _compute_shap_values(
                            model,
                            X_test,
                            config.shap_max_samples,
                            config.random_state
                        )
                    
                    if shap_result is not None:
                        explainer, shap_values, X_sample = shap_result
                        
                        st.success(f"✅ SHAP obliczony dla {len(X_sample)} próbek")
                        
                        # Generuj wykres
                        try:
                            import matplotlib.pyplot as plt
                            
                            st.caption(
                                "📊 SHAP Summary Plot (beeswarm) - "
                                "pokazuje wpływ cech na predykcje"
                            )
                            
                            # Plot
                            fig, ax = plt.subplots(figsize=(10, 8))
                            
                            import shap
                            shap.summary_plot(
                                shap_values,
                                X_sample,
                                max_display=20,
                                show=False
                            )
                            
                            plt.tight_layout()
                            
                            # Wyświetl w Streamlit
                            st.pyplot(fig, use_container_width=True)
                            plt.close()
                            
                        except Exception as e:
                            st.error(f"Błąd generowania wykresu SHAP: {e}")
                            logger.error(f"SHAP plot error: {e}", exc_info=True)
                    else:
                        st.error("❌ Nie udało się obliczyć SHAP values")
                else:
                    st.info("👆 Kliknij przycisk aby obliczyć SHAP values")
            else:
                st.info(
                    "ℹ️ SHAP wyłączony w ustawieniach.\n\n"
                    "Włącz w sidebar, aby uzyskać wyjaśnialność modelu."
                )
        
        # ========================================================================
        # TAB 5: EXPORT
        # ========================================================================
        
        with tabs[4]:
            st.subheader("💾 Eksport artefaktów")
            
            col1, col2, col3 = st.columns(3)
            
            # Model export
            with col1:
                try:
                    model_bytes = io.BytesIO()
                    joblib.dump(model, model_bytes)
                    model_bytes.seek(0)
                    
                    st.download_button(
                        "⬇️ Model (.joblib)",
                        data=model_bytes,
                        file_name=f"model_{detected_ptype}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Błąd exportu modelu: {e}")
            
            # Metrics export
            with col2:
                try:
                    metrics_json = json.dumps(
                        model_metrics.to_dict(),
                        ensure_ascii=False,
                        indent=2
                    )
                    
                    st.download_button(
                        "⬇️ Metryki (.json)",
                        data=metrics_json,
                        file_name=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Błąd exportu metryk: {e}")
            
            # Predictions export
            with col3:
                try:
                    preds_df = pd.DataFrame({
                        "y_true": y_test.values,
                        "y_pred": y_pred if y_pred.ndim == 1 else np.argmax(y_pred, axis=1)
                    })
                    
                    preds_csv = preds_df.to_csv(index=False)
                    
                    st.download_button(
                        "⬇️ Predykcje (.csv)",
                        data=preds_csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Błąd exportu predykcji: {e}")
            
            st.divider()
            
            # Feature names export
            with st.expander("📋 Feature names (TXT)", expanded=False):
                features_txt = "\n".join(result.feature_names)
                st.download_button(
                    "⬇️ Pobierz listę cech",
                    data=features_txt,
                    file_name="features.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
    except ValueError as ve:
        progress_bar.empty()
        status_text.empty()
        
        st.error(f"❌ Błąd walidacji: {ve}")
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        
        st.error(f"❌ Błąd treningu: {e}")
        logger.error(f"Training error: {e}", exc_info=True)
        
        st.info(
            "💡 **Możliwe rozwiązania:**\n"
            "- Sprawdź czy dane nie mają NaN w targecie\n"
            "- Zwiększ liczbę próbek\n"
            "- Sprawdź czy target ma prawidłowy typ\n"
            "- Spróbuj innego random_state"
        )

else:
    st.info(
        "👆 Ustaw parametry w sidebar i kliknij **Trenuj AutoML**\n\n"
        f"Target: **{target}** • "
        f"Test size: **{config.test_size*100:.0f}%** • "
        f"SHAP: **{'✓' if config.enable_shap else '✗'}**"
    )

# ========================================================================================
# HISTORIA TRENINGU
# ========================================================================================

history = st.session_state.get("training_history", [])

if history:
    st.divider()
    st.subheader("📚 Historia treningów")
    
    for idx, hist_entry in enumerate(history):
        timestamp = hist_entry.get("timestamp", "Unknown")
        target_name = hist_entry.get("target", "N/A")
        ptype = hist_entry.get("problem_type", "N/A")
        train_time = hist_entry.get("training_time", 0)
        
        with st.expander(f"🕒 {timestamp} | {target_name} ({ptype})", expanded=(idx == 0)):
            col1, col2, col3 = st.columns(3)
            col1.metric("Train samples", f"{hist_entry.get('train_samples', 0):,}")
            col2.metric("Test samples", f"{hist_entry.get('test_samples', 0):,}")
            col3.metric("Training time", f"{train_time:.2f}s")
            
            # Metryki
            metrics = hist_entry.get("metrics", {})
            if metrics:
                with st.expander("📊 Metryki", expanded=False):
                    st.json(metrics)

# ========================================================================================
# WSKAZÓWKI NAWIGACJI
# ========================================================================================

st.divider()
st.success(
    "✨ **Co dalej?**\n\n"
    "- **📊 EDA Analysis** — przeanalizuj dane przed treningiem\n"
    "- **🤖 AI Insights** — uzyskaj wnioski z AI\n"
    "- **📈 Forecasting** — prognozy szeregów czasowych\n"
    "- **📄 Reports** — wygeneruj raport z treningu"
)