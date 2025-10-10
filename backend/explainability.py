from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import shap

# opcjonalnie przyspieszenie/bezpieczeństwo pamięci
try:
    import scipy.sparse as sp
except Exception:
    sp = None  # type: ignore

@dataclass
class ShapMeta:
    model_name: str
    problem_type: str                      # "classification" | "regression" | "unknown"
    is_multiclass: bool
    class_names: Optional[List[str]]
    feature_names: List[str]
    n_samples: int
    n_features: int
    background_size: int
    explained_estimator: Optional[str]     # gdy Voting*, który składnik wyjaśniamy
    expected_value: Union[float, List[float], None]

def _is_tree_model(model: Any) -> bool:
    n = model.__class__.__name__.lower()
    m = model.__class__.__module__.lower()
    return any(k in n or k in m for k in ["xgb", "xgboost", "lgbm", "lightgbm", "catboost", "randomforest", "extratrees", "decisiontree", "gradientboost"])

def _pick_estimator_for_voting(model: Any) -> Tuple[Any, Optional[str]]:
    """Jeśli Voting* – wybierz pierwszy estimator o typie drzewiastym (SHAP-friendly)."""
    if hasattr(model, "estimators"):
        for name, est in model.estimators:
            if _is_tree_model(est):
                return est, name
        # brak drzew – weź pierwszy
        if model.estimators:
            return model.estimators[0][1], model.estimators[0][0]
    return model, None

def _feature_names_from_pre(pre, X: pd.DataFrame) -> List[str]:
    if pre is None:
        return list(getattr(X, "columns", [f"f{i}" for i in range(X.shape[1])]))
    try:
        return list(pre.get_feature_names_out())
    except Exception:
        # ColumnTransformer w starszych sklearn
        return [f"f{i}" for i in range(pre.transform(X[:1]).shape[1])]

def _to_dense_if_small(Xt, max_cells: int = 2_000_000):
    """Konwersja sparse->dense tylko jeśli rozmiar sensowny (chroni RAM)."""
    if sp is not None and sp.issparse(Xt):
        cells = int(Xt.shape[0]) * int(Xt.shape[1])
        if cells <= max_cells:
            return Xt.toarray()
        # zostaw sparsa – TreeExplainer zwykle i tak akceptuje CSR
        return Xt.tocsr()
    return Xt

def _infer_problem(model: Any) -> str:
    if hasattr(model, "predict_proba") or hasattr(model, "classes_"):
        return "classification"
    if hasattr(model, "predict"):
        return "regression"
    return "unknown"

def estimate_shap_values(
    pipeline,
    X_sample: pd.DataFrame,
    *,
    max_samples: int = 500,
    max_background: int = 200,
    return_meta: bool = False,
):
    """
    Oblicza wartości SHAP dla modelu w pipeline:
      - pipeline: sklearn/imblearn Pipeline z krokami 'pre' i 'model' (nazwa kroków nie jest wymagana)
      - X_sample: surowe cechy (przed preprocesingiem)
      - max_samples: ogranicza liczbę próbek do obliczeń (szybkość)
      - max_background: tło do Explainer (dla shap.Explainer / Kernel/Linear)
    Zwraca:
      - domyślnie: dokładnie to, co zwraca SHAP (np. ndarray albo lista ndarray),
      - jeśli return_meta=True: (shap_values, ShapMeta).
    """
    # --- kroki pipeline ---
    steps = getattr(pipeline, "named_steps", {})
    pre = steps.get("pre", None)
    model = steps.get("model", pipeline)

    # Voting* – wybierz estimator do wyjaśnień
    model_for_explain, explained_name = _pick_estimator_for_voting(model)

    # Transform i sampling
    if pre is not None:
        Xt = pre.transform(X_sample)
    else:
        Xt = X_sample.values if isinstance(X_sample, pd.DataFrame) else X_sample

    # sampling
    if isinstance(Xt, (np.ndarray,)) and Xt.shape[0] > max_samples:
        idx = np.random.RandomState(42).choice(Xt.shape[0], size=max_samples, replace=False)
        Xt_eval = Xt[idx]
        X_eval_raw = X_sample.iloc[idx] if isinstance(X_sample, pd.DataFrame) else None
    else:
        Xt_eval = Xt
        X_eval_raw = X_sample

    if sp is not None and sp.issparse(Xt_eval):
        Xt_eval = Xt_eval.tocsr()

    # background (masker)
    if isinstance(Xt, (np.ndarray,)) and Xt.shape[0] > max_background:
        b_idx = np.random.RandomState(42).choice(Xt.shape[0], size=max_background, replace=False)
        background = Xt[b_idx]
    else:
        background = Xt

    # nazwy cech
    feature_names = _feature_names_from_pre(pre, X_sample)

    # problem/klasy
    problem = _infer_problem(model_for_explain)
    is_multiclass = bool(getattr(model_for_explain, "classes_", [])) and len(getattr(model_for_explain, "classes_", [])) > 2
    class_names = list(getattr(model_for_explain, "classes_", [])) if hasattr(model_for_explain, "classes_") else None

    # --- wybór Explainer-a ---
    shap_values = None
    expected = None

    try:
        if _is_tree_model(model_for_explain):
            # Szybko i stabilnie dla drzew
            explainer = shap.TreeExplainer(model_for_explain, feature_perturbation="interventional")
            Xt_eval_dense = _to_dense_if_small(Xt_eval)
            shap_values = explainer.shap_values(Xt_eval_dense)
            expected = explainer.expected_value
        else:
            # Model-agnostic fallback
            masker = shap.maskers.Independent(background)
            explainer = shap.Explainer(model_for_explain, masker=masker)
            shap_values = explainer(Xt_eval)  # może zwrócić obiekt Explanation
            expected = getattr(shap_values, "base_values", getattr(explainer, "expected_value", None))
            # Wyciągnij ndarray z Explanation, gdy trzeba
            if hasattr(shap_values, "values"):
                shap_values = shap_values.values
    except Exception as e:
        # ostatnia deska – prosta próba z LinearExplainer albo KernelExplainer
        try:
            if problem == "regression":
                lexp = shap.LinearExplainer(model_for_explain, background)
                sv = lexp.shap_values(Xt_eval)
                shap_values = sv
                expected = lexp.expected_value
            else:
                # KernelExplainer – wolny; używaj z małym max_samples
                kexp = shap.KernelExplainer(lambda x: model_for_explain.predict_proba(x), background)
                sv = kexp.shap_values(Xt_eval)
                shap_values = sv
                expected = getattr(kexp, "expected_value", None)
        except Exception:
            # nic nie udało się – kompatybilne z Twoim API:
            return (None, None) if return_meta else None

    # --- META ---
    meta = ShapMeta(
        model_name=model_for_explain.__class__.__name__,
        problem_type=problem,
        is_multiclass=is_multiclass,
        class_names=class_names,
        feature_names=feature_names,
        n_samples=int(Xt_eval.shape[0]),
        n_features=int(Xt_eval.shape[1]),
        background_size=int(background.shape[0]) if hasattr(background, "shape") else max_background,
        explained_estimator=explained_name,
        expected_value=expected,
    )

    return (shap_values, meta) if return_meta else shap_values
