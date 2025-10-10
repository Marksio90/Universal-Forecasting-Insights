from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import os, json
import numpy as np
import pandas as pd

# Twój limiter (zostaje)
from ..utils.rate_limiter import rate_limited
from .prompt_templates import SYSTEM_INSIGHTS

# === NAZWA_SEKCJI === KONFIG (ENV) ===
_MAX_ROWS = int(os.getenv("INSIGHTS_MAX_ROWS", "100000"))     # twardy limit wierszy do obliczeń
_MAX_COLS = int(os.getenv("INSIGHTS_MAX_COLS", "300"))        # limit liczby kolumn do analiz
_TOP_SIGNALS = int(os.getenv("INSIGHTS_TOP_SIGNALS", "12"))   # ile cech wypisać jako top sygnały
_HI_CARD_RATIO = float(os.getenv("HI_CARD_RATIO", "0.5"))     # >50% unikalności => wysokokard.
_CONST_TOL = float(os.getenv("CONST_TOL", "1e-12"))           # tolerancja stałych kolumn
_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_OPENAI_TEMP = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
_OPENAI_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "500"))

def _get_openai_key() -> Optional[str]:
    # st.secrets-like pattern bez importu streamlit
    return os.getenv("OPENAI_API_KEY") or None

# === NAZWA_SEKCJI === HELPERY ===
def _infer_target(df: pd.DataFrame, target: Optional[str]) -> Optional[str]:
    if target and target in df.columns:
        return target
    # heurystyka: ostatnia kolumna, jeśli nie wygląda na ID (dużo unikalnych) / timestamp
    last = df.columns[-1] if len(df.columns) else None
    if not last:
        return None
    s = df[last]
    if pd.api.types.is_datetime64_any_dtype(s):
        return None
    # Jeśli wygląda na identyfikator (bardzo wiele unikalnych) — odrzuć
    if s.nunique(dropna=False) > 0.95 * len(s):
        return None
    return last

def _infer_problem_type(y: pd.Series) -> str:
    if y is None or y.empty:
        return "unknown"
    if pd.api.types.is_float_dtype(y):
        return "regression"
    nun = y.nunique(dropna=True)
    return "classification" if nun <= 50 else "regression"

def _basic_schema(df: pd.DataFrame) -> Dict[str, Any]:
    dtypes = {c: str(df[c].dtype) for c in df.columns[:_MAX_COLS]}
    return {"rows": int(df.shape[0]), "cols": int(df.shape[1]), "dtypes": dtypes}

def _missing_stats(df: pd.DataFrame) -> Dict[str, float]:
    na = (df.isna().sum() / len(df)).to_dict()
    # Zredukuj do sensownego rozmiaru
    if len(na) > _MAX_COLS:
        na = {k: na[k] for k in list(df.columns)[:_MAX_COLS]}
    return {k: float(v) for k, v in na.items()}

def _constant_cols(df: pd.DataFrame) -> List[str]:
    out = []
    for c in df.columns[:_MAX_COLS]:
        s = df[c].dropna()
        if s.empty:
            continue
        if pd.api.types.is_numeric_dtype(s):
            if float(s.max() - s.min()) <= _CONST_TOL:
                out.append(c)
        else:
            if s.nunique(dropna=True) <= 1:
                out.append(c)
    return out

def _high_card_cols(df: pd.DataFrame) -> List[str]:
    out = []
    n = len(df)
    if n == 0:
        return out
    for c in df.columns[:_MAX_COLS]:
        k = df[c].nunique(dropna=True)
        if k / max(1, n) >= _HI_CARD_RATIO:
            out.append(c)
    return out

def _date_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns[:_MAX_COLS] if pd.api.types.is_datetime64_any_dtype(df[c])]

def _encode_target(y: pd.Series) -> Tuple[Optional[np.ndarray], Optional[Dict[Any, int]]]:
    if y is None or y.empty:
        return None, None
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(float).to_numpy(), None
    codes, uniques = pd.factorize(y, sort=True)
    return codes.astype(float), {str(val): int(idx) for idx, val in enumerate(uniques)}

def _numeric_signals(X_num: pd.DataFrame, y_vec: Optional[np.ndarray]) -> Dict[str, float]:
    """
    Dla regresji: |Pearson r| z targetem.
    Dla klasyfikacji (binarnej/ogólnej, przybliżenie): |corr| z kodami klas (ostrożnie interpretować).
    """
    scores: Dict[str, float] = {}
    if X_num.empty:
        return scores
    if y_vec is None:
        # bez targetu: wariancja jako sygnał „informacyjności”
        for c in X_num.columns:
            try:
                scores[c] = float(np.nanvar(X_num[c].astype(float)))
            except Exception:
                scores[c] = 0.0
        return scores
    # z targetem: korelacja Pearsona
    y0 = y_vec
    for c in X_num.columns:
        try:
            x = X_num[c].astype(float).to_numpy()
            mask = np.isfinite(x) & np.isfinite(y0)
            if mask.sum() < 3:
                continue
            xc = x[mask]
            yc = y0[mask]
            vx = np.nanvar(xc); vy = np.nanvar(yc)
            if vx <= 1e-12 or vy <= 1e-12:
                continue
            r = float(np.corrcoef(xc, yc)[0, 1])
            scores[c] = abs(r)
        except Exception:
            continue
    return scores

def _top_k(d: Dict[str, float], k: int) -> List[Dict[str, Any]]:
    return [{"feature": k1, "score": float(v)} for k1, v in sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]]

def _class_stats(y: pd.Series) -> Optional[Dict[str, Any]]:
    if y is None or y.empty:
        return None
    if _infer_problem_type(y) != "classification":
        return None
    vc = y.value_counts(dropna=False)
    total = vc.sum()
    top = vc.head(10).to_dict()
    frac = {str(k): float(v) / float(total) for k, v in top.items()}
    return {"classes": {str(k): int(v) for k, v in top.items()}, "fractions": frac, "n_classes": int(y.nunique(dropna=True))}

def _recommendations(schema: Dict[str, Any], const_cols: List[str], high_card: List[str],
                     problem_type: str, target: Optional[str], class_stats: Optional[Dict[str, Any]]) -> List[str]:
    rec: List[str] = []
    if any(v > 0.0 for v in schema.get("dtypes", {}).values()):  # no-op, placeholder warunku
        pass
    if const_cols:
        rec.append(f"Usuń stałe/niemal stałe kolumny: {', '.join(const_cols[:8])}{' …' if len(const_cols)>8 else ''}.")
    if high_card:
        rec.append(f"Dla wysokiej kardynalności użyj target encoding/WOE/Hashing: {', '.join(high_card[:8])}{' …' if len(high_card)>8 else ''}.")
    if problem_type == "classification" and class_stats:
        fracs = sorted(class_stats["fractions"].values(), reverse=True)
        if fracs and fracs[0] > 0.8:
            rec.append("Silna nierównowaga klas — rozważ stratified CV, class_weight/SMOTE, metryki niezależne od progu (PR AUC).")
        rec.append("Dla modeli drzewiastych spróbuj: LightGBM/XGBoost; dodaj kalibrację prawdopodobieństw (Platt/Isotonic) przy decyzjach progowych.")
    if problem_type == "regression":
        rec.append("Sprawdź nieliniowości i heteroskedastyczność; modele: LightGBM/CatBoost, a dla linii – ElasticNet.")
    rec.append("Włącz monitoring driftu (PSI/KL) na kluczowych cechach oraz przekaż referencję z treningu.")
    return rec

def _truncate_df(df: pd.DataFrame) -> pd.DataFrame:
    # Zabezpieczenie: duże zbiory i zbyt szerokie tabele
    dfn = df
    if len(dfn) > _MAX_ROWS:
        dfn = dfn.sample(_MAX_ROWS, random_state=42).reset_index(drop=True)
    if dfn.shape[1] > _MAX_COLS:
        keep = list(dfn.columns[:_MAX_COLS])
        dfn = dfn[keep]
    return dfn

# === NAZWA_SEKCJI === OPENAI (opcjonalnie) ===
def _llm_narrative(context_blob: str) -> Optional[str]:
    key = _get_openai_key()
    if not key:
        return None
    try:
        # lazy import (bez twardej zależności)
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=key)
        msgs = [
            {"role": "system", "content": SYSTEM_INSIGHTS},
            {"role": "user", "content": f"Provide a crisp, actionable 6–9 bullet insights list for this dataset context:\n{context_blob}\nTone: senior data scientist, Polish language."}
        ]
        resp = client.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=msgs,
            temperature=_OPENAI_TEMP,
            max_tokens=_OPENAI_TOKENS,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return None

def _context_blob_for_llm(schema: Dict[str, Any], na: Dict[str, float], target: Optional[str],
                          problem_type: str, top_signals: List[Dict[str, Any]], class_stats: Optional[Dict[str, Any]]) -> str:
    cols = list(schema.get("dtypes", {}).keys())
    na_top = sorted(na.items(), key=lambda x: x[1], reverse=True)[:10]
    na_txt = ", ".join([f"{k}:{v:.0%}" for k, v in na_top])
    sig_txt = ", ".join([f"{s['feature']}:{s['score']:.3f}" for s in top_signals[:10]])
    cls_txt = ""
    if class_stats:
        cls_txt = f" | classes={class_stats['classes']} frac≈{ {k: round(v,3) for k,v in class_stats['fractions'].items()} }"
    return (
        f"rows={schema['rows']} cols={schema['cols']} | target={target} | type={problem_type}"
        f" | top_signals=[{sig_txt}] | NA_top=[{na_txt}] | cols={cols[:30]}{' …' if len(cols)>30 else ''}{cls_txt}"
    )

# === NAZWA_SEKCJI === GŁÓWNA FUNKCJA ===
@rate_limited(0.6)
def generate_insights(df: pd.DataFrame, *, target: Optional[str] = None) -> Dict[str, Any]:
    """
    Zwraca deterministyczny słownik wniosków nt. danych:
      {
        "mode": "offline"|"online",
        "system": <SYSTEM_INSIGHTS>,
        "summary": {...},
        "eda": {...},
        "signals": [{"feature": str, "score": float}, ...],
        "recommendations": [str, ...],
        "online_text": Optional[str]
      }
    Przy braku klucza OpenAI lub błędzie sieci działa w pełnym trybie offline.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "mode": "offline",
            "system": SYSTEM_INSIGHTS,
            "summary": {"rows": 0, "cols": 0, "suggested_target": None, "notes": ["Pusty DataFrame."]},
            "eda": {},
            "signals": [],
            "recommendations": ["Załaduj dane i spróbuj ponownie."],
            "online_text": None,
        }

    # 1) przygotowanie danych (limity)
    dfx = _truncate_df(df)

    # 2) schemat i braki
    schema = _basic_schema(dfx)
    na = _missing_stats(dfx)
    const_cols = _constant_cols(dfx)
    hi_card = _high_card_cols(dfx)
    dates = _date_cols(dfx)

    # 3) target + typ problemu
    t_col = _infer_target(dfx, target)
    y = dfx[t_col] if (t_col and t_col in dfx.columns) else None
    problem = _infer_problem_type(y) if y is not None else "unknown"

    # 4) proste sygnały
    X_num = dfx.select_dtypes(include="number").copy()
    if t_col and t_col in X_num.columns:
        X_num = X_num.drop(columns=[t_col])  # nie używaj targetu jako cechy
    y_vec, y_map = _encode_target(y) if y is not None else (None, None)
    sig_scores = _numeric_signals(X_num, y_vec)
    top_signals = _top_k(sig_scores, _TOP_SIGNALS)

    # 5) klasy (dla klasyfikacji)
    cls_stats = _class_stats(y) if y is not None else None

    # 6) rekomendacje
    recos = _recommendations(schema, const_cols, hi_card, problem, t_col, cls_stats)

    # 7) podsumowanie (stabilny zestaw)
    summary = {
        "rows": int(schema["rows"]),
        "cols": int(schema["cols"]),
        "suggested_target": t_col,
        "notes": [
            "Sprawdź brakujące wartości i dystrybucje cech.",
            "Usuń stałe i nadmiarowe kolumny; rozważ standaryzację dla modeli liniowych.",
            "Rozważ modele drzewiaste (LightGBM/XGBoost) dla nieliniowości."
        ],
    }

    # 8) (opcjonalnie) narracja LLM
    online_text: Optional[str] = None
    mode = "offline"
    key = _get_openai_key()
    if key:
        blob = _context_blob_for_llm(schema, na, t_col, problem, top_signals, cls_stats)
        online_text = _llm_narrative(blob)
        mode = "online" if online_text else "offline"

    # 9) wynik
    out: Dict[str, Any] = {
        "mode": mode,
        "system": SYSTEM_INSIGHTS,
        "summary": summary,
        "eda": {
            "schema": schema,
            "missing_ratio": na,
            "constant_cols": const_cols,
            "high_cardinality": hi_card,
            "date_cols": dates,
            "target": {
                "name": t_col,
                "problem_type": problem,
                "class_stats": cls_stats
            }
        },
        "signals": top_signals,
        "recommendations": recos,
        "online_text": online_text
    }
    return out
