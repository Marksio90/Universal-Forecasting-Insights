from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any, List
import numpy as np
import pandas as pd

# === TYPY DANYCH ===
@dataclass
class PSIBinRow:
    bin_left: float
    bin_right: float
    p_base: float
    p_curr: float
    psi_contrib: float
    n_base: int
    n_curr: int

@dataclass
class PSIResult:
    feature: str
    psi: float
    bins: List[PSIBinRow]
    n_base: int
    n_curr: int
    strategy: str
    bins_count: int
    dropped_nan_base: int
    dropped_nan_curr: int
    note: Optional[str] = None

# === NARZĘDZIA ===
def _is_constant(x: pd.Series) -> bool:
    return x.nunique(dropna=True) <= 1

def _clip_series(x: pd.Series, clip_low: Optional[float], clip_high: Optional[float]) -> pd.Series:
    if clip_low is None and clip_high is None:
        return x
    lo = x.quantile(clip_low) if clip_low is not None else None
    hi = x.quantile(clip_high) if clip_high is not None else None
    return x.clip(lower=lo if lo is not None else x.min(), upper=hi if hi is not None else x.max())

# === PSI 1D ===
def population_stability_index(
    base: pd.Series,
    curr: pd.Series,
    bins: int = 10,
    *,
    strategy: Literal["quantile", "uniform"] = "quantile",
    min_bin_pct: float = 1e-6,
    epsilon: float = 1e-12,
    clip_low: Optional[float] = None,
    clip_high: Optional[float] = None,
    feature_name: Optional[str] = None,
) -> PSIResult:
    """
    Oblicza PSI dla jednej cechy.
    - strategy="quantile": równa liczność (qcut)
    - strategy="uniform": równa szerokość (cut)
    - min_bin_pct: dolna granica udziału (stabilizacja), fallback gdy koszyk pusty
    - epsilon: stabilizacja log-ratio

    Zwraca PSIResult z pełnym breakdownem binów.
    """
    if not isinstance(base, pd.Series) or not isinstance(curr, pd.Series):
        raise TypeError("base and curr must be pandas.Series")

    name = feature_name or (base.name or "feature")

    # Drop NaN + opcjonalny clipping
    base_clean = base.dropna()
    curr_clean = curr.dropna()
    dropped_base = len(base) - len(base_clean)
    dropped_curr = len(curr) - len(curr_clean)

    base_clean = _clip_series(base_clean, clip_low, clip_high)
    curr_clean = _clip_series(curr_clean, clip_low, clip_high)

    # Edge cases
    if len(base_clean) == 0 or len(curr_clean) == 0:
        return PSIResult(
            feature=name, psi=float("nan"), bins=[], n_base=len(base), n_curr=len(curr),
            strategy=strategy, bins_count=bins, dropped_nan_base=dropped_base, dropped_nan_curr=dropped_curr,
            note="empty_series_after_nan_clip"
        )
    if _is_constant(base_clean) and _is_constant(curr_clean):
        # identycznie stałe -> brak przesunięcia
        return PSIResult(
            feature=name, psi=0.0, bins=[], n_base=len(base), n_curr=len(curr),
            strategy=strategy, bins_count=1, dropped_nan_base=dropped_base, dropped_nan_curr=dropped_curr,
            note="both_constant"
        )

    # Ustalenie granic binów na podstawie BASE (stabilność)
    if strategy == "quantile":
        # quantile bins — mogą się duplikować, więc drop duplicates
        qs = np.linspace(0, 1, bins + 1)
        edges = np.unique(base_clean.quantile(qs).to_numpy())
        # zabezpieczenie: jeśli mało unikalnych wartości
        if len(edges) <= 2:  # sprowadza się do stałej/2-koszykowej
            edges = np.linspace(base_clean.min(), base_clean.max(), min(bins, 2) + 1)
    elif strategy == "uniform":
        edges = np.linspace(base_clean.min(), base_clean.max(), bins + 1)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")

    # cut z include_lowest=True, żeby pierwszy koszyk łapał min
    base_bins = pd.cut(base_clean, bins=edges, include_lowest=True, duplicates="drop")
    curr_bins = pd.cut(curr_clean, bins=edges, include_lowest=True, duplicates="drop")

    # Rozkłady
    b_counts = base_bins.value_counts().sort_index()
    c_counts = curr_bins.value_counts().sort_index()

    # Wyrównanie indeksów (koszyków)
    all_idx = b_counts.index.union(c_counts.index)
    b_counts = b_counts.reindex(all_idx, fill_value=0)
    c_counts = c_counts.reindex(all_idx, fill_value=0)

    b_total = b_counts.sum()
    c_total = c_counts.sum()

    # Udziały z dolnym progiem min_bin_pct (stabilizacja)
    b_pct = (b_counts / max(b_total, 1)).clip(lower=min_bin_pct)
    c_pct = (c_counts / max(c_total, 1)).clip(lower=min_bin_pct)

    # PSI = sum((c - b) * ln(c/b))
    contrib = (c_pct - b_pct) * np.log((c_pct + epsilon) / (b_pct + epsilon))
    psi_value = float(contrib.sum())

    # Budujemy breakdown
    rows: List[PSIBinRow] = []
    for interval, pb, pc, cc, nb, nc in zip(all_idx, b_pct, c_pct, contrib, b_counts, c_counts):
        left = float(interval.left) if hasattr(interval, "left") else float("nan")
        right = float(interval.right) if hasattr(interval, "right") else float("nan")
        rows.append(PSIBinRow(
            bin_left=left,
            bin_right=right,
            p_base=float(pb),
            p_curr=float(pc),
            psi_contrib=float(cc),
            n_base=int(nb),
            n_curr=int(nc),
        ))

    return PSIResult(
        feature=name,
        psi=psi_value,
        bins=rows,
        n_base=int(b_total),
        n_curr=int(c_total),
        strategy=strategy,
        bins_count=len(all_idx),
        dropped_nan_base=int(dropped_base),
        dropped_nan_curr=int(dropped_curr),
        note=None
    )

# === PSI DLA CAŁEGO DATAFRAME ===
def psi_df(
    base_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    *,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    bins: int = 10,
    strategy: Literal["quantile", "uniform"] = "quantile",
    min_bin_pct: float = 1e-6,
    epsilon: float = 1e-12,
    clip_low: Optional[float] = None,
    clip_high: Optional[float] = None,
) -> Dict[str, PSIResult]:
    """
    Liczy PSI dla wszystkich kolumn numerycznych (lub podanych w include).
    Zwraca dict {kolumna: PSIResult}, posortowany malejąco po PSI (Ordered).
    """
    if include is None:
        cols = [c for c in base_df.select_dtypes(include="number").columns if c in curr_df.columns]
    else:
        cols = [c for c in include if c in base_df.columns and c in curr_df.columns]
    if exclude:
        cols = [c for c in cols if c not in set(exclude)]

    results: Dict[str, PSIResult] = {}
    for c in cols:
        res = population_stability_index(
            base_df[c], curr_df[c], bins=bins, strategy=strategy,
            min_bin_pct=min_bin_pct, epsilon=epsilon,
            clip_low=clip_low, clip_high=clip_high, feature_name=c
        )
        results[c] = res

    # sort malejąco po psi
    results = dict(sorted(results.items(), key=lambda kv: (np.nan_to_num(kv[1].psi, nan=-1),), reverse=True))
    return results
