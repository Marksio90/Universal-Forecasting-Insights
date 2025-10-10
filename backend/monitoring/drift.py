from __future__ import annotations
import pandas as pd, numpy as np

def population_stability_index(base: pd.Series, curr: pd.Series, bins: int = 10)->float:
    base = pd.qcut(base.rank(method='first'), q=bins, duplicates='drop')
    curr = pd.qcut(curr.rank(method='first'), q=bins, duplicates='drop')
    b = base.value_counts(normalize=True).sort_index()
    c = curr.value_counts(normalize=True).sort_index()
    idx = b.index.union(c.index)
    b = b.reindex(idx, fill_value=1e-6)
    c = c.reindex(idx, fill_value=1e-6)
    psi = float(((c - b) * np.log((c + 1e-12) / (b + 1e-12))).sum())
    return psi
