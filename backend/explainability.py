from __future__ import annotations
from typing import Dict, Any
import numpy as np
import shap

def estimate_shap_values(pipeline, X_sample):
    try:
        model = pipeline.named_steps["model"]
        # Use TreeExplainer when possible
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(pipeline.named_steps["pre"].transform(X_sample))
        return sv
    except Exception:
        return None
