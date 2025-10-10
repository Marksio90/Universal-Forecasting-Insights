from __future__ import annotations
import pandas as pd
from backend.automl_fusion import train_fusion
from backend.model_io import save_model
from backend.reports.report_builder import build_html_summary, build_pdf_from_html

def train_job(csv_path: str, target: str, trials: int = 30)->str:
    df = pd.read_csv(csv_path)
    res = train_fusion(df, target=target, trials=trials)
    model_path = save_model(res.model, f"FUSION_{res.problem_type}")
    html = build_html_summary(df, {res.metric_name: res.best_score})
    pdf = build_pdf_from_html(html)
    return model_path
