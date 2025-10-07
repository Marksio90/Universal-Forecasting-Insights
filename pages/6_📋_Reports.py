import io
import json
import time
import pathlib
import zipfile
import streamlit as st
import pandas as pd
from src.ai_engine.report_generator import build_report_html

st.title("📋 Reports & Export — PRO")

# ---------------------------------
# Dane z sesji (bezpieczne odczyty)
# ---------------------------------
df = st.session_state.get("df") or st.session_state.get("df_raw")
goal = st.session_state.get("goal")
problem_type = st.session_state.get("problem_type")
target = st.session_state.get("target")
model = st.session_state.get("model")  # opcjonalnie, gdy był trening

# ewentualne artefakty z innych zakładek, jeśli je tam zapiszesz:
forecast_df = st.session_state.get("forecast_df")  # jeśli gdzieś odkładasz prognozę
anomalies_df = st.session_state.get("anomalies_df")  # jeśli w przyszłości dodasz wykrywanie anomalii
automl_metrics = st.session_state.get("last_metrics")  # jeśli chcesz, ustaw w Predictions

# ---------------------------------
# Opcje raportu (sidebar)
# ---------------------------------
with st.sidebar:
    st.subheader("⚙️ Opcje raportu")
    include_data_dictionary = st.checkbox("Dołącz skrócony słownik danych", value=True)
    include_df_preview = st.checkbox("Dołącz podgląd danych (tabela)", value=False)
    include_kpis = st.checkbox("Dołącz KPI danych", value=True)
    include_tags = st.checkbox("Dołącz tagi kontekstu", value=True)
    # jeśli w przyszłości zapiszesz progn/anom do session_state — pojawią się w raporcie:
    include_forecast = st.checkbox("Dołącz prognozę (jeśli dostępna)", value=True)
    include_anomalies = st.checkbox("Dołącz anomalie (jeśli dostępne)", value=True)

# ---------------------------------
# Helpery do formatu raportu
# ---------------------------------
def df_to_table_dict(d: pd.DataFrame, max_rows: int = 500) -> dict:
    """Konwertuje DataFrame do struktury {columns, rows} akceptowanej przez template."""
    if d is None or not isinstance(d, pd.DataFrame) or d.empty:
        return None
    d2 = d.head(max_rows).copy()
    return {"columns": list(map(str, d2.columns)), "rows": d2.astype(object).to_dict(orient="records")}

def build_kpis_from_df(d: pd.DataFrame) -> list[dict]:
    if d is None or not isinstance(d, pd.DataFrame) or d.empty:
        return []
    missing_pct = float((d.isna().sum().sum()) / (d.size or 1)) * 100.0
    dupes = int(d.duplicated().sum())
    return [
        {"label": "Wiersze", "value": f"{len(d):,}", "status": "ok"},
        {"label": "Kolumny", "value": f"{d.shape[1]:,}", "status": "ok"},
        {"label": "Braki", "value": f"{missing_pct:.2f}%", "status": "warn" if missing_pct > 1.0 else "ok"},
        {"label": "Duplikaty", "value": f"{dupes:,}", "status": "warn" if dupes > 0 else "ok"},
    ]

def data_dictionary(d: pd.DataFrame, top_k: int = 30) -> pd.DataFrame:
    if d is None or not isinstance(d, pd.DataFrame) or d.empty:
        return pd.DataFrame()
    rows = []
    for c in d.columns:
        s = d[c]
        rows.append({
            "column": str(c),
            "dtype": str(s.dtype),
            "missing_pct": round(float(s.isna().mean()) * 100.0, 2),
            "nunique": int(s.nunique(dropna=True)),
            "example": (str(s.dropna().iloc[0])[:60] if s.dropna().shape[0] else "")
        })
    out = pd.DataFrame(rows).sort_values(["missing_pct", "nunique"], ascending=[False, True]).head(top_k)
    return out

# ---------------------------------
# Zbieranie metryk / kontekstu
# ---------------------------------
metrics = {}
if problem_type:
    metrics["problem_type"] = problem_type
if target:
    metrics["target"] = target
if isinstance(automl_metrics, dict):
    metrics["automl"] = automl_metrics

# Tagi kontekstu
tags = []
if include_tags:
    if goal: tags.append("goal")
    if problem_type: tags.append(problem_type)
    if target: tags.append(f"y:{target}")
    if df is not None: tags.append(f"{len(df)}x{df.shape[1]}")

# KPI
kpis = build_kpis_from_df(df) if (include_kpis and isinstance(df, pd.DataFrame)) else []

# Słownik danych
dd = data_dictionary(df)
dd_table = df_to_table_dict(dd) if (include_data_dictionary and not dd.empty) else None

# Podgląd danych (mały)
df_preview_table = df_to_table_dict(df.head(50)) if (include_df_preview and isinstance(df, pd.DataFrame)) else None

# Forecast/anomalies jeśli obecne i wybrano
forecast_blob = df_to_table_dict(forecast_df) if (include_forecast and isinstance(forecast_df, pd.DataFrame)) else None
anomalies_blob = df_to_table_dict(anomalies_df) if (include_anomalies and isinstance(anomalies_df, pd.DataFrame)) else None

# Model card (jeśli chcesz — do wypełnienia w Predictions i zapisania do session_state)
model_card = st.session_state.get("model_card")
# jeżeli nie było, spróbuj zbudować minimalny na podstawie dostępnych informacji:
if model_card is None and model is not None:
    model_card = {
        "name": getattr(model, "__class__", type("M", (), {})).__name__,
        "type": problem_type or "—",
        "version": time.strftime("%Y.%m.%d"),
        "dataset": "session_dataframe",
        "split": "—",
        "hparams": None,
        "metrics": automl_metrics or None
    }

# ---------------------------------
# Notatki
# ---------------------------------
notes = st.text_area("Notatki / Wnioski", placeholder="Np. rekomendacje biznesowe…")

# ---------------------------------
# Budowa kontekstu do template
# ---------------------------------
context = {
    "title": "Raport Biznesowy",
    "subtitle": goal or "",
    "run_meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M")},
    "metrics": metrics,
    "notes": notes,
    "kpis": kpis if kpis else None,
    "tags": tags if tags else None,
    # Tabele (opcjonalne sekcje w template)
    "insights": st.session_state.get("ai_top_insights"),          # jeśli zapiszesz w AI Insights
    "recommendations": st.session_state.get("ai_recommendations"),# jeśli zapiszesz w AI Insights
    "forecast": forecast_blob,
    "anomalies": anomalies_blob,
    # Dodatkowe sekcje użytkowe
    "data_dictionary": dd_table,      # nieużywana w template PRO domyślnie, ale zostawiamy w kontekście
    "df_preview": df_preview_table,   # jw.
    "model_card": model_card,
}

# ---------------------------------
# Podgląd kontekstu (debug)
# ---------------------------------
with st.expander("🧪 Podgląd kontekstu (debug)"):
    st.json({k: ("<large object>" if isinstance(v, (list, dict)) and len(str(v)) > 500 else v) for k, v in context.items()})

# ---------------------------------
# Generowanie raportu
# ---------------------------------
col_a, col_b = st.columns(2)
with col_a:
    if st.button("🧾 Generuj raport HTML", type="primary"):
        html = build_report_html(context)
        out_path = pathlib.Path("data/exports/report.html")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        st.success(f"Zapisano: {out_path}")
        st.download_button("⬇️ Pobierz raport (HTML)", data=html, file_name="report.html", mime="text/html")
        with st.expander("Podgląd raportu"):
            st.components.v1.html(html, height=680, scrolling=True)

with col_b:
    if st.button("📦 Eksport ZIP (raport + dane + meta)"):
        html = build_report_html(context)
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
            # raport
            z.writestr("report.html", html)
            # dane
            if isinstance(df, pd.DataFrame) and not df.empty:
                z.writestr("data.csv", df.to_csv(index=False))
            # meta
            z.writestr("meta.json", json.dumps(context, ensure_ascii=False, indent=2))
            # opcjonalnie: dołącz słownik danych
            if dd_table is not None:
                dd_csv = io.StringIO()
                dd.to_csv(dd_csv, index=False)
                z.writestr("data_dictionary.csv", dd_csv.getvalue())
            # opcjonalnie: dołącz prognozę/anomalie
            if forecast_df is not None and isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
                z.writestr("forecast.csv", forecast_df.to_csv(index=False))
            if anomalies_df is not None and isinstance(anomalies_df, pd.DataFrame) and not anomalies_df.empty:
                z.writestr("anomalies.csv", anomalies_df.to_csv(index=False))
        mem.seek(0)
        st.download_button("⬇️ Pobierz export.zip", data=mem, file_name="export.zip")

st.caption("Tip: aby w raporcie pojawiły się sekcje **Wizualizacje/Forecast/Anomalie/Model Card**, zapisz je wcześniej do `st.session_state` (lub włącz odpowiednie opcje).")
