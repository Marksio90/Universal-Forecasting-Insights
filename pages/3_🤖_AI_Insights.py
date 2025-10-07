import io
import time
import json
import streamlit as st
import pandas as pd

from src.ai_engine.insights_generator import generate_insights
from src.utils.validators import basic_quality_checks

st.title("🤖 AI Insights — PRO")

# ---------------------------
# Dane z sesji
# ---------------------------
df = st.session_state.get("df") or st.session_state.get("df_raw")
if df is None or not isinstance(df, pd.DataFrame) or df.empty:
    st.warning("Brak danych. Przejdź do Upload.")
    st.stop()

goal = st.session_state.get("goal", "Nie podano celu biznesowego")

# ---------------------------
# Sidebar: Opcje
# ---------------------------
with st.sidebar:
    st.subheader("⚙️ Ustawienia analizy AI")
    depth = st.radio(
        "Tryb analizy",
        ["Szybka (krótkie wnioski)", "Pogłębiona (pełny raport)"],
        index=1,
    )
    include_schema = st.checkbox("Uwzględnij strukturę kolumn", value=True)
    include_quality = st.checkbox("Uwzględnij metryki jakości danych", value=True)
    show_json = st.checkbox("Pokaż surowy JSON (debug)", value=False)

# ---------------------------
# Kontekst danych
# ---------------------------
stats = basic_quality_checks(df)
meta = {
    "rows": stats["rows"],
    "cols": stats["cols"],
    "missing_pct": round(stats["missing_pct"] * 100, 2),
    "dupes": stats["dupes"],
}
schema = [
    {"col": c, "dtype": str(df[c].dtype), "nunique": int(df[c].nunique(dropna=True))}
    for c in df.columns
]

# ---------------------------
# Prompt building
# ---------------------------
prompt_parts = [
    f"Cel analizy: {goal}",
    f"Wielkość danych: {meta['rows']}×{meta['cols']}, braki {meta['missing_pct']}%, duplikaty {meta['dupes']}",
]
if include_schema:
    prompt_parts.append(
        "Struktura danych:\n" + "\n".join([f"- {s['col']} ({s['dtype']}, {s['nunique']} unikalnych)" for s in schema[:25]])
        + ("..." if len(schema) > 25 else "")
    )
if include_quality:
    prompt_parts.append(
        "Metryki jakości:\n" + json.dumps(meta, indent=2, ensure_ascii=False)
    )
prompt_parts.append(
    "Wygeneruj syntetyczny raport w formacie JSON z kluczami: "
    "`summary`, `top_insights`, `recommendations`, `risks`."
)
if depth.startswith("Pogłębiona"):
    prompt_parts.append(
        "Dodaj więcej szczegółów, przykłady, rekomendacje KPI i hipotezy."
    )

user_prompt = "\n\n".join(prompt_parts)

# ---------------------------
# Cache dla AI (by uniknąć ponownych żądań)
# ---------------------------
@st.cache_data(show_spinner=False)
def cached_ai_insights(df_hash: str, prompt: str) -> dict:
    """Wraper do generate_insights, zwraca JSON"""
    try:
        txt = generate_insights(df, prompt)
        # Próba sparsowania JSONa (jeśli AI zwróci tekst)
        try:
            data = json.loads(txt)
            if isinstance(data, dict):
                return data
        except Exception:
            # fallback: wrzucamy całość w summary
            return {"summary": txt}
    except Exception as e:
        return {"summary": f"Błąd generowania insightów: {e}"}
    return {"summary": "Brak wyników"}

# ---------------------------
# Hash danych (dla cache)
# ---------------------------
df_hash = str(hash(tuple(df.columns))) + str(len(df))

# ---------------------------
# Uruchomienie AI
# ---------------------------
if st.button("🔮 Wygeneruj insighty", type="primary"):
    with st.spinner("Generuję insighty z OpenAI..."):
        t0 = time.time()
        insights = cached_ai_insights(df_hash, user_prompt)
        dt = time.time() - t0

    st.success(f"✅ Wygenerowano w {dt:.1f}s")

    # -----------------------
    # Prezentacja wyników
    # -----------------------
    if "summary" in insights:
        st.markdown(f"### 🧭 Executive Summary\n{insights['summary']}")
    if "top_insights" in insights:
        st.markdown("### 📌 Top Insights")
        for i, item in enumerate(insights["top_insights"], 1):
            st.markdown(f"{i}. {item}")
    if "recommendations" in insights:
        st.markdown("### 💡 Rekomendacje")
        for i, item in enumerate(insights["recommendations"], 1):
            st.markdown(f"- {item}")
    if "risks" in insights:
        st.markdown("### ⚠️ Ryzyka / Anomalie")
        for i, item in enumerate(insights["risks"], 1):
            st.markdown(f"- {item}")

    # Eksport
    export_txt = io.StringIO()
    json.dump(insights, export_txt, ensure_ascii=False, indent=2)
    st.download_button(
        "⬇️ Pobierz insighty (JSON)",
        data=export_txt.getvalue(),
        file_name="ai_insights.json",
        mime="application/json",
    )

    if show_json:
        st.json(insights)
else:
    st.caption("Kliknij przycisk, aby wygenerować insighty z OpenAI.")
