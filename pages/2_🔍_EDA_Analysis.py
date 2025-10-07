import io
import time
import pathlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.data_processing.data_validator import validate
from src.data_processing.data_profiler import make_profile_html
from src.utils.helpers import infer_problem_type

st.title("🔍 EDA Analysis — PRO")

# ---------------------------
# Dane z sesji
# ---------------------------
df = st.session_state.get("df") or st.session_state.get("df_raw")
if df is None or not isinstance(df, pd.DataFrame) or df.empty:
    st.warning("Brak danych. Przejdź do Upload.")
    st.stop()

# ---------------------------
# Sidebar: Opcje EDA
# ---------------------------
with st.sidebar:
    st.subheader("⚙️ Opcje EDA")
    sample_n = st.number_input("Sample (0 = pełny zbiór)", min_value=0, value=min(5000, len(df)), step=500,
                               help="Użyj próby dla szybszych wykresów i profilowania.")
    corr_method = st.selectbox("Metoda korelacji", ["pearson", "spearman", "kendall"], index=0)
    top_k = st.slider("Top kolumn (unikalne / info)", 5, 50, 20)
    profile_minimal = st.checkbox("Profil minimalny (szybszy)", value=True)
    export_profile_btn = st.checkbox("Po wygenerowaniu dołącz przycisk pobrania HTML", value=True)

# Próbka do części analiz (bez modyfikowania oryginału)
df_view = df.sample(sample_n) if (sample_n and sample_n > 0 and sample_n < len(df)) else df

# ---------------------------
# KPI / jakość
# ---------------------------
st.subheader("📊 KPI jakości danych")
stats = validate(df_view)
mem_mb = round(df_view.memory_usage(deep=True).sum()/1e6, 2)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Wiersze", f"{stats['rows']:,}")
c2.metric("Kolumny", f"{stats['cols']:,}")
c3.metric("Braki (%)", f"{stats['missing_pct']*100:.2f}%")
c4.metric("Duplikaty", f"{stats['dupes']:,}")
st.caption(f"Zużycie pamięci (próba): **{mem_mb} MB**")

# ---------------------------
# Szybka charakterystyka kolumn
# ---------------------------
st.subheader("🧭 Słownik danych (skrócony)")

def _data_dictionary(d: pd.DataFrame) -> pd.DataFrame:
    out = []
    for c in d.columns:
        s = d[c]
        miss = float(s.isna().mean())
        nunique = int(s.nunique(dropna=True))
        dtype = str(s.dtype)
        sample_val = s.dropna().iloc[0] if s.dropna().shape[0] else ""
        out.append({
            "column": c,
            "dtype": dtype,
            "missing_pct": round(miss*100, 2),
            "nunique": nunique,
            "example": str(sample_val)[:60],
        })
    return pd.DataFrame(out).sort_values(["missing_pct","nunique"], ascending=[False, True]).head(top_k)

dd = _data_dictionary(df_view)
st.dataframe(dd, use_container_width=True)

# Eksport CSV słownika
csv_buf = io.StringIO()
dd.to_csv(csv_buf, index=False)
st.download_button("⬇️ Pobierz słownik danych (CSV)", csv_buf.getvalue(), file_name="data_dictionary.csv", mime="text/csv")

# ---------------------------
# Podgląd danych
# ---------------------------
with st.expander("📄 Podgląd danych (próba)", expanded=True):
    st.dataframe(df_view.head(50), use_container_width=True)

# ---------------------------
# Braki i typy
# ---------------------------
st.subheader("🕳️ Brakujące wartości")
nulls = df_view.isna().mean().sort_values(ascending=False)
if nulls.sum() == 0:
    st.success("Brak braków w próbie.")
else:
    miss_df = nulls[nulls > 0].mul(100).round(2).rename("missing_pct").to_frame()
    st.dataframe(miss_df, use_container_width=True)

# ---------------------------
# Korelacje (tylko numeryczne)
# ---------------------------
st.subheader("🔗 Korelacje (numeryczne)")
num = df_view.select_dtypes(include=np.number)
if num.shape[1] >= 2:
    corr = num.corr(method=corr_method)
    fig = px.imshow(corr, text_auto=False, aspect="auto", title=f"Macierz korelacji ({corr_method})")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Za mało kolumn numerycznych do wyliczenia korelacji.")

# ---------------------------
# Rozkłady — szybkie wykresy
# ---------------------------
with st.expander("📈 Rozkłady / Relacje"):
    cols = list(df_view.columns)
    num_cols = list(num.columns)
    if num_cols:
        x = st.selectbox("Histogram kolumny", options=num_cols)
        st.plotly_chart(px.histogram(df_view, x=x, nbins=40, title=f"Histogram: {x}"), use_container_width=True)
    if len(num_cols) >= 2:
        x2 = st.selectbox("Wykres rozrzutu: X", options=num_cols, key="sc_x")
        y2 = st.selectbox("Wykres rozrzutu: Y", options=[c for c in num_cols if c != x2], key="sc_y")
        st.plotly_chart(px.scatter(df_view, x=x2, y=y2, trendline="ols",
                                   title=f"Scatter: {x2} vs {y2}"),
                        use_container_width=True)

# ---------------------------
# Sugestia celu i typu problemu
# ---------------------------
st.subheader("🎯 Sugestia celu / typu problemu")
target_candidates = []
# heurystyki nazw kolumn na cel
name_hints = ("target","y","label","sales","sprzeda","zuzy","consum","amount","profit","revenue")
for c in df.columns:
    lc = c.lower()
    if any(h in lc for h in name_hints):
        target_candidates.append(c)
target = st.selectbox("Wybierz kolumnę celu (opcjonalnie)", options=["—"] + list(dict.fromkeys(target_candidates)))
if target != "—":
    st.session_state["target"] = target
    ptype = infer_problem_type(df, target)
    if ptype == "classification":
        st.info("Wykryto **klasyfikację** (niewiele unikalnych wartości w celu).")
    elif ptype == "regression":
        st.info("Wykryto **regresję** (wartości ciągłe).")
    elif ptype == "timeseries":
        st.info("Wykryto **szereg czasowy** (kolumna czasu/indeks czasowy).")
    else:
        st.info("Nie udało się jednoznacznie określić typu problemu.")
else:
    st.caption("Nie wybrano celu — możesz zrobić to później w zakładce Predictions/Forecasting.")

# ---------------------------
# Profil danych (ydata-profiling) z cache
# ---------------------------
st.subheader("🧪 Raport profilujący (HTML)")
with st.expander("Pokaż raport profilujący"):
    @st.cache_resource(show_spinner=False)
    def _cached_profile_html(df_for_profile: pd.DataFrame, title: str, minimal: bool) -> str:
        # użyj mniejszej próby, aby uniknąć długiego czasu generowania
        sample = df_for_profile.sample(min(len(df_for_profile), 2000)) if len(df_for_profile) > 2000 else df_for_profile
        return make_profile_html(sample, title=title)

    try:
        t0 = time.time()
        html = _cached_profile_html(df_view, "Data Profile", profile_minimal)
        st.components.v1.html(html, height=600, scrolling=True)
        st.caption(f"⏱️ Generowanie (cache-aware): {time.time()-t0:.2f}s")
        if export_profile_btn:
            st.download_button("⬇️ Pobierz profil (HTML)", html, file_name="profile.html", mime="text/html")
            # opcjonalnie zapisz do katalogu projektu
            out = pathlib.Path("data/exports/profile.html")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(html, encoding="utf-8")
    except Exception as e:
        st.info(f"Profilowanie pominięte: {e}")

st.success("Gotowe! Przejdź do **AI Insights** lub **Predictions**.")
