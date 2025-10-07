import io
import time
import hashlib
import streamlit as st
import pandas as pd

from src.data_processing.file_parser import parse_any
from src.data_processing.data_cleaner import quick_clean
from src.data_processing.feature_engineering import basic_feature_engineering
from src.utils.validators import basic_quality_checks
from src.ai_engine.nlp_analyzer import summarize_text

st.title("📤 Upload Data — Inteligentny Ingest")

# ---------------------------
# Sidebar: Advanced options
# ---------------------------
with st.sidebar:
    st.subheader("⚙️ Opcje wczytywania")
    merge_mode = st.radio(
        "Łączenie wielu plików",
        options=["union (po nazwach kolumn)", "intersection (wspólne kolumny)"],
        index=0,
        help="Union zachowuje wszystkie kolumny (puste uzupełnia NaN). Intersection zostawia tylko wspólne."
    )
    sample_preview = st.slider(
        "Podgląd pierwszych N wierszy", min_value=5, max_value=200, value=20, step=5
    )
    limit_rows = st.number_input(
        "Opcjonalny limit wierszy (0 = bez limitu)", min_value=0, value=0, step=1000,
        help="Dla bardzo dużych plików — pozwala skrócić czas wczytania/analizy."
    )
    st.markdown("---")
    st.subheader("🧹 Opcje czyszczenia")
    fill_missing_numeric = st.selectbox(
        "Wypełnianie braków (kolumny numeryczne)",
        options=["median", "mean", "none"],
        index=0
    )
    encode_low_cardinality = st.checkbox(
        "Koduj kategorie o niskiej krotności (≤ 20 unikalnych)", value=True
    )

# ---------------------------
# Upload
# ---------------------------
files = st.file_uploader(
    "Wybierz pliki (CSV/XLSX/JSON/DOCX/PDF/legacy DOC*)",
    type=["csv", "xlsx", "json", "docx", "pdf", "doc"],
    accept_multiple_files=True
)

goal = st.text_input(
    "🎯 Cel biznesowy / pytanie analityczne",
    placeholder="np. prognoza sprzedaży Q4, prognoza zużycia energii w jednostkach X",
)
st.session_state["goal"] = goal

# ---------------------------
# Helpers
# ---------------------------
def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]

@st.cache_data(show_spinner=False)
def _cached_parse(name: str, data: bytes):
    return parse_any(name, data)

def _concat_frames(frames: list[pd.DataFrame], how: str) -> pd.DataFrame:
    if how.startswith("union"):
        return pd.concat(frames, axis=0, ignore_index=True, sort=True)
    # intersection: dopasuj do wspólnego zbioru kolumn
    common = set(frames[0].columns)
    for f in frames[1:]:
        common &= set(f.columns)
    aligned = [f[list(common)].copy() for f in frames]
    return pd.concat(aligned, axis=0, ignore_index=True)

def _apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df2 = quick_clean(df)
    # Opcjonalne zmiany wg sidebaru
    if fill_missing_numeric != "none":
        for c in df2.select_dtypes(include="number").columns:
            if fill_missing_numeric == "median":
                df2[c] = df2[c].fillna(df2[c].median())
            elif fill_missing_numeric == "mean":
                df2[c] = df2[c].fillna(df2[c].mean())
    df2 = basic_feature_engineering(df2) if encode_low_cardinality else df2
    return df2

def _quick_stats(df: pd.DataFrame) -> dict:
    stats = basic_quality_checks(df)
    stats["memory_mb"] = round(df.memory_usage(deep=True).sum() / 1e6, 2)
    return stats

# ---------------------------
# Main ingest
# ---------------------------
dfs, texts = [], []
manifest = []

if files:
    with st.status("🔎 Przetwarzam pliki...", expanded=True) as status:
        for f in files:
            data = f.read()
            fid = _hash_bytes(data)
            st.write(f"• **{f.name}**  _(id: {fid})_")

            # Ostrzeżenie dla legacy .doc
            if f.name.lower().endswith(".doc"):
                st.warning(f"{f.name}: format .doc jest legacy — sugeruję konwersję do .docx dla lepszego rezultatu.")

            try:
                df, txt = _cached_parse(f.name, data)
            except Exception as e:
                st.error(f"{f.name}: błąd parsowania: {e}")
                continue

            # DataFrame
            if isinstance(df, pd.DataFrame):
                if limit_rows and len(df) > limit_rows:
                    df = df.head(limit_rows).copy()
                dfs.append(df)
                manifest.append({"file": f.name, "id": fid, "rows": len(df), "cols": df.shape[1], "type": "table"})
            # Text
            if txt:
                preview = txt[:2000] + ("..." if len(txt) > 2000 else "")
                texts.append({"file": f.name, "id": fid, "text": preview, "summary": summarize_text(preview)})
                manifest.append({"file": f.name, "id": fid, "chars": len(txt), "type": "text"})

        status.update(label="✅ Zakończono wczytywanie", state="complete")

# ---------------------------
# Results / UI
# ---------------------------
if dfs:
    # Łączenie
    with st.spinner("Łączę ramki danych..."):
        df = _concat_frames(dfs, merge_mode)
        st.session_state["df_raw"] = df

    # KPI
    st.subheader("📊 Podgląd i metryki")
    c1, c2, c3, c4 = st.columns(4)
    stats = _quick_stats(df)
    c1.metric("Wiersze", f"{stats['rows']:,}")
    c2.metric("Kolumny", f"{stats['cols']:,}")
    c3.metric("Braki (%)", f"{stats['missing_pct']*100:.2f}%")
    c4.metric("Duplikaty", f"{stats['dupes']:,}")

    st.caption(f"Zużycie pamięci: **{stats['memory_mb']} MB** • Tryb łączenia: **{merge_mode}**")

    st.subheader("📄 Podgląd danych")
    st.dataframe(df.head(sample_preview), use_container_width=True)

    # Szybkie czyszczenie + FE
    if st.button("🧹 Szybkie czyszczenie + FE", type="primary"):
        t0 = time.time()
        with st.spinner("Czyszczę i wzbogacam cechy..."):
            df2 = _apply_cleaning(df)
        st.session_state["df"] = df2
        st.success(f"Dane wyczyszczone i wzbogacone. ({len(df2):,}×{df2.shape[1]})  ⏱️ {time.time()-t0:.2f}s")
        st.dataframe(df2.head(sample_preview), use_container_width=True)

# Teksty z dokumentów
if texts:
    st.subheader("📄 Wydobyty tekst (DOCX/PDF)")
    for t in texts:
        with st.expander(f"{t['file']} — podsumowanie"):
            if t.get("summary"):
                st.markdown(f"**Skrót:** {t['summary']}")
            st.text_area("Fragment", t["text"], height=160, key=f"ta_{t['id']}")

# Manifest + wskazówka co dalej
if files:
    st.divider()
    st.caption("Manifest wczytania (do debugowania / audytu):")
    st.json(manifest)

st.info("Po wczytaniu danych przejdź do **EDA Analysis** lub od razu do **AI Insights**.")
