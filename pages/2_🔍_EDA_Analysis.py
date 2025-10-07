"""
Moduł EDA Analysis PRO - Zaawansowana eksploracyjna analiza danych.

Funkcjonalności:
- Kompleksowe statystyki jakości danych
- Słownik danych z eksportem
- Analiza braków i korelacji
- Interaktywne wizualizacje (Plotly)
- Automatyczne wykrywanie typu problemu
- Generowanie raportów HTML (ydata-profiling)
- Sugestie kolumn docelowych
"""

from __future__ import annotations

import io
import time
import logging
import pathlib
from typing import Optional, Literal
from dataclasses import dataclass

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_processing.data_validator import validate
from src.data_processing.data_profiler import make_profile_html
from src.utils.helpers import infer_problem_type

# ========================================================================================
# KONFIGURACJA
# ========================================================================================

logger = logging.getLogger(__name__)

# Limity bezpieczeństwa
MAX_SAMPLE_SIZE = 50_000
MAX_PROFILE_SIZE = 5_000
MAX_CORRELATION_COLS = 100
DEFAULT_SAMPLE = 5_000

# Hinty dla wykrywania kolumny docelowej
TARGET_NAME_HINTS = (
    "target", "y", "label", "sales", "sprzeda", "zuzy", "consum",
    "amount", "profit", "revenue", "price", "cena", "wartość", "value"
)

CorrelationMethod = Literal["pearson", "spearman", "kendall"]


# ========================================================================================
# DATACLASSES
# ========================================================================================

@dataclass
class EDAConfig:
    """Konfiguracja analizy EDA."""
    sample_size: int
    correlation_method: CorrelationMethod
    top_k_columns: int
    minimal_profile: bool
    export_profile: bool


@dataclass
class DataQualityStats:
    """Statystyki jakości danych."""
    rows: int
    cols: int
    missing_pct: float
    dupes: int
    memory_mb: float
    numeric_cols: int
    categorical_cols: int
    datetime_cols: int


# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

def _validate_dataframe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Waliduje DataFrame z session state.
    
    Args:
        df: DataFrame do walidacji (może być None)
        
    Returns:
        Zwalidowany DataFrame
        
    Raises:
        ValueError: Jeśli DataFrame jest nieprawidłowy
    """
    if df is None:
        raise ValueError("Brak danych w session state")
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Oczekiwano DataFrame, otrzymano {type(df)}")
    
    if df.empty:
        raise ValueError("DataFrame jest pusty")
    
    return df


def _get_sample(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """
    Pobiera próbkę danych z walidacją.
    
    Args:
        df: DataFrame źródłowy
        sample_size: Rozmiar próbki (0 = pełny zbiór)
        
    Returns:
        DataFrame - próbka lub pełny zbiór
    """
    if sample_size <= 0 or sample_size >= len(df):
        return df
    
    # Walidacja rozmiaru próbki
    actual_sample = min(sample_size, MAX_SAMPLE_SIZE, len(df))
    
    if actual_sample < sample_size:
        logger.warning(
            f"Zmniejszono próbkę z {sample_size:,} do {actual_sample:,} "
            f"(limit: {MAX_SAMPLE_SIZE:,})"
        )
    
    return df.sample(n=actual_sample, random_state=42)


def _compute_quality_stats(df: pd.DataFrame) -> DataQualityStats:
    """
    Oblicza kompleksowe statystyki jakości danych.
    
    Args:
        df: DataFrame do analizy
        
    Returns:
        DataQualityStats z metrykami
    """
    # Podstawowe statystyki z walidatora
    base_stats = validate(df)
    
    # Pamięć
    memory_mb = round(df.memory_usage(deep=True).sum() / 1e6, 2)
    
    # Typy kolumn
    numeric_cols = len(df.select_dtypes(include=np.number).columns)
    categorical_cols = len(df.select_dtypes(include=["object", "category"]).columns)
    datetime_cols = len(df.select_dtypes(include=["datetime64", "timedelta64"]).columns)
    
    return DataQualityStats(
        rows=base_stats["rows"],
        cols=base_stats["cols"],
        missing_pct=base_stats["missing_pct"],
        dupes=base_stats["dupes"],
        memory_mb=memory_mb,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        datetime_cols=datetime_cols
    )


def _create_data_dictionary(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """
    Tworzy słownik danych z podsumowaniem kolumn.
    
    Args:
        df: DataFrame do analizy
        top_k: Liczba zwracanych kolumn (posortowanych)
        
    Returns:
        DataFrame ze słownikiem danych
    """
    rows = []
    
    for col in df.columns:
        series = df[col]
        
        # Podstawowe metryki
        missing_pct = float(series.isna().mean())
        nunique = int(series.nunique(dropna=True))
        dtype = str(series.dtype)
        
        # Przykładowa wartość
        non_null = series.dropna()
        sample_val = str(non_null.iloc[0])[:60] if len(non_null) > 0 else ""
        
        # Dodatkowe info dla numerycznych
        if pd.api.types.is_numeric_dtype(series):
            min_val = series.min()
            max_val = series.max()
            mean_val = series.mean()
            extra_info = f"min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}"
        else:
            extra_info = ""
        
        rows.append({
            "kolumna": col,
            "typ": dtype,
            "braki_%": round(missing_pct * 100, 2),
            "unikalne": nunique,
            "przykład": sample_val,
            "info": extra_info
        })
    
    # Sortuj po brakach i unikalnych
    dict_df = pd.DataFrame(rows)
    dict_df = dict_df.sort_values(
        ["braki_%", "unikalne"],
        ascending=[False, True]
    ).head(top_k)
    
    return dict_df


def _find_target_candidates(df: pd.DataFrame) -> list[str]:
    """
    Znajduje potencjalne kolumny docelowe na podstawie nazw.
    
    Args:
        df: DataFrame do przeszukania
        
    Returns:
        Lista nazw kolumn - kandydatów
    """
    candidates = []
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Sprawdź czy nazwa zawiera hint
        if any(hint in col_lower for hint in TARGET_NAME_HINTS):
            candidates.append(col)
    
    # Usuń duplikaty zachowując kolejność
    return list(dict.fromkeys(candidates))


def _compute_correlation(
    df: pd.DataFrame,
    method: CorrelationMethod
) -> Optional[pd.DataFrame]:
    """
    Oblicza macierz korelacji dla kolumn numerycznych.
    
    Args:
        df: DataFrame z danymi
        method: Metoda korelacji
        
    Returns:
        DataFrame z korelacją lub None jeśli brak kolumn numerycznych
    """
    numeric_df = df.select_dtypes(include=np.number)
    
    if numeric_df.shape[1] < 2:
        return None
    
    # Limit kolumn dla wydajności
    if numeric_df.shape[1] > MAX_CORRELATION_COLS:
        logger.warning(
            f"Za dużo kolumn numerycznych ({numeric_df.shape[1]}). "
            f"Używam pierwszych {MAX_CORRELATION_COLS}"
        )
        numeric_df = numeric_df.iloc[:, :MAX_CORRELATION_COLS]
    
    try:
        corr = numeric_df.corr(method=method)
        return corr
    except Exception as e:
        logger.error(f"Błąd obliczania korelacji: {e}", exc_info=True)
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def _generate_profile_html(
    df: pd.DataFrame,
    title: str,
    minimal: bool,
    df_hash: str  # dla unique cache key
) -> str:
    """
    Generuje raport HTML z profilowaniem (cachowane).
    
    Args:
        df: DataFrame do profilowania
        title: Tytuł raportu
        minimal: Czy tryb minimalny
        df_hash: Hash dla cache key
        
    Returns:
        HTML jako string
    """
    # Ogranicz próbkę dla wydajności
    if len(df) > MAX_PROFILE_SIZE:
        df_sample = df.sample(n=MAX_PROFILE_SIZE, random_state=42)
        logger.info(f"Profil: używam próbki {MAX_PROFILE_SIZE} z {len(df)}")
    else:
        df_sample = df
    
    return make_profile_html(df_sample, title=title)


def _save_profile_to_disk(html: str, filename: str = "profile.html") -> pathlib.Path:
    """
    Zapisuje profil HTML na dysk z walidacją ścieżki.
    
    Args:
        html: Zawartość HTML
        filename: Nazwa pliku
        
    Returns:
        Ścieżka do zapisanego pliku
    """
    # Bezpieczna ścieżka
    export_dir = pathlib.Path("data/exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = export_dir / filename
    
    # Walidacja ścieżki (prevent path traversal)
    if not output_path.resolve().is_relative_to(export_dir.resolve()):
        raise ValueError("Nieprawidłowa ścieżka zapisu")
    
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"Profil zapisany: {output_path}")
    
    return output_path


# ========================================================================================
# STREAMLIT UI
# ========================================================================================

st.title("🔍 EDA Analysis — PRO")

# ========================================================================================
# WALIDACJA DANYCH
# ========================================================================================

try:
    df_raw = st.session_state.get("df") or st.session_state.get("df_raw")
    df_main = _validate_dataframe(df_raw)
except ValueError as e:
    st.warning(f"⚠️ {e}")
    st.info("Przejdź do **📤 Upload Data**, aby wczytać dane.")
    st.stop()

# ========================================================================================
# SIDEBAR: KONFIGURACJA
# ========================================================================================

with st.sidebar:
    st.subheader("⚙️ Opcje EDA")
    
    sample_n = st.number_input(
        "Próbka (0 = pełny zbiór)",
        min_value=0,
        max_value=MAX_SAMPLE_SIZE,
        value=min(DEFAULT_SAMPLE, len(df_main)),
        step=500,
        help=f"Maksymalna próbka: {MAX_SAMPLE_SIZE:,} wierszy"
    )
    
    corr_method = st.selectbox(
        "Metoda korelacji",
        options=["pearson", "spearman", "kendall"],
        index=0,
        help="Pearson - liniowa, Spearman - monotoniczna, Kendall - rangowa"
    )
    
    top_k = st.slider(
        "Top kolumn (słownik)",
        min_value=5,
        max_value=50,
        value=20,
        help="Liczba kolumn w słowniku danych"
    )
    
    profile_minimal = st.checkbox(
        "Profil minimalny (szybszy)",
        value=True,
        help="Tryb minimalny pomija niektóre kosztowne analizy"
    )
    
    export_profile_btn = st.checkbox(
        "Przycisk pobrania profilu",
        value=True
    )

# Konfiguracja
config = EDAConfig(
    sample_size=sample_n,
    correlation_method=corr_method,
    top_k_columns=top_k,
    minimal_profile=profile_minimal,
    export_profile=export_profile_btn
)

# ========================================================================================
# PRÓBKOWANIE
# ========================================================================================

with st.spinner("🔄 Przygotowuję próbkę danych..."):
    df_view = _get_sample(df_main, config.sample_size)

if len(df_view) < len(df_main):
    st.info(
        f"ℹ️ Używam próbki: **{len(df_view):,}** z **{len(df_main):,}** wierszy "
        f"({len(df_view)/len(df_main)*100:.1f}%)"
    )

# ========================================================================================
# STATYSTYKI JAKOŚCI
# ========================================================================================

st.subheader("📊 KPI jakości danych")

try:
    with st.spinner("📈 Obliczam statystyki..."):
        stats = _compute_quality_stats(df_view)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Wiersze", f"{stats.rows:,}")
    col2.metric("Kolumny", f"{stats.cols:,}")
    col3.metric("Braki (%)", f"{stats.missing_pct * 100:.2f}%")
    col4.metric("Duplikaty", f"{stats.dupes:,}")
    
    # Dodatkowe info
    st.caption(
        f"💾 Pamięć: **{stats.memory_mb} MB** • "
        f"🔢 Numeryczne: **{stats.numeric_cols}** • "
        f"📝 Kategoryczne: **{stats.categorical_cols}** • "
        f"📅 Datetime: **{stats.datetime_cols}**"
    )
    
except Exception as e:
    st.error(f"❌ Błąd obliczania statystyk: {e}")
    logger.error(f"Błąd statystyk: {e}", exc_info=True)

# ========================================================================================
# SŁOWNIK DANYCH
# ========================================================================================

st.subheader("🧭 Słownik danych")

try:
    with st.spinner("📋 Tworzę słownik..."):
        data_dict = _create_data_dictionary(df_view, config.top_k_columns)
    
    st.dataframe(data_dict, use_container_width=True, height=400)
    
    # Eksport CSV
    csv_buffer = io.StringIO()
    data_dict.to_csv(csv_buffer, index=False)
    
    st.download_button(
        "⬇️ Pobierz słownik danych (CSV)",
        data=csv_buffer.getvalue(),
        file_name="data_dictionary.csv",
        mime="text/csv",
        use_container_width=True
    )
    
except Exception as e:
    st.error(f"❌ Błąd tworzenia słownika: {e}")
    logger.error(f"Błąd słownika: {e}", exc_info=True)

# ========================================================================================
# PODGLĄD DANYCH
# ========================================================================================

with st.expander("📄 Podgląd danych", expanded=False):
    preview_rows = st.slider(
        "Liczba wierszy do podglądu",
        min_value=5,
        max_value=min(200, len(df_view)),
        value=50
    )
    st.dataframe(
        df_view.head(preview_rows),
        use_container_width=True,
        height=400
    )

# ========================================================================================
# ANALIZA BRAKÓW
# ========================================================================================

st.subheader("🕳️ Brakujące wartości")

try:
    nulls_series = df_view.isna().mean()
    nulls_pct = nulls_series[nulls_series > 0].sort_values(ascending=False)
    
    if len(nulls_pct) == 0:
        st.success("✅ Brak braków w danych!")
    else:
        # DataFrame z brakami
        nulls_df = (nulls_pct * 100).round(2).to_frame(name="braki_%")
        nulls_df["liczba_braków"] = (nulls_series[nulls_pct.index] * len(df_view)).astype(int)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(nulls_df, use_container_width=True, height=300)
        
        with col2:
            # Wykres słupkowy
            fig_nulls = px.bar(
                nulls_df.reset_index(),
                x="index",
                y="braki_%",
                title="Braki wg kolumn",
                labels={"index": "Kolumna", "braki_%": "Braki (%)"}
            )
            fig_nulls.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_nulls, use_container_width=True)
        
except Exception as e:
    st.error(f"❌ Błąd analizy braków: {e}")
    logger.error(f"Błąd braków: {e}", exc_info=True)

# ========================================================================================
# KORELACJE
# ========================================================================================

st.subheader("🔗 Korelacje (numeryczne)")

try:
    with st.spinner(f"📊 Obliczam korelację ({config.correlation_method})..."):
        corr_matrix = _compute_correlation(df_view, config.correlation_method)
    
    if corr_matrix is None:
        st.info("ℹ️ Za mało kolumn numerycznych do wyliczenia korelacji (minimum: 2)")
    else:
        # Heatmapa
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title=f"Macierz korelacji ({config.correlation_method})"
        )
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Top korelacje
        with st.expander("🔝 Najsilniejsze korelacje", expanded=False):
            # Wyciągnij pary korelacji (bez diagonali)
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        "kolumna_1": corr_matrix.columns[i],
                        "kolumna_2": corr_matrix.columns[j],
                        "korelacja": corr_matrix.iloc[i, j]
                    })
            
            if corr_pairs:
                corr_df = pd.DataFrame(corr_pairs)
                corr_df = corr_df.sort_values("korelacja", key=abs, ascending=False).head(20)
                st.dataframe(corr_df, use_container_width=True)
        
except Exception as e:
    st.error(f"❌ Błąd obliczania korelacji: {e}")
    logger.error(f"Błąd korelacji: {e}", exc_info=True)

# ========================================================================================
# WIZUALIZACJE
# ========================================================================================

with st.expander("📈 Rozkłady i relacje", expanded=False):
    numeric_cols = list(df_view.select_dtypes(include=np.number).columns)
    
    if not numeric_cols:
        st.info("Brak kolumn numerycznych do wizualizacji")
    else:
        tab1, tab2 = st.tabs(["📊 Histogram", "🔍 Scatter Plot"])
        
        # HISTOGRAM
        with tab1:
            hist_col = st.selectbox(
                "Wybierz kolumnę",
                options=numeric_cols,
                key="hist_col"
            )
            
            nbins = st.slider(
                "Liczba bins",
                min_value=10,
                max_value=100,
                value=40,
                key="hist_bins"
            )
            
            try:
                fig_hist = px.histogram(
                    df_view,
                    x=hist_col,
                    nbins=nbins,
                    title=f"Rozkład: {hist_col}",
                    marginal="box"  # Dodaj boxplot
                )
                fig_hist.update_layout(height=500)
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Statystyki opisowe
                desc = df_view[hist_col].describe()
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Średnia", f"{desc['mean']:.2f}")
                col2.metric("Mediana", f"{desc['50%']:.2f}")
                col3.metric("Odch. std", f"{desc['std']:.2f}")
                col4.metric("Zakres", f"{desc['max'] - desc['min']:.2f}")
                
            except Exception as e:
                st.error(f"Błąd generowania histogramu: {e}")
        
        # SCATTER PLOT
        with tab2:
            if len(numeric_cols) < 2:
                st.info("Potrzebne minimum 2 kolumny numeryczne")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("Oś X", options=numeric_cols, key="scatter_x")
                with col2:
                    y_options = [c for c in numeric_cols if c != x_col]
                    y_col = st.selectbox("Oś Y", options=y_options, key="scatter_y")
                
                color_col = st.selectbox(
                    "Kolor (opcjonalnie)",
                    options=["Brak"] + list(df_view.columns),
                    key="scatter_color"
                )
                
                try:
                    fig_scatter = px.scatter(
                        df_view,
                        x=x_col,
                        y=y_col,
                        color=None if color_col == "Brak" else color_col,
                        trendline="ols",
                        title=f"Zależność: {x_col} vs {y_col}",
                        opacity=0.6
                    )
                    fig_scatter.update_layout(height=500)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Błąd generowania wykresu: {e}")

# ========================================================================================
# SUGESTIA CELU
# ========================================================================================

st.divider()
st.subheader("🎯 Sugestia celu / typu problemu")

try:
    target_candidates = _find_target_candidates(df_main)
    
    if target_candidates:
        st.info(f"💡 Znaleziono {len(target_candidates)} kandydatów na cel")
    
    target_col = st.selectbox(
        "Wybierz kolumnę celu (opcjonalnie)",
        options=["—"] + target_candidates + [c for c in df_main.columns if c not in target_candidates],
        help="Wybierz zmienną którą chcesz przewidywać"
    )
    
    if target_col != "—":
        st.session_state["target"] = target_col
        
        with st.spinner("🔍 Analizuję typ problemu..."):
            try:
                problem_type = infer_problem_type(df_main, target_col)
                
                if problem_type == "classification":
                    nunique = df_main[target_col].nunique()
                    st.success(f"✅ Wykryto **klasyfikację** ({nunique} klas)")
                    
                    # Rozkład klas
                    class_dist = df_main[target_col].value_counts()
                    fig_classes = px.bar(
                        x=class_dist.index.astype(str),
                        y=class_dist.values,
                        title="Rozkład klas",
                        labels={"x": "Klasa", "y": "Liczba"}
                    )
                    st.plotly_chart(fig_classes, use_container_width=True)
                    
                elif problem_type == "regression":
                    st.success("✅ Wykryto **regresję** (wartości ciągłe)")
                    
                    # Statystyki
                    if pd.api.types.is_numeric_dtype(df_main[target_col]):
                        desc = df_main[target_col].describe()
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Min", f"{desc['min']:.2f}")
                        col2.metric("Średnia", f"{desc['mean']:.2f}")
                        col3.metric("Max", f"{desc['max']:.2f}")
                    
                elif problem_type == "timeseries":
                    st.success("✅ Wykryto **szereg czasowy**")
                    
                else:
                    st.warning("⚠️ Nie udało się jednoznacznie określić typu problemu")
                    
            except Exception as e:
                st.error(f"Błąd wykrywania typu problemu: {e}")
                logger.error(f"Błąd typu problemu: {e}", exc_info=True)
    else:
        st.caption(
            "💡 Nie wybrano celu — możesz to zrobić później w zakładce "
            "**Predictions** lub **Forecasting**"
        )
        
except Exception as e:
    st.error(f"❌ Błąd sugestii celu: {e}")
    logger.error(f"Błąd sugestii: {e}", exc_info=True)

# ========================================================================================
# RAPORT PROFILUJĄCY
# ========================================================================================

st.divider()
st.subheader("🧪 Raport profilujący (HTML)")

with st.expander("📄 Generuj raport profilujący", expanded=False):
    st.info(
        f"ℹ️ Raport zostanie wygenerowany dla próbki do "
        f"**{MAX_PROFILE_SIZE:,}** wierszy"
    )
    
    if st.button("🚀 Generuj raport", use_container_width=True):
        try:
            start_time = time.time()
            
            # Hash dla cache
            df_hash = str(hash(tuple(df_view.columns)) + len(df_view))
            
            with st.spinner("🔄 Generuję raport (może potrwać chwilę)..."):
                html_report = _generate_profile_html(
                    df_view,
                    title="Data Profile Report",
                    minimal=config.minimal_profile,
                    df_hash=df_hash
                )
            
            elapsed = time.time() - start_time
            
            # Wyświetl raport
            st.components.v1.html(html_report, height=600, scrolling=True)
            
            st.success(f"✅ Raport wygenerowany w {elapsed:.2f}s")
            
            # Przyciski akcji
            col1, col2 = st.columns(2)
            
            if config.export_profile:
                with col1:
                    st.download_button(
                        "⬇️ Pobierz raport (HTML)",
                        data=html_report,
                        file_name="eda_profile.html",
                        mime="text/html",
                        use_container_width=True
                    )
                
                with col2:
                    if st.button("💾 Zapisz na dysk", use_container_width=True):
                        try:
                            saved_path = _save_profile_to_disk(html_report)
                            st.success(f"✅ Zapisano: `{saved_path}`")
                        except Exception as e:
                            st.error(f"Błąd zapisu: {e}")
            
        except Exception as e:
            st.error(f"❌ Błąd generowania raportu: {e}")
            logger.error(f"Błąd raportu: {e}", exc_info=True)
            st.info("💡 Spróbuj zmniejszyć rozmiar próbki lub użyć trybu minimalnego")

# ========================================================================================
# NAWIGACJA
# ========================================================================================

st.divider()
st.success(
    "✨ **Analiza EDA zakończona!** Co dalej?\n\n"
    "- **🤖 AI Insights** — zaawansowana analiza AI\n"
    "- **🎯 Predictions** — trenowanie modeli\n"
    "- **📈 Forecasting** — prognozy szeregów czasowych"
)