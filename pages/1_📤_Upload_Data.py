"""
Moduł Upload Data - Inteligentny ingest plików z zaawansowanym przetwarzaniem.

Obsługuje:
- Wiele formatów (CSV/XLSX/JSON/DOCX/PDF)
- Inteligentne łączenie DataFrame'ów
- Automatyczne czyszczenie i feature engineering
- Ekstrakcję tekstu z dokumentów
- Audyt i manifest wczytania
"""

from __future__ import annotations

import io
import time
import hashlib
import logging
from typing import Optional, Literal
from dataclasses import dataclass

import streamlit as st
import pandas as pd

from src.data_processing.file_parser import parse_any
from src.data_processing.data_cleaner import quick_clean
from src.data_processing.feature_engineering import basic_feature_engineering
from src.utils.validators import basic_quality_checks
from src.ai_engine.nlp_analyzer import summarize_text

# ========================================================================================
# KONFIGURACJA
# ========================================================================================

logger = logging.getLogger(__name__)

# Limity bezpieczeństwa
MAX_FILE_SIZE_MB = 500
MAX_TOTAL_ROWS = 10_000_000
MAX_MEMORY_MB = 2048

# Typy łączenia
MergeMode = Literal["union", "intersection"]


# ========================================================================================
# DATACLASSES
# ========================================================================================

@dataclass
class FileManifest:
    """Metadane wczytanego pliku."""
    file: str
    id: str
    rows: Optional[int] = None
    cols: Optional[int] = None
    chars: Optional[int] = None
    type: Literal["table", "text"] = "table"
    error: Optional[str] = None


@dataclass
class DataStats:
    """Statystyki DataFrame."""
    rows: int
    cols: int
    missing_pct: float
    dupes: int
    memory_mb: float


# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

def _hash_bytes(data: bytes) -> str:
    """
    Generuje krótki hash SHA-256 z danych binarnych.
    
    Args:
        data: Dane binarne do zahashowania
        
    Returns:
        12-znakowy hash hex
    """
    return hashlib.sha256(data).hexdigest()[:12]


def _validate_file_size(data: bytes, filename: str) -> None:
    """
    Waliduje rozmiar pliku.
    
    Args:
        data: Dane binarne pliku
        filename: Nazwa pliku (dla logowania)
        
    Raises:
        ValueError: Jeśli plik przekracza limit
    """
    size_mb = len(data) / 1e6
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"Plik {filename} jest za duży ({size_mb:.1f} MB). "
            f"Maksymalny rozmiar: {MAX_FILE_SIZE_MB} MB"
        )


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_parse(name: str, data: bytes, file_hash: str) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Cachowane parsowanie pliku z TTL.
    
    Args:
        name: Nazwa pliku
        data: Dane binarne
        file_hash: Hash do klucza cache
        
    Returns:
        Tuple (DataFrame lub None, tekst lub None)
    """
    try:
        return parse_any(name, data)
    except Exception as e:
        logger.error(f"Błąd parsowania {name}: {e}", exc_info=True)
        raise


def _concat_frames(frames: list[pd.DataFrame], mode: str) -> pd.DataFrame:
    """
    Łączy wiele DataFrame'ów zgodnie z wybranym trybem.
    
    Args:
        frames: Lista DataFrame'ów do połączenia
        mode: "union" lub "intersection"
        
    Returns:
        Połączony DataFrame
        
    Raises:
        ValueError: Jeśli brak ramek lub nieprawidłowy tryb
    """
    if not frames:
        raise ValueError("Brak ramek danych do połączenia")
    
    if len(frames) == 1:
        return frames[0].copy()
    
    if mode.startswith("union"):
        # Union: zachowaj wszystkie kolumny, wypełnij NaN
        return pd.concat(frames, axis=0, ignore_index=True, sort=True)
    
    elif mode.startswith("intersection"):
        # Intersection: tylko wspólne kolumny
        common_cols = set(frames[0].columns)
        for df in frames[1:]:
            common_cols &= set(df.columns)
        
        if not common_cols:
            raise ValueError("Brak wspólnych kolumn między plikami!")
        
        common_list = sorted(common_cols)
        aligned = [df[common_list].copy() for df in frames]
        return pd.concat(aligned, axis=0, ignore_index=True)
    
    else:
        raise ValueError(f"Nieznany tryb łączenia: {mode}")


def _apply_cleaning(
    df: pd.DataFrame,
    fill_strategy: str,
    encode_categories: bool
) -> pd.DataFrame:
    """
    Aplikuje czyszczenie i feature engineering.
    
    Args:
        df: DataFrame do wyczyszczenia
        fill_strategy: Strategia wypełniania ("median", "mean", "none")
        encode_categories: Czy kodować kategorie niskopoziomowe
        
    Returns:
        Wyczyszczony DataFrame
    """
    # Podstawowe czyszczenie
    df_clean = quick_clean(df)
    
    # Wypełnianie braków numerycznych
    if fill_strategy != "none":
        numeric_cols = df_clean.select_dtypes(include="number").columns
        for col in numeric_cols:
            if df_clean[col].isna().any():
                if fill_strategy == "median":
                    fill_value = df_clean[col].median()
                elif fill_strategy == "mean":
                    fill_value = df_clean[col].mean()
                else:
                    continue
                
                df_clean[col] = df_clean[col].fillna(fill_value)
    
    # Feature engineering
    if encode_categories:
        df_clean = basic_feature_engineering(df_clean)
    
    return df_clean


def _compute_stats(df: pd.DataFrame) -> DataStats:
    """
    Oblicza kompleksowe statystyki DataFrame.
    
    Args:
        df: DataFrame do analizy
        
    Returns:
        DataStats z metrykami
    """
    quality = basic_quality_checks(df)
    memory_mb = round(df.memory_usage(deep=True).sum() / 1e6, 2)
    
    return DataStats(
        rows=quality["rows"],
        cols=quality["cols"],
        missing_pct=quality["missing_pct"],
        dupes=quality["dupes"],
        memory_mb=memory_mb
    )


def _safe_session_set(key: str, value, max_memory_mb: float = MAX_MEMORY_MB) -> None:
    """
    Bezpiecznie zapisuje do session_state z limitem pamięci.
    
    Args:
        key: Klucz session state
        value: Wartość do zapisania
        max_memory_mb: Maksymalne zużycie pamięci
    """
    if isinstance(value, pd.DataFrame):
        memory_mb = value.memory_usage(deep=True).sum() / 1e6
        if memory_mb > max_memory_mb:
            st.warning(
                f"⚠️ DataFrame zbyt duży ({memory_mb:.1f} MB). "
                f"Rozważ przefiltrowanie danych."
            )
    
    st.session_state[key] = value


# ========================================================================================
# STREAMLIT UI
# ========================================================================================

st.title("📤 Upload Data — Inteligentny Ingest")

# ========================================================================================
# SIDEBAR: OPCJE
# ========================================================================================

with st.sidebar:
    st.subheader("⚙️ Opcje wczytywania")
    
    merge_mode = st.radio(
        "Łączenie wielu plików",
        options=["union (po nazwach kolumn)", "intersection (wspólne kolumny)"],
        index=0,
        help="Union zachowuje wszystkie kolumny (puste = NaN). Intersection tylko wspólne."
    )
    
    sample_preview = st.slider(
        "Podgląd pierwszych N wierszy",
        min_value=5,
        max_value=200,
        value=20,
        step=5
    )
    
    limit_rows = st.number_input(
        "Limit wierszy (0 = bez limitu)",
        min_value=0,
        value=0,
        step=1000,
        help="Dla dużych plików — przyspiesza wczytanie/analizę."
    )
    
    st.markdown("---")
    st.subheader("🧹 Opcje czyszczenia")
    
    fill_missing_numeric = st.selectbox(
        "Wypełnianie braków (numeryczne)",
        options=["median", "mean", "none"],
        index=0
    )
    
    encode_low_cardinality = st.checkbox(
        "Koduj kategorie (≤ 20 unikalnych)",
        value=True
    )

# ========================================================================================
# UPLOAD FILES
# ========================================================================================

files = st.file_uploader(
    "Wybierz pliki (CSV/XLSX/JSON/DOCX/PDF)",
    type=["csv", "xlsx", "json", "docx", "pdf", "doc"],
    accept_multiple_files=True,
    help="Maksymalny rozmiar pojedynczego pliku: 500 MB"
)

goal = st.text_input(
    "🎯 Cel biznesowy / pytanie analityczne",
    placeholder="np. prognoza sprzedaży Q4, optymalizacja zużycia energii",
    help="Określ kontekst biznesowy dla lepszej analizy AI"
)

# Zapisz cel do session state
_safe_session_set("goal", goal)

# ========================================================================================
# MAIN PROCESSING
# ========================================================================================

dataframes: list[pd.DataFrame] = []
texts: list[dict] = []
manifest: list[FileManifest] = []

if files:
    # Walidacja liczby plików
    if len(files) > 50:
        st.error("❌ Zbyt wiele plików naraz (max 50). Podziel na mniejsze partie.")
        st.stop()
    
    with st.status("🔎 Przetwarzam pliki...", expanded=True) as status_container:
        progress_bar = st.progress(0)
        total_files = len(files)
        
        for idx, file in enumerate(files):
            # Progress
            progress_pct = (idx + 1) / total_files
            progress_bar.progress(progress_pct)
            
            st.write(f"**{idx+1}/{total_files}** • {file.name}")
            
            # Odczyt danych
            try:
                file_data = file.read()
                file_id = _hash_bytes(file_data)
                
                # Walidacja rozmiaru
                _validate_file_size(file_data, file.name)
                
            except ValueError as ve:
                st.error(f"❌ {ve}")
                manifest.append(FileManifest(
                    file=file.name,
                    id="",
                    error=str(ve),
                    type="table"
                ))
                continue
            except Exception as e:
                st.error(f"❌ Błąd odczytu {file.name}: {e}")
                manifest.append(FileManifest(
                    file=file.name,
                    id="",
                    error=f"Błąd odczytu: {e}",
                    type="table"
                ))
                continue
            
            # Ostrzeżenie dla legacy .doc
            if file.name.lower().endswith(".doc"):
                st.warning(
                    f"⚠️ {file.name}: format .doc jest przestarzały. "
                    "Konwertuj do .docx dla lepszej jakości."
                )
            
            # Parsowanie
            try:
                df, text = _cached_parse(file.name, file_data, file_id)
                
            except Exception as e:
                logger.error(f"Błąd parsowania {file.name}: {e}", exc_info=True)
                st.error(f"❌ Błąd parsowania {file.name}: {e}")
                manifest.append(FileManifest(
                    file=file.name,
                    id=file_id,
                    error=f"Błąd parsowania: {e}",
                    type="table"
                ))
                continue
            
            # Przetwarzanie DataFrame
            if isinstance(df, pd.DataFrame) and not df.empty:
                # Limit wierszy jeśli ustawiony
                if limit_rows > 0 and len(df) > limit_rows:
                    st.info(f"ℹ️ Ograniczam do {limit_rows:,} wierszy (z {len(df):,})")
                    df = df.head(limit_rows).copy()
                
                dataframes.append(df)
                manifest.append(FileManifest(
                    file=file.name,
                    id=file_id,
                    rows=len(df),
                    cols=df.shape[1],
                    type="table"
                ))
            
            # Przetwarzanie tekstu
            if text:
                text_preview = text[:2000] + ("..." if len(text) > 2000 else "")
                
                # Podsumowanie AI (z timeout)
                try:
                    summary = summarize_text(text_preview)
                except Exception as e:
                    logger.warning(f"Błąd podsumowania {file.name}: {e}")
                    summary = "[Nie udało się wygenerować podsumowania]"
                
                texts.append({
                    "file": file.name,
                    "id": file_id,
                    "text": text_preview,
                    "summary": summary
                })
                
                manifest.append(FileManifest(
                    file=file.name,
                    id=file_id,
                    chars=len(text),
                    type="text"
                ))
        
        status_container.update(
            label=f"✅ Zakończono wczytywanie ({total_files} plików)",
            state="complete"
        )

# ========================================================================================
# WYŚWIETLANIE WYNIKÓW
# ========================================================================================

if dataframes:
    st.divider()
    
    # Łączenie DataFrame'ów
    with st.spinner("🔗 Łączę ramki danych..."):
        try:
            merged_df = _concat_frames(dataframes, merge_mode)
            
            # Walidacja rozmiaru wyniku
            if len(merged_df) > MAX_TOTAL_ROWS:
                st.error(
                    f"❌ Połączone dane mają {len(merged_df):,} wierszy, "
                    f"co przekracza limit {MAX_TOTAL_ROWS:,}. Zastosuj filtry."
                )
                st.stop()
            
            _safe_session_set("df_raw", merged_df)
            
        except Exception as e:
            st.error(f"❌ Błąd łączenia danych: {e}")
            logger.error(f"Błąd łączenia: {e}", exc_info=True)
            st.stop()
    
    # Statystyki
    st.subheader("📊 Podgląd i metryki")
    
    try:
        stats = _compute_stats(merged_df)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Wiersze", f"{stats.rows:,}")
        col2.metric("Kolumny", f"{stats.cols:,}")
        col3.metric("Braki (%)", f"{stats.missing_pct * 100:.2f}%")
        col4.metric("Duplikaty", f"{stats.dupes:,}")
        
        st.caption(
            f"💾 Pamięć: **{stats.memory_mb} MB** • "
            f"🔗 Tryb: **{merge_mode.split()[0]}**"
        )
        
    except Exception as e:
        st.warning(f"⚠️ Nie udało się obliczyć statystyk: {e}")
    
    # Podgląd danych
    st.subheader("📄 Podgląd danych")
    st.dataframe(
        merged_df.head(sample_preview),
        use_container_width=True,
        height=400
    )
    
    # Przycisk czyszczenia
    st.divider()
    clean_button = st.button(
        "🧹 Szybkie czyszczenie + Feature Engineering",
        type="primary",
        use_container_width=True,
        help="Usuwa braki, outliers, koduje kategorie i generuje nowe cechy"
    )
    
    if clean_button:
        start_time = time.time()
        
        with st.spinner("🔧 Czyszczę i wzbogacam cechy..."):
            try:
                cleaned_df = _apply_cleaning(
                    merged_df,
                    fill_missing_numeric,
                    encode_low_cardinality
                )
                
                _safe_session_set("df", cleaned_df)
                
                elapsed = time.time() - start_time
                
                st.success(
                    f"✅ Dane wyczyszczone i wzbogacone! "
                    f"**{len(cleaned_df):,} × {cleaned_df.shape[1]}** kolumn "
                    f"⏱️ {elapsed:.2f}s"
                )
                
                # Nowe statystyki
                st.subheader("📊 Po czyszczeniu")
                new_stats = _compute_stats(cleaned_df)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Zmiana w brakach",
                        f"{new_stats.missing_pct * 100:.2f}%",
                        delta=f"{(new_stats.missing_pct - stats.missing_pct) * 100:.2f}%",
                        delta_color="inverse"
                    )
                with col2:
                    st.metric(
                        "Zmiana w kolumnach",
                        new_stats.cols,
                        delta=new_stats.cols - stats.cols
                    )
                
                st.dataframe(
                    cleaned_df.head(sample_preview),
                    use_container_width=True,
                    height=400
                )
                
            except Exception as e:
                st.error(f"❌ Błąd czyszczenia danych: {e}")
                logger.error(f"Błąd czyszczenia: {e}", exc_info=True)

# ========================================================================================
# TEKSTY Z DOKUMENTÓW
# ========================================================================================

if texts:
    st.divider()
    st.subheader("📄 Wydobyty tekst (DOCX/PDF)")
    
    for text_info in texts:
        with st.expander(f"📄 {text_info['file']}", expanded=False):
            if text_info.get("summary"):
                st.markdown(f"**🤖 Podsumowanie AI:**")
                st.info(text_info["summary"])
            
            st.markdown("**📝 Fragment tekstu:**")
            st.text_area(
                "Tekst",
                text_info["text"],
                height=200,
                key=f"text_{text_info['id']}",
                label_visibility="collapsed"
            )

# ========================================================================================
# MANIFEST I NAWIGACJA
# ========================================================================================

if files:
    st.divider()
    
    # Podsumowanie wczytania
    successful = sum(1 for m in manifest if not m.error)
    failed = sum(1 for m in manifest if m.error)
    
    col1, col2 = st.columns(2)
    col1.metric("✅ Sukces", successful)
    col2.metric("❌ Błędy", failed)
    
    # Manifest
    with st.expander("🔍 Manifest wczytania (audyt)", expanded=False):
        manifest_data = [
            {
                "Plik": m.file,
                "ID": m.id,
                "Typ": m.type,
                "Wiersze": m.rows or "-",
                "Kolumny": m.cols or "-",
                "Znaki": m.chars or "-",
                "Status": "❌ Błąd" if m.error else "✅ OK",
                "Błąd": m.error or "-"
            }
            for m in manifest
        ]
        st.dataframe(manifest_data, use_container_width=True)

# ========================================================================================
# WSKAZÓWKI NAWIGACJI
# ========================================================================================

if dataframes:
    st.success(
        "✨ **Dane gotowe!** Przejdź teraz do:\n"
        "- **📊 EDA Analysis** — eksploracja danych\n"
        "- **🤖 AI Insights** — analiza AI\n"
        "- **🎯 Model Training** — trenowanie modeli"
    )
else:
    st.info(
        "👆 Wczytaj pliki, aby rozpocząć analizę. "
        "Obsługiwane formaty: CSV, Excel, JSON, DOCX, PDF"
    )