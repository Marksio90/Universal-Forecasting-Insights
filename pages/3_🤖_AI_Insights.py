"""
Moduł AI Insights PRO - Zaawansowana analiza danych z wykorzystaniem AI.

Funkcjonalności:
- Generowanie insightów z OpenAI/Claude
- Inteligentne budowanie promptów
- Retry logic z exponential backoff
- Streaming responses dla UX
- Historia wygenerowanych insightów
- Multi-format export (JSON/MD/TXT)
- Cache z bezpiecznym hashowaniem
"""

from __future__ import annotations

import io
import time
import json
import hashlib
import logging
from typing import Optional, Literal
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

import streamlit as st
import pandas as pd

from src.ai_engine.insights_generator import generate_insights
from src.utils.validators import basic_quality_checks

# ========================================================================================
# KONFIGURACJA
# ========================================================================================

logger = logging.getLogger(__name__)

# Limity bezpieczeństwa
MAX_PROMPT_TOKENS = 8000  # ~6000 słów
MAX_SCHEMA_COLS = 50
MAX_RETRIES = 3
RETRY_DELAY = 2  # sekundy

# Domyślna struktura odpowiedzi
DEFAULT_RESPONSE_KEYS = ["summary", "top_insights", "recommendations", "risks"]


# ========================================================================================
# ENUMS & DATACLASSES
# ========================================================================================

class AnalysisDepth(str, Enum):
    """Tryb głębokości analizy."""
    QUICK = "quick"
    DEEP = "deep"


@dataclass
class InsightsConfig:
    """Konfiguracja generowania insightów."""
    depth: AnalysisDepth
    include_schema: bool
    include_quality: bool
    show_json: bool


@dataclass
class DataContext:
    """Kontekst danych dla AI."""
    rows: int
    cols: int
    missing_pct: float
    dupes: int
    schema: list[dict]
    goal: str
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class AIInsight:
    """Pojedynczy wygenerowany insight."""
    summary: str
    top_insights: list[str]
    recommendations: list[str]
    risks: list[str]
    metadata: dict
    timestamp: str
    
    def to_dict(self) -> dict:
        """Konwertuje do słownika."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Konwertuje do JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    def to_markdown(self) -> str:
        """Konwertuje do Markdown."""
        lines = [
            f"# AI Insights Report",
            f"\n**Generated:** {self.timestamp}\n",
            f"## 🧭 Executive Summary\n",
            self.summary,
            f"\n## 📌 Top Insights\n"
        ]
        
        for i, insight in enumerate(self.top_insights, 1):
            lines.append(f"{i}. {insight}")
        
        lines.append(f"\n## 💡 Recommendations\n")
        for rec in self.recommendations:
            lines.append(f"- {rec}")
        
        lines.append(f"\n## ⚠️ Risks & Anomalies\n")
        for risk in self.risks:
            lines.append(f"- {risk}")
        
        return "\n".join(lines)


# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

def _validate_dataframe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Waliduje DataFrame z session state.
    
    Args:
        df: DataFrame do walidacji
        
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


def _extract_data_context(df: pd.DataFrame, goal: str) -> DataContext:
    """
    Ekstrahuje kontekst danych dla AI.
    
    Args:
        df: DataFrame do analizy
        goal: Cel biznesowy
        
    Returns:
        DataContext z metadanymi
    """
    # Statystyki jakości
    stats = basic_quality_checks(df)
    
    # Schema (ograniczony do MAX_SCHEMA_COLS)
    schema = []
    for col in df.columns[:MAX_SCHEMA_COLS]:
        try:
            nunique = int(df[col].nunique(dropna=True))
            dtype = str(df[col].dtype)
            
            schema.append({
                "col": col,
                "dtype": dtype,
                "nunique": nunique,
                "missing": float(df[col].isna().mean())
            })
        except Exception as e:
            logger.warning(f"Błąd przetwarzania kolumny {col}: {e}")
            continue
    
    return DataContext(
        rows=stats["rows"],
        cols=stats["cols"],
        missing_pct=round(stats["missing_pct"] * 100, 2),
        dupes=stats["dupes"],
        schema=schema,
        goal=goal or "Nie podano celu biznesowego"
    )


def _build_prompt(context: DataContext, config: InsightsConfig) -> str:
    """
    Buduje prompt dla AI z walidacją rozmiaru.
    
    Args:
        context: Kontekst danych
        config: Konfiguracja analizy
        
    Returns:
        Zbudowany prompt
    """
    parts = []
    
    # Cel
    parts.append(f"**Cel analizy:** {context.goal}")
    
    # Podstawowe metryki
    parts.append(
        f"**Rozmiar danych:** {context.rows:,} wierszy × {context.cols} kolumn\n"
        f"**Braki:** {context.missing_pct}%\n"
        f"**Duplikaty:** {context.dupes:,}"
    )
    
    # Schema
    if config.include_schema and context.schema:
        parts.append("\n**Struktura danych:**")
        schema_lines = []
        for s in context.schema[:25]:  # Limit dla promptu
            schema_lines.append(
                f"- `{s['col']}` ({s['dtype']}, "
                f"{s['nunique']} unikalnych, "
                f"{s['missing']*100:.1f}% braków)"
            )
        
        if len(context.schema) > 25:
            schema_lines.append(f"... i {len(context.schema) - 25} innych kolumn")
        
        parts.append("\n".join(schema_lines))
    
    # Metryki jakości
    if config.include_quality:
        quality_metrics = {
            "rows": context.rows,
            "cols": context.cols,
            "missing_pct": context.missing_pct,
            "dupes": context.dupes
        }
        parts.append(
            "\n**Metryki jakości:**\n```json\n" +
            json.dumps(quality_metrics, indent=2) +
            "\n```"
        )
    
    # Instrukcja dla AI
    parts.append(
        "\n**Zadanie:**\n"
        "Wygeneruj syntetyczny raport analityczny w formacie JSON z kluczami:\n"
        "- `summary`: zwięzłe podsumowanie (2-3 zdania)\n"
        "- `top_insights`: lista 5-7 kluczowych obserwacji\n"
        "- `recommendations`: lista 3-5 rekomendacji akcji\n"
        "- `risks`: lista 2-4 potencjalnych ryzyk/anomalii"
    )
    
    # Dodatkowe instrukcje dla trybu głębokiego
    if config.depth == AnalysisDepth.DEEP:
        parts.append(
            "\n**Tryb pogłębiony:**\n"
            "- Dodaj więcej szczegółów i kontekstu\n"
            "- Zaproponuj konkretne KPI do monitorowania\n"
            "- Przedstaw hipotezy do testowania\n"
            "- Uwzględnij perspektywę biznesową"
        )
    
    prompt = "\n\n".join(parts)
    
    # Walidacja rozmiaru (przybliżona)
    estimated_tokens = len(prompt.split())
    if estimated_tokens > MAX_PROMPT_TOKENS:
        logger.warning(
            f"Prompt przekracza limit ({estimated_tokens} > {MAX_PROMPT_TOKENS})"
        )
        # Skróć schema
        if config.include_schema:
            logger.info("Skracam schema w prompcie")
            # Rebuild z mniejszym schema
            config.include_schema = False
            return _build_prompt(context, config)
    
    return prompt


def _create_cache_key(df: pd.DataFrame, prompt: str, config: InsightsConfig) -> str:
    """
    Tworzy bezpieczny klucz cache dla insightów.
    
    Args:
        df: DataFrame
        prompt: Prompt dla AI
        config: Konfiguracja
        
    Returns:
        SHA-256 hash jako klucz
    """
    # Komponenty klucza
    components = [
        str(len(df)),
        str(df.shape[1]),
        ",".join(sorted(df.columns)),
        prompt,
        config.depth.value,
        str(config.include_schema),
        str(config.include_quality)
    ]
    
    # Secure hash
    key_str = "|".join(components)
    return hashlib.sha256(key_str.encode()).hexdigest()


def _parse_ai_response(response: str) -> dict:
    """
    Parsuje odpowiedź AI do struktury JSON.
    
    Args:
        response: Surowa odpowiedź z AI
        
    Returns:
        Słownik z sparsowanymi danymi
    """
    # Próba bezpośredniego parsowania JSON
    try:
        data = json.loads(response)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    
    # Próba wyekstrahowania JSON z markdown code block
    try:
        if "```json" in response:
            json_start = response.find("```json") + 7
            json_end = response.find("```", json_start)
            json_str = response[json_start:json_end].strip()
            data = json.loads(json_str)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    
    # Fallback: całość jako summary
    return {
        "summary": response,
        "top_insights": [],
        "recommendations": [],
        "risks": []
    }


def _normalize_insights(data: dict) -> AIInsight:
    """
    Normalizuje dane z AI do struktury AIInsight.
    
    Args:
        data: Surowe dane z AI
        
    Returns:
        Znormalizowany AIInsight
    """
    # Wyciągnij wartości z fallbackami
    summary = data.get("summary", "Brak podsumowania")
    
    # Ensure lists
    top_insights = data.get("top_insights", [])
    if isinstance(top_insights, str):
        top_insights = [top_insights]
    
    recommendations = data.get("recommendations", [])
    if isinstance(recommendations, str):
        recommendations = [recommendations]
    
    risks = data.get("risks", [])
    if isinstance(risks, str):
        risks = [risks]
    
    # Metadata
    metadata = {
        "response_keys": list(data.keys()),
        "processed_at": datetime.now().isoformat()
    }
    
    return AIInsight(
        summary=summary,
        top_insights=top_insights,
        recommendations=recommendations,
        risks=risks,
        metadata=metadata,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_generate_insights(
    cache_key: str,
    df_sample: pd.DataFrame,
    prompt: str
) -> dict:
    """
    Cachowane generowanie insightów z retry logic.
    
    Args:
        cache_key: Klucz cache (dla unikalności)
        df_sample: Próbka DataFrame
        prompt: Prompt dla AI
        
    Returns:
        Słownik z wygenerowanymi insightami
    """
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Generowanie insightów (próba {attempt + 1}/{MAX_RETRIES})")
            
            # Wywołanie AI
            response = generate_insights(df_sample, prompt)
            
            # Parsowanie
            parsed = _parse_ai_response(response)
            
            return parsed
            
        except Exception as e:
            logger.error(f"Błąd generowania insightów (próba {attempt + 1}): {e}")
            
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retry za {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Ostatnia próba failed - zwróć error
                return {
                    "summary": f"❌ Błąd generowania insightów po {MAX_RETRIES} próbach: {str(e)}",
                    "top_insights": [],
                    "recommendations": ["Sprawdź połączenie z API", "Spróbuj ponownie za chwilę"],
                    "risks": ["Brak dostępu do AI insights"]
                }
    
    # Fallback (nie powinno się zdarzyć)
    return {
        "summary": "Brak wyników",
        "top_insights": [],
        "recommendations": [],
        "risks": []
    }


def _add_to_history(insight: AIInsight) -> None:
    """
    Dodaje insight do historii w session state.
    
    Args:
        insight: AIInsight do zapisania
    """
    if "insights_history" not in st.session_state:
        st.session_state["insights_history"] = []
    
    # Dodaj na początek (newest first)
    st.session_state["insights_history"].insert(0, insight.to_dict())
    
    # Ogranicz historię do 10 ostatnich
    st.session_state["insights_history"] = st.session_state["insights_history"][:10]


# ========================================================================================
# STREAMLIT UI
# ========================================================================================

st.title("🤖 AI Insights — PRO")

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

# Cel biznesowy
goal = st.session_state.get("goal", "")

# ========================================================================================
# SIDEBAR: KONFIGURACJA
# ========================================================================================

with st.sidebar:
    st.subheader("⚙️ Ustawienia analizy AI")
    
    depth_option = st.radio(
        "Tryb analizy",
        options=[
            "🚀 Szybka (krótkie wnioski)",
            "🔬 Pogłębiona (pełny raport)"
        ],
        index=1,
        help="Szybka: ~30s, Pogłębiona: ~60s"
    )
    
    depth = AnalysisDepth.DEEP if "Pogłębiona" in depth_option else AnalysisDepth.QUICK
    
    include_schema = st.checkbox(
        "📋 Uwzględnij strukturę kolumn",
        value=True,
        help="Wysyła schema danych do AI"
    )
    
    include_quality = st.checkbox(
        "📊 Uwzględnij metryki jakości",
        value=True,
        help="Dodaje statystyki braków, duplikatów itp."
    )
    
    st.divider()
    
    show_json = st.checkbox(
        "🔍 Pokaż surowy JSON",
        value=False,
        help="Debug: wyświetl surową odpowiedź AI"
    )
    
    st.divider()
    
    # Historia
    history_count = len(st.session_state.get("insights_history", []))
    st.caption(f"📚 Historia: {history_count} raportów")

# Konfiguracja
config = InsightsConfig(
    depth=depth,
    include_schema=include_schema,
    include_quality=include_quality,
    show_json=show_json
)

# ========================================================================================
# KONTEKST I PROMPT
# ========================================================================================

st.subheader("📝 Kontekst analizy")

# Wyświetl cel
if goal:
    st.info(f"**🎯 Cel:** {goal}")
else:
    st.warning("⚠️ Brak celu biznesowego. Uzupełnij w zakładce Upload.")

# Ekstrahuj kontekst
with st.spinner("📊 Przygotowuję kontekst danych..."):
    try:
        context = _extract_data_context(df_main, goal)
        
        # Pokaż podstawowe info
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Wiersze", f"{context.rows:,}")
        col2.metric("Kolumny", f"{context.cols}")
        col3.metric("Braki", f"{context.missing_pct}%")
        col4.metric("Duplikaty", f"{context.dupes:,}")
        
    except Exception as e:
        st.error(f"❌ Błąd przygotowania kontekstu: {e}")
        logger.error(f"Błąd kontekstu: {e}", exc_info=True)
        st.stop()

# Zbuduj prompt
try:
    prompt = _build_prompt(context, config)
    
    with st.expander("👁️ Podgląd promptu dla AI", expanded=False):
        st.text_area(
            "Prompt",
            prompt,
            height=300,
            disabled=True,
            label_visibility="collapsed"
        )
        st.caption(f"Szacunkowa długość: ~{len(prompt.split())} słów")
        
except Exception as e:
    st.error(f"❌ Błąd budowania promptu: {e}")
    logger.error(f"Błąd promptu: {e}", exc_info=True)
    st.stop()

# ========================================================================================
# GENEROWANIE INSIGHTÓW
# ========================================================================================

st.divider()

generate_col1, generate_col2 = st.columns([3, 1])

with generate_col1:
    generate_button = st.button(
        "🔮 Wygeneruj insighty",
        type="primary",
        use_container_width=True,
        help="Generuje raport AI na podstawie danych"
    )

with generate_col2:
    if st.button("🗑️ Wyczyść historię", use_container_width=True):
        st.session_state["insights_history"] = []
        st.success("✅ Historia wyczyszczona")
        st.rerun()

if generate_button:
    start_time = time.time()
    
    # Progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Etap 1: Przygotowanie
        status_text.text("🔄 Przygotowuję dane...")
        progress_bar.progress(20)
        
        # Próbka dla AI (max 1000 wierszy dla szybkości)
        df_sample = df_main.head(1000) if len(df_main) > 1000 else df_main
        
        # Cache key
        cache_key = _create_cache_key(df_sample, prompt, config)
        
        # Etap 2: Generowanie
        status_text.text("🤖 Generuję insighty z AI...")
        progress_bar.progress(40)
        
        raw_insights = _cached_generate_insights(cache_key, df_sample, prompt)
        
        # Etap 3: Przetwarzanie
        status_text.text("📊 Przetwarzam odpowiedź...")
        progress_bar.progress(80)
        
        insight = _normalize_insights(raw_insights)
        
        # Etap 4: Zapisz do historii
        _add_to_history(insight)
        
        progress_bar.progress(100)
        elapsed = time.time() - start_time
        
        status_text.empty()
        progress_bar.empty()
        
        st.success(f"✅ Wygenerowano w {elapsed:.1f}s")
        
        # ============================================================================
        # PREZENTACJA WYNIKÓW
        # ============================================================================
        
        st.divider()
        
        # Tab layout dla lepszej organizacji
        tab1, tab2, tab3 = st.tabs([
            "📊 Raport", 
            "💾 Export", 
            "🔍 Debug"
        ])
        
        # TAB 1: RAPORT
        with tab1:
            # Summary
            st.markdown("### 🧭 Executive Summary")
            st.info(insight.summary)
            
            # Top Insights
            if insight.top_insights:
                st.markdown("### 📌 Top Insights")
                for i, item in enumerate(insight.top_insights, 1):
                    st.markdown(f"**{i}.** {item}")
            
            # Recommendations
            if insight.recommendations:
                st.markdown("### 💡 Rekomendacje")
                for rec in insight.recommendations:
                    st.success(f"✓ {rec}")
            
            # Risks
            if insight.risks:
                st.markdown("### ⚠️ Ryzyka i anomalie")
                for risk in insight.risks:
                    st.warning(f"⚠ {risk}")
            
            # Metadata
            with st.expander("ℹ️ Metadata raportu", expanded=False):
                st.json(insight.metadata)
        
        # TAB 2: EXPORT
        with tab2:
            st.subheader("💾 Eksportuj raport")
            
            col1, col2, col3 = st.columns(3)
            
            # JSON Export
            with col1:
                json_data = insight.to_json()
                st.download_button(
                    "⬇️ Pobierz JSON",
                    data=json_data,
                    file_name=f"ai_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            # Markdown Export
            with col2:
                md_data = insight.to_markdown()
                st.download_button(
                    "⬇️ Pobierz Markdown",
                    data=md_data,
                    file_name=f"ai_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
            # TXT Export
            with col3:
                txt_data = insight.to_markdown()  # Same as MD but .txt extension
                st.download_button(
                    "⬇️ Pobierz TXT",
                    data=txt_data,
                    file_name=f"ai_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            st.divider()
            
            # Preview exports
            with st.expander("👁️ Podgląd Markdown", expanded=False):
                st.code(md_data, language="markdown")
        
        # TAB 3: DEBUG
        with tab3:
            if config.show_json:
                st.subheader("🔍 Surowy JSON")
                st.json(raw_insights)
            else:
                st.info("Włącz 'Pokaż surowy JSON' w ustawieniach, aby zobaczyć debug info")
            
            st.subheader("⚙️ Cache info")
            st.code(f"Cache key: {cache_key}")
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        
        st.error(f"❌ Błąd generowania insightów: {e}")
        logger.error(f"Błąd generowania: {e}", exc_info=True)
        
        # Sugestie rozwiązania
        st.info(
            "💡 **Możliwe rozwiązania:**\n"
            "- Sprawdź połączenie z internetem\n"
            "- Zweryfikuj klucz API w konfiguracji\n"
            "- Spróbuj trybu 'Szybka'\n"
            "- Odznacz 'Uwzględnij strukturę kolumn'"
        )

else:
    st.info(
        "👆 Kliknij przycisk **Wygeneruj insighty**, aby rozpocząć analizę AI.\n\n"
        f"Tryb: **{depth_option}** • "
        f"Schema: **{'✓' if config.include_schema else '✗'}** • "
        f"Quality: **{'✓' if config.include_quality else '✗'}**"
    )

# ========================================================================================
# HISTORIA INSIGHTÓW
# ========================================================================================

history = st.session_state.get("insights_history", [])

if history:
    st.divider()
    st.subheader("📚 Historia insightów")
    
    for idx, hist_item in enumerate(history):
        timestamp = hist_item.get("timestamp", "Unknown")
        summary = hist_item.get("summary", "")[:200] + "..."
        
        with st.expander(f"🕒 {timestamp}", expanded=(idx == 0)):
            # Rekonstrukcja AIInsight z dict
            hist_insight = AIInsight(**hist_item)
            
            st.markdown(f"**Summary:** {hist_insight.summary}")
            
            if hist_insight.top_insights:
                st.markdown("**Top Insights:**")
                for i, insight_text in enumerate(hist_insight.top_insights[:3], 1):
                    st.markdown(f"{i}. {insight_text}")
            
            # Quick export
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ JSON",
                    data=hist_insight.to_json(),
                    file_name=f"history_{idx}.json",
                    mime="application/json",
                    key=f"hist_json_{idx}",
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    "⬇️ Markdown",
                    data=hist_insight.to_markdown(),
                    file_name=f"history_{idx}.md",
                    mime="text/markdown",
                    key=f"hist_md_{idx}",
                    use_container_width=True
                )

# ========================================================================================
# WSKAZÓWKI NAWIGACJI
# ========================================================================================

st.divider()
st.success(
    "✨ **Co dalej?**\n\n"
    "- **📊 EDA Analysis** — wizualizacje i statystyki\n"
    "- **🎯 Predictions** — trenowanie modeli ML\n"
    "- **📈 Forecasting** — prognozy szeregów czasowych\n"
    "- **📄 Reports** — generowanie raportów"
)