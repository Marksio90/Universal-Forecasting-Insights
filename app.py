# app.py - REDESIGNED Professional Landing Page
from __future__ import annotations
import os
import pathlib
import streamlit as st
from dotenv import load_dotenv
import pandas as pd

# --- Paths & Config
APP_DIR = pathlib.Path(__file__).resolve().parent
ASSETS = APP_DIR / "assets"
load_dotenv()

from src.utils.logger import configure_logger, get_logger
configure_logger()
log = get_logger(__name__)

# --- Page Config
st.set_page_config(
    page_title="Intelligent Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load Custom CSS
css_path = ASSETS / "styles" / "custom.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ==============================================================================
# HERO SECTION
# ==============================================================================
def render_hero():
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0 2rem;">
        <h1 style="font-size: 3rem; font-weight: 800; margin: 0; background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🔮 Intelligent Predictor
        </h1>
        <p style="font-size: 1.3rem; color: #A5ADBA; margin-top: 0.5rem;">
            End-to-End Analytics & Forecasting Suite
        </p>
        <p style="font-size: 1rem; color: #6B7280; max-width: 600px; margin: 1rem auto;">
            Zaawansowana analiza danych, AutoML i prognozowanie biznesowe w jednym intuicyjnym interfejsie
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# QUICK START CARDS
# ==============================================================================
def render_quick_start():
    st.markdown("### 🚀 Szybki Start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📤</div>
            <h3>1. Wczytaj Dane</h3>
            <p>Obsługa CSV, XLSX, JSON, DOCX, PDF. Inteligentne parsowanie i walidacja.</p>
            <a href="/Upload_Data" target="_self" class="card-link">Zacznij tutaj →</a>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <h3>2. Analizuj & Trenuj</h3>
            <p>Automatyczna EDA, AI insights i AutoML z jednym kliknięciem.</p>
            <a href="/EDA_Analysis" target="_self" class="card-link">Eksploruj →</a>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3>3. Prognozuj</h3>
            <p>Szeregi czasowe z Prophet. Pasma niepewności i backtesting.</p>
            <a href="/Forecasting" target="_self" class="card-link">Przewiduj →</a>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# CAPABILITIES SECTION
# ==============================================================================
def render_capabilities():
    st.markdown("### ✨ Główne Możliwości")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔍 Eksploracja Danych**
        - Automatyczne profilowanie (ydata-profiling)
        - Interaktywne wizualizacje Plotly
        - Wykrywanie anomalii (Isolation Forest)
        - Smart data cleaning i feature engineering
        
        **📈 Modelowanie Predykcyjne**
        - AutoML (LightGBM → XGBoost → RF)
        - Regresja i klasyfikacja
        - SHAP interpretability
        - Cross-validation i hyperparameter tuning
        """)
    
    with col2:
        st.markdown("""
        **🤖 AI-Powered Insights**
        - Biznesowe wnioski z GPT-4
        - Automatyczne hipotezy i rekomendacje
        - Generowanie raportów HTML
        - Eksport kompletnych analiz
        
        **📊 Time Series Forecasting**
        - Prophet z konfigurowalnymi sezonowościami
        - Backtesting z rolling origin
        - Pasma niepewności (90%)
        - Obsługa regresorów zewnętrznych
        """)

# ==============================================================================
# DEMO DATA SECTION
# ==============================================================================
def render_demo_section():
    st.markdown("### 🧪 Wypróbuj na Danych Demo")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("📈 Timeseries: Daily Sales", use_container_width=True, type="primary"):
            from src.utils.helpers import _demo_timeseries_helper  # helper function
            df = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=120, freq='D'),
                'sales': 100 + 0.2 * range(120) + 10 * pd.np.sin(2 * pd.np.pi * pd.np.arange(120) / 7)
            })
            st.session_state['df_raw'] = df
            st.session_state['df'] = df
            st.session_state['goal'] = "Prognoza sprzedaży na kolejny miesiąc"
            st.success("✅ Załadowano dane demo. Przejdź do **Forecasting**")
            st.rerun()
    
    with col2:
        if st.button("🎯 Classification: Customer Churn", use_container_width=True):
            import numpy as np
            rng = np.random.default_rng(7)
            n = 200
            df = pd.DataFrame({
                'tenure': rng.integers(1, 72, n),
                'monthly_charges': rng.uniform(20, 120, n),
                'total_charges': rng.uniform(100, 8000, n),
                'churn': rng.choice([0, 1], n, p=[0.7, 0.3])
            })
            st.session_state['df_raw'] = df
            st.session_state['df'] = df
            st.session_state['goal'] = "Predykcja rezygnacji klienta"
            st.success("✅ Załadowano dane demo. Przejdź do **Predictions**")
            st.rerun()
    
    with col3:
        if st.button("📊 Regression: House Prices", use_container_width=True):
            import numpy as np
            rng = np.random.default_rng(42)
            n = 150
            df = pd.DataFrame({
                'sqft': rng.integers(800, 4000, n),
                'bedrooms': rng.integers(1, 6, n),
                'age': rng.integers(0, 50, n),
                'price': rng.uniform(150_000, 800_000, n)
            })
            st.session_state['df_raw'] = df
            st.session_state['df'] = df
            st.session_state['goal'] = "Prognoza ceny nieruchomości"
            st.success("✅ Załadowano dane demo. Przejdź do **Predictions**")
            st.rerun()

# ==============================================================================
# CURRENT SESSION PREVIEW (if data loaded)
# ==============================================================================
def render_session_preview():
    df = st.session_state.get("df")
    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
        st.markdown("---")
        st.markdown("### 📊 Aktywna Sesja")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Wiersze", f"{len(df):,}")
        col2.metric("Kolumny", f"{df.shape[1]:,}")
        col3.metric("Pamięć", f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
        
        target = st.session_state.get("target")
        if target:
            col4.metric("Cel", target)
        
        with st.expander("🔎 Podgląd danych", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)

# ==============================================================================
# TECH STACK (minimal, professional)
# ==============================================================================
def render_tech_stack():
    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **Core**  
        Python 3.10+ • Streamlit • Pandas
        """)
    
    with col2:
        st.markdown("""
        **ML/AI**  
        LightGBM • XGBoost • Prophet • OpenAI
        """)
    
    with col3:
        st.markdown("""
        **Visualization**  
        Plotly • ydata-profiling • SHAP
        """)
    
    with col4:
        st.markdown("""
        **Infrastructure**  
        SQLite • Redis • ChromaDB • Docker
        """)

# ==============================================================================
# HEALTH STATUS (collapsed by default)
# ==============================================================================
def render_health_status():
    with st.expander("🔧 System Status", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**APIs**")
            openai_ok = bool(os.getenv("OPENAI_API_KEY"))
            st.markdown(f"{'🟢' if openai_ok else '🔴'} OpenAI API")
        
        with col2:
            st.markdown("**ML Libraries**")
            try:
                import xgboost
                st.markdown("🟢 XGBoost")
            except:
                st.markdown("🔴 XGBoost")
            
            try:
                import lightgbm
                st.markdown("🟢 LightGBM")
            except:
                st.markdown("🔴 LightGBM")
            
            try:
                import prophet
                st.markdown("🟢 Prophet")
            except:
                st.markdown("🔴 Prophet")
        
        with col3:
            st.markdown("**Storage**")
            try:
                from src.database.db_manager import health_check
                db_ok = health_check()
                st.markdown(f"{'🟢' if db_ok else '🔴'} Database")
            except:
                st.markdown("🔴 Database")

# ==============================================================================
# MAIN RENDER
# ==============================================================================
def main():
    render_hero()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    render_quick_start()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    render_demo_section()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    render_session_preview()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    render_capabilities()
    
    render_tech_stack()
    
    render_health_status()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6B7280; padding: 2rem 0;">
        <p>© 2024 Intelligent Predictor • Zbudowano z ❤️ używając Python & Streamlit</p>
        <p style="font-size: 0.9rem;">
            <a href="https://github.com/your-org/intelligent-predictor" target="_blank" style="color: #4A90E2; text-decoration: none;">GitHub</a> • 
            <a href="https://docs.intelligent-predictor.io" target="_blank" style="color: #4A90E2; text-decoration: none;">Dokumentacja</a> • 
            <a href="mailto:support@intelligent-predictor.io" style="color: #4A90E2; text-decoration: none;">Kontakt</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()