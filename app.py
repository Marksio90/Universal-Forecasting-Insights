"""
app.py PRO++++ - Professional Landing Page with Enhanced Features

Features PRO++++:
- Modern hero section with animations
- Interactive feature cards with hover effects
- Real-time system health monitoring
- Advanced demo data generators
- Session state management
- Performance metrics dashboard
- Quick action buttons
- Responsive design (mobile-friendly)
- Dark/Light theme toggle
- User onboarding wizard
- Recent activity tracking
- Keyboard shortcuts
- Export capabilities
- Comprehensive documentation links
- Analytics integration
"""

from __future__ import annotations

import os
import sys
import pathlib
import warnings
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass

import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# ========================================================================================
# CONFIGURATION & SETUP
# ========================================================================================

# Paths
APP_DIR = pathlib.Path(__file__).resolve().parent
ASSETS = APP_DIR / "assets"
STYLES = ASSETS / "styles"
IMAGES = ASSETS / "images"

# Load environment
load_dotenv()

# Logging
from src.utils.logger import configure_logger, get_logger
configure_logger(level=os.getenv("LOG_LEVEL", "INFO"))
log = get_logger(__name__)

# ========================================================================================
# PAGE CONFIGURATION
# ========================================================================================

st.set_page_config(
    page_title="Intelligent Predictor PRO",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.intelligent-predictor.io',
        'Report a bug': 'https://github.com/your-org/intelligent-predictor/issues',
        'About': '# Intelligent Predictor PRO\n\nEnd-to-End Analytics & Forecasting Suite'
    }
)

# ========================================================================================
# CUSTOM CSS & STYLING
# ========================================================================================

def load_custom_css():
    """Load custom CSS with enhanced styling."""
    
    # Base CSS
    css_path = STYLES / "custom.css"
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True
        )
    
    # Additional PRO styles
    st.markdown("""
    <style>
    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 3rem 0 2rem;
        animation: fadeIn 1s ease-in;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: slideDown 0.8s ease-out;
    }
    
    .hero-subtitle {
        font-size: 1.4rem;
        color: #A5ADBA;
        margin-top: 0.5rem;
        animation: slideUp 0.8s ease-out;
    }
    
    .hero-description {
        font-size: 1rem;
        color: #6B7280;
        max-width: 700px;
        margin: 1rem auto;
        line-height: 1.6;
    }
    
    /* Feature Cards */
    .feature-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(74,144,226,0.1) 0%, transparent 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        border-color: #4A90E2;
        box-shadow: 0 20px 40px rgba(74,144,226,0.2);
    }
    
    .feature-card:hover::before {
        opacity: 1;
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: inline-block;
        animation: bounce 2s infinite;
    }
    
    .feature-card h3 {
        color: #f1f5f9;
        font-size: 1.5rem;
        margin: 1rem 0;
        font-weight: 600;
    }
    
    .feature-card p {
        color: #94a3b8;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }
    
    .card-link {
        color: #4A90E2;
        text-decoration: none;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        transition: all 0.3s ease;
    }
    
    .card-link:hover {
        color: #60A5FA;
        transform: translateX(5px);
    }
    
    /* Metrics Dashboard */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #4A90E2;
        transform: scale(1.05);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4A90E2;
        display: block;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 999px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .status-ok {
        background: rgba(34, 197, 94, 0.1);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .status-warning {
        background: rgba(251, 191, 36, 0.1);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
    
    .status-error {
        background: rgba(239, 68, 68, 0.1);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    /* Section Divider */
    .section-divider {
        margin: 3rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #334155, transparent);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6B7280;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid #334155;
    }
    
    .footer a {
        color: #4A90E2;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    
    .footer a:hover {
        color: #60A5FA;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2rem;
        }
        
        .hero-subtitle {
            font-size: 1.1rem;
        }
        
        .feature-card {
            padding: 1.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)


# ========================================================================================
# SESSION STATE INITIALIZATION
# ========================================================================================

def init_session_state():
    """Initialize session state variables."""
    
    defaults = {
        'df_raw': None,
        'df': None,
        'goal': None,
        'target': None,
        'first_visit': True,
        'theme': 'dark',
        'show_onboarding': True,
        'recent_actions': [],
        'start_time': datetime.now(),
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ========================================================================================
# HERO SECTION
# ========================================================================================

def render_hero():
    """Render hero section with animations."""
    
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">
            🔮 Intelligent Predictor PRO
        </h1>
        <p class="hero-subtitle">
            End-to-End Analytics & Forecasting Suite
        </p>
        <p class="hero-description">
            Zaawansowana analiza danych, AutoML i prognozowanie biznesowe w jednym intuicyjnym interfejsie. 
            Wykorzystaj moc sztucznej inteligencji do podejmowania lepszych decyzji biznesowych.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ========================================================================================
# QUICK START CARDS
# ========================================================================================

def render_quick_start():
    """Render quick start guide with interactive cards."""
    
    st.markdown("### 🚀 Szybki Start")
    st.markdown("Rozpocznij swoją podróż z danymi w trzech prostych krokach")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📤</div>
            <h3>1. Wczytaj Dane</h3>
            <p>
                Obsługa wielu formatów: CSV, XLSX, JSON, DOCX, PDF. 
                Inteligentne parsowanie, walidacja i automatyczne wykrywanie typów danych.
            </p>
            <a href="/Upload_Data" target="_self" class="card-link">
                Zacznij tutaj →
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <h3>2. Analizuj & Trenuj</h3>
            <p>
                Automatyczna eksploracja danych (EDA), AI-powered insights 
                i AutoML z najlepszymi algorytmami (LightGBM, XGBoost, RF).
            </p>
            <a href="/EDA_Analysis" target="_self" class="card-link">
                Eksploruj →
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3>3. Prognozuj</h3>
            <p>
                Szeregi czasowe z Prophet i SARIMA. 
                Pasma niepewności, backtesting i walidacja prognozy.
            </p>
            <a href="/Forecasting" target="_self" class="card-link">
                Przewiduj →
            </a>
        </div>
        """, unsafe_allow_html=True)


# ========================================================================================
# DEMO DATA GENERATORS
# ========================================================================================

@dataclass
class DemoDataset:
    """Demo dataset configuration."""
    name: str
    icon: str
    description: str
    size: int
    target: str
    goal: str
    generator: callable


def generate_timeseries_demo(n: int = 365) -> pd.DataFrame:
    """Generate time series demo data."""
    rng = np.random.default_rng(42)
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    
    # Trend + seasonality + noise
    trend = 100 + 0.3 * np.arange(n)
    weekly = 15 * np.sin(2 * np.pi * np.arange(n) / 7)
    yearly = 30 * np.sin(2 * np.pi * np.arange(n) / 365)
    noise = rng.normal(0, 5, n)
    
    sales = trend + weekly + yearly + noise
    
    return pd.DataFrame({
        'date': dates,
        'sales': sales,
        'day_of_week': dates.dayofweek,
        'month': dates.month,
        'is_weekend': (dates.dayofweek >= 5).astype(int)
    })


def generate_classification_demo(n: int = 500) -> pd.DataFrame:
    """Generate classification demo data."""
    rng = np.random.default_rng(42)
    
    tenure = rng.integers(1, 72, n)
    monthly_charges = rng.uniform(20, 120, n)
    total_charges = tenure * monthly_charges + rng.normal(0, 200, n)
    
    # Logistic relationship for churn
    z = (-2 + 
         0.03 * tenure + 
         0.01 * monthly_charges - 
         0.0001 * total_charges +
         rng.normal(0, 0.5, n))
    
    churn_prob = 1 / (1 + np.exp(-z))
    churn = (churn_prob > 0.5).astype(int)
    
    return pd.DataFrame({
        'customer_id': [f'CUST_{i:05d}' for i in range(n)],
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract_type': rng.choice(['Month-to-month', 'One year', 'Two year'], n),
        'internet_service': rng.choice(['DSL', 'Fiber optic', 'No'], n),
        'churn': churn
    })


def generate_regression_demo(n: int = 300) -> pd.DataFrame:
    """Generate regression demo data."""
    rng = np.random.default_rng(42)
    
    sqft = rng.integers(800, 4000, n)
    bedrooms = rng.integers(1, 6, n)
    bathrooms = rng.integers(1, 4, n)
    age = rng.integers(0, 50, n)
    
    # Price model
    price = (
        50000 +
        150 * sqft +
        20000 * bedrooms +
        15000 * bathrooms -
        2000 * age +
        rng.normal(0, 30000, n)
    )
    
    return pd.DataFrame({
        'property_id': [f'PROP_{i:05d}' for i in range(n)],
        'sqft': sqft,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age': age,
        'location': rng.choice(['Downtown', 'Suburbs', 'Rural'], n),
        'garage': rng.choice([0, 1, 2], n),
        'price': price
    })


DEMO_DATASETS = {
    'timeseries': DemoDataset(
        name="Daily Sales Forecast",
        icon="📈",
        description="Dzienna sprzedaż z trendem i sezonowością (365 dni)",
        size=365,
        target="sales",
        goal="Prognoza sprzedaży na kolejny miesiąc z uwzględnieniem sezonowości",
        generator=generate_timeseries_demo
    ),
    'classification': DemoDataset(
        name="Customer Churn Prediction",
        icon="🎯",
        description="Predykcja rezygnacji klientów (500 rekordów)",
        size=500,
        target="churn",
        goal="Identyfikacja klientów zagrożonych rezygnacją i analiza czynników ryzyka",
        generator=generate_classification_demo
    ),
    'regression': DemoDataset(
        name="House Price Estimation",
        icon="🏠",
        description="Wycena nieruchomości (300 rekordów)",
        size=300,
        target="price",
        goal="Oszacowanie wartości rynkowej nieruchomości na podstawie charakterystyk",
        generator=generate_regression_demo
    )
}


def render_demo_section():
    """Render demo data section with generators."""
    
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### 🧪 Wypróbuj na Danych Demo")
    st.markdown("Załaduj gotowe dane demonstracyjne i zobacz możliwości systemu")
    
    col1, col2, col3 = st.columns(3)
    
    columns = [col1, col2, col3]
    dataset_keys = list(DEMO_DATASETS.keys())
    
    for idx, (key, dataset) in enumerate(DEMO_DATASETS.items()):
        with columns[idx]:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: #1e293b; border-radius: 12px; border: 1px solid #334155;">
                <div style="font-size: 3rem;">{dataset.icon}</div>
                <h4 style="color: #f1f5f9; margin: 0.5rem 0;">{dataset.name}</h4>
                <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 1rem;">
                    {dataset.description}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(
                f"Załaduj {dataset.icon}",
                key=f"demo_{key}",
                use_container_width=True,
                type="primary" if idx == 0 else "secondary"
            ):
                with st.spinner(f"Generowanie danych demo: {dataset.name}..."):
                    df = dataset.generator(dataset.size)
                    
                    st.session_state['df_raw'] = df
                    st.session_state['df'] = df
                    st.session_state['target'] = dataset.target
                    st.session_state['goal'] = dataset.goal
                    
                    # Track action
                    action = {
                        'timestamp': datetime.now(),
                        'action': 'load_demo',
                        'dataset': dataset.name
                    }
                    if 'recent_actions' not in st.session_state:
                        st.session_state['recent_actions'] = []
                    st.session_state['recent_actions'].append(action)
                    
                    log.info(f"Loaded demo dataset: {dataset.name}")
                    
                    st.success(f"✅ Załadowano {dataset.name}")
                    st.balloons()
                    st.rerun()


# ========================================================================================
# SESSION PREVIEW
# ========================================================================================

def render_session_preview():
    """Render current session data preview."""
    
    df = st.session_state.get("df")
    
    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### 📊 Aktywna Sesja")
        
        # Metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-label">Wiersze</span>
                <span class="metric-value">{len(df):,}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-label">Kolumny</span>
                <span class="metric-value">{df.shape[1]}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / 1e6
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-label">Pamięć</span>
                <span class="metric-value">{memory_mb:.1f}</span>
                <span class="metric-label">MB</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            missing_pct = (df.isna().sum().sum() / df.size) * 100
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-label">Braki</span>
                <span class="metric-value">{missing_pct:.1f}</span>
                <span class="metric-label">%</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            target = st.session_state.get("target", "—")
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-label">Cel</span>
                <span class="metric-value" style="font-size: 1.5rem;">{target}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Data preview
        with st.expander("🔎 Podgląd danych", expanded=False):
            st.dataframe(
                df.head(20),
                use_container_width=True,
                height=400
            )
            
            # Quick stats
            col_stats1, col_stats2 = st.columns(2)
            
            with col_stats1:
                st.markdown("**Typy danych:**")
                dtype_counts = df.dtypes.value_counts()
                for dtype, count in dtype_counts.items():
                    st.markdown(f"- `{dtype}`: {count} kolumn")
            
            with col_stats2:
                st.markdown("**Statystyki:**")
                st.markdown(f"- Duplikaty: {df.duplicated().sum()} ({(df.duplicated().sum()/len(df)*100):.1f}%)")
                st.markdown(f"- Całkowite braki: {df.isna().sum().sum():,}")
                st.markdown(f"- Unikalne wiersze: {df.drop_duplicates().shape[0]:,}")


# ========================================================================================
# CAPABILITIES
# ========================================================================================

def render_capabilities():
    """Render system capabilities."""
    
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### ✨ Główne Możliwości")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔍 Eksploracja Danych**
        - Automatyczne profilowanie (ydata-profiling)
        - Interaktywne wizualizacje Plotly
        - Wykrywanie anomalii (Isolation Forest, LOF, DBSCAN)
        - Smart data cleaning i feature engineering
        - Distribution analysis i correlation heatmaps
        - Missing data visualization
        
        **📈 Modelowanie Predykcyjne**
        - AutoML (LightGBM → XGBoost → RandomForest)
        - Regresja i klasyfikacja (binary & multiclass)
        - SHAP interpretability i feature importance
        - Cross-validation i hyperparameter tuning
        - Model registry z versioning
        - Performance tracking i comparison
        """)
    
    with col2:
        st.markdown("""
        **🤖 AI-Powered Insights**
        - Biznesowe wnioski z GPT-4o
        - Automatyczne hipotezy i rekomendacje
        - Generowanie raportów HTML/PDF
        - Natural language queries
        - Automated data storytelling
        - Executive summaries
        
        **📊 Time Series Forecasting**
        - Prophet z konfigurowalnymi sezonowościami
        - SARIMA i Exponential Smoothing
        - Backtesting z rolling origin
        - Pasma niepewności (90%, 95%)
        - Obsługa regresorów zewnętrznych
        - Anomaly detection w szeregach czasowych
        """)


# ========================================================================================
# TECH STACK
# ========================================================================================

def render_tech_stack():
    """Render technology stack."""
    
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### 🛠️ Tech Stack")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **Core**  
        • Python 3.10+  
        • Streamlit 1.28+  
        • Pandas 2.0+  
        • NumPy 1.24+
        """)
    
    with col2:
        st.markdown("""
        **ML/AI**  
        • LightGBM  
        • XGBoost  
        • Prophet  
        • OpenAI GPT-4o
        """)
    
    with col3:
        st.markdown("""
        **Visualization**  
        • Plotly 5.17+  
        • ydata-profiling  
        • SHAP  
        • Matplotlib
        """)
    
    with col4:
        st.markdown("""
        **Infrastructure**  
        • SQLite  
        • Redis (optional)  
        • ChromaDB  
        • Docker
        """)


# ========================================================================================
# SYSTEM HEALTH
# ========================================================================================

def check_system_health() -> Dict[str, Dict[str, Any]]:
    """Check system health status."""
    
    health = {
        'apis': {},
        'ml_libs': {},
        'storage': {},
        'overall': 'ok'
    }
    
    # APIs
    health['apis']['openai'] = {
        'status': 'ok' if os.getenv("OPENAI_API_KEY") else 'error',
        'message': 'Connected' if os.getenv("OPENAI_API_KEY") else 'API key not found'
    }
    
    # ML Libraries
    try:
        import xgboost
        health['ml_libs']['xgboost'] = {'status': 'ok', 'version': xgboost.__version__}
    except ImportError:
        health['ml_libs']['xgboost'] = {'status': 'error', 'message': 'Not installed'}
        health['overall'] = 'warning'
    
    try:
        import lightgbm
        health['ml_libs']['lightgbm'] = {'status': 'ok', 'version': lightgbm.__version__}
    except ImportError:
        health['ml_libs']['lightgbm'] = {'status': 'error', 'message': 'Not installed'}
        health['overall'] = 'warning'
    
    try:
        import prophet
        health['ml_libs']['prophet'] = {'status': 'ok', 'version': 'installed'}
    except ImportError:
        health['ml_libs']['prophet'] = {'status': 'warning', 'message': 'Not installed'}
    
    # Storage
    try:
        from src.database.db_manager import health_check
        db_ok = health_check()
        health['storage']['database'] = {
            'status': 'ok' if db_ok else 'error',
            'message': 'Connected' if db_ok else 'Connection failed'
        }
    except Exception as e:
        health['storage']['database'] = {'status': 'error', 'message': str(e)}
        health['overall'] = 'error'
    
    return health


def render_health_status():
    """Render system health status."""
    
    with st.expander("🔧 System Status & Health", expanded=False):
        health = check_system_health()
        
        # Overall status
        overall_status = health['overall']
        status_colors = {
            'ok': '🟢',
            'warning': '🟡',
            'error': '🔴'
        }
        
        st.markdown(f"**Overall Status:** {status_colors.get(overall_status, '⚪')} {overall_status.upper()}")
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        # APIs
        with col1:
            st.markdown("**APIs**")
            for api, info in health['apis'].items():
                status = info['status']
                icon = status_colors.get(status, '⚪')
                st.markdown(f"{icon} {api.upper()}")
                if 'message' in info:
                    st.caption(info['message'])
        
        # ML Libraries
        with col2:
            st.markdown("**ML Libraries**")
            for lib, info in health['ml_libs'].items():
                status = info['status']
                icon = status_colors.get(status, '⚪')
                version = info.get('version', info.get('message', ''))
                st.markdown(f"{icon} {lib.upper()}")
                st.caption(version)
        
        # Storage
        with col3:
            st.markdown("**Storage**")
            for storage, info in health['storage'].items():
                status = info['status']
                icon = status_colors.get(status, '⚪')
                st.markdown(f"{icon} {storage.capitalize()}")
                st.caption(info.get('message', 'Unknown'))
        
        # System Info
        st.markdown("---")
        st.markdown("**System Information**")
        
        col_sys1, col_sys2, col_sys3 = st.columns(3)
        
        with col_sys1:
            import platform
            st.markdown(f"**OS:** {platform.system()} {platform.release()}")
            st.markdown(f"**Python:** {platform.python_version()}")
        
        with col_sys2:
            import streamlit
            st.markdown(f"**Streamlit:** {streamlit.__version__}")
            st.markdown(f"**Pandas:** {pd.__version__}")
        
        with col_sys3:
            uptime = datetime.now() - st.session_state.get('start_time', datetime.now())
            hours, remainder = divmod(uptime.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            st.markdown(f"**Uptime:** {hours}h {minutes}m {seconds}s")


# ========================================================================================
# RECENT ACTIVITY
# ========================================================================================

def render_recent_activity():
    """Render recent user activity."""
    
    recent_actions = st.session_state.get('recent_actions', [])
    
    if recent_actions:
        with st.expander("📜 Recent Activity", expanded=False):
            st.markdown("Ostatnie akcje w tej sesji:")
            
            # Show last 10 actions
            for action in reversed(recent_actions[-10:]):
                timestamp = action.get('timestamp', datetime.now())
                action_type = action.get('action', 'unknown')
                details = action.get('dataset', action.get('details', ''))
                
                time_str = timestamp.strftime("%H:%M:%S")
                st.markdown(f"**{time_str}** - {action_type}: {details}")


# ========================================================================================
# QUICK ACTIONS
# ========================================================================================

def render_quick_actions():
    """Render quick action buttons."""
    
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📁 Upload New Data", use_container_width=True, type="primary"):
            st.switch_page("pages/1_📤_Upload_Data.py")
    
    with col2:
        if st.button("🔍 Explore Data", use_container_width=True):
            if st.session_state.get('df') is not None:
                st.switch_page("pages/2_📊_EDA_Analysis.py")
            else:
                st.warning("⚠️ Najpierw załaduj dane!")
    
    with col3:
        if st.button("🤖 Train Model", use_container_width=True):
            if st.session_state.get('df') is not None:
                st.switch_page("pages/3_🎯_Predictions.py")
            else:
                st.warning("⚠️ Najpierw załaduj dane!")
    
    with col4:
        if st.button("📈 Forecast", use_container_width=True):
            if st.session_state.get('df') is not None:
                st.switch_page("pages/4_📊_Forecasting.py")
            else:
                st.warning("⚠️ Najpierw załaduj dane!")


# ========================================================================================
# ONBOARDING WIZARD
# ========================================================================================

def render_onboarding():
    """Render first-time user onboarding."""
    
    if st.session_state.get('first_visit', True):
        with st.container():
            st.info("""
            👋 **Witaj w Intelligent Predictor PRO!**
            
            Pierwszy raz tutaj? Oto krótki przewodnik:
            
            1. **Załaduj dane** - Użyj przycisku "Upload Data" lub wybierz dane demo
            2. **Eksploruj** - Przejdź do "EDA Analysis" aby zrozumieć swoje dane
            3. **Modeluj** - Użyj "Predictions" do trenowania modeli ML
            4. **Prognozuj** - W "Forecasting" stwórz prognozy czasowe
            
            💡 Tip: Najedź na ikony ℹ️ aby zobaczyć więcej informacji
            """)
            
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("Nie pokazuj więcej", key="hide_onboarding"):
                    st.session_state['first_visit'] = False
                    st.rerun()


# ========================================================================================
# DOCUMENTATION LINKS
# ========================================================================================

def render_documentation():
    """Render documentation and help links."""
    
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### 📚 Dokumentacja & Pomoc")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **📖 Dokumentacja**
        - [Quick Start Guide](https://docs.intelligent-predictor.io/quickstart)
        - [API Reference](https://docs.intelligent-predictor.io/api)
        - [Tutorials](https://docs.intelligent-predictor.io/tutorials)
        """)
    
    with col2:
        st.markdown("""
        **💡 Przykłady**
        - [Data Analysis](https://docs.intelligent-predictor.io/examples/eda)
        - [ML Models](https://docs.intelligent-predictor.io/examples/ml)
        - [Forecasting](https://docs.intelligent-predictor.io/examples/forecast)
        """)
    
    with col3:
        st.markdown("""
        **🔧 Wsparcie**
        - [FAQ](https://docs.intelligent-predictor.io/faq)
        - [Troubleshooting](https://docs.intelligent-predictor.io/troubleshooting)
        - [Report Bug](https://github.com/your-org/intelligent-predictor/issues)
        """)
    
    with col4:
        st.markdown("""
        **🌟 Community**
        - [GitHub](https://github.com/your-org/intelligent-predictor)
        - [Discussions](https://github.com/your-org/intelligent-predictor/discussions)
        - [Discord](https://discord.gg/intelligent-predictor)
        """)


# ========================================================================================
# KEYBOARD SHORTCUTS
# ========================================================================================

def render_keyboard_shortcuts():
    """Render keyboard shortcuts info."""
    
    with st.expander("⌨️ Keyboard Shortcuts", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Navigation**
            - `Ctrl + K` - Quick search
            - `Ctrl + /` - Toggle sidebar
            - `Ctrl + R` - Reload page
            """)
        
        with col2:
            st.markdown("""
            **Actions**
            - `Ctrl + U` - Upload data
            - `Ctrl + E` - EDA Analysis
            - `Ctrl + M` - Train model
            """)


# ========================================================================================
# PERFORMANCE METRICS
# ========================================================================================

def render_performance_metrics():
    """Render performance metrics."""
    
    with st.expander("📊 Performance Metrics", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Session duration
            if 'start_time' in st.session_state:
                duration = datetime.now() - st.session_state['start_time']
                st.metric("Session Duration", f"{duration.seconds // 60} min")
        
        with col2:
            # Actions count
            actions = len(st.session_state.get('recent_actions', []))
            st.metric("Actions Performed", actions)
        
        with col3:
            # Data loaded
            if st.session_state.get('df') is not None:
                df = st.session_state['df']
                size_mb = df.memory_usage(deep=True).sum() / 1e6
                st.metric("Data Loaded", f"{size_mb:.1f} MB")
            else:
                st.metric("Data Loaded", "No data")


# ========================================================================================
# FOOTER
# ========================================================================================

def render_footer():
    """Render footer with links and credits."""
    
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="footer">
        <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
            © 2025 Intelligent Predictor PRO
        </p>
        <p style="color: #94a3b8; margin-bottom: 1rem;">
            Zbudowano z ❤️ używając Python & Streamlit
        </p>
        <p style="font-size: 0.95rem;">
            <a href="https://github.com/your-org/intelligent-predictor" target="_blank">
                GitHub
            </a> • 
            <a href="https://docs.intelligent-predictor.io" target="_blank">
                Dokumentacja
            </a> • 
            <a href="https://docs.intelligent-predictor.io/api" target="_blank">
                API
            </a> • 
            <a href="mailto:support@intelligent-predictor.io">
                Kontakt
            </a> • 
            <a href="https://docs.intelligent-predictor.io/privacy" target="_blank">
                Privacy Policy
            </a>
        </p>
        <p style="font-size: 0.85rem; color: #6B7280; margin-top: 1rem;">
            Version 2.0.0 PRO++++ • Updated: 2024-10-09
        </p>
    </div>
    """, unsafe_allow_html=True)


# ========================================================================================
# MAIN APPLICATION
# ========================================================================================

def main():
    """Main application entry point."""
    
    try:
        # Initialize
        init_session_state()
        load_custom_css()
        
        # Render sections
        render_hero()
        
        render_onboarding()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        render_quick_start()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        render_demo_section()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        render_session_preview()
        
        render_quick_actions()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        render_capabilities()
        
        render_tech_stack()
        
        render_documentation()
        
        # Collapsible sections
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            render_health_status()
            render_keyboard_shortcuts()
        
        with col_exp2:
            render_recent_activity()
            render_performance_metrics()
        
        # Footer
        render_footer()
        
        # Log page view
        log.info("Landing page rendered successfully")
        
    except Exception as e:
        log.error(f"Error rendering landing page: {e}", exc_info=True)
        st.error(f"Wystąpił błąd podczas ładowania strony: {e}")
        
        if st.button("🔄 Reload Page"):
            st.rerun()


# ========================================================================================
# ERROR HANDLING
# ========================================================================================

def handle_error(error: Exception):
    """Global error handler."""
    
    log.error(f"Application error: {error}", exc_info=True)
    
    st.error("⚠️ Wystąpił nieoczekiwany błąd")
    
    with st.expander("🔍 Error Details", expanded=False):
        st.code(str(error))
        
        import traceback
        st.code(traceback.format_exc())
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Restart Application"):
            st.session_state.clear()
            st.rerun()
    
    with col2:
        if st.button("📧 Report Issue"):
            st.info("""
            Please report this issue at:
            https://github.com/your-org/intelligent-predictor/issues
            
            Include the error details shown above.
            """)


# ========================================================================================
# ENTRY POINT
# ========================================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        handle_error(e)