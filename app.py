"""
Predictive Analytics Platform - Main Application
================================================
Aplikacja do automatycznej analizy i prognozowania biznesowego
"""

import streamlit as st
from pathlib import Path
import sys

# Dodaj katalog główny do ścieżki
sys.path.append(str(Path(__file__).parent))

from frontend.utils.session_state import init_session_state
from utils.logger import setup_logger
from config import APP_CONFIG

# Konfiguracja strony
st.set_page_config(
    page_title="Predictive Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.example.com',
        'Report a bug': 'https://github.com/example/issues',
        'About': '# Platforma Analityki Predykcyjnej\nWersja 1.0'
    }
)

# Setup logger
logger = setup_logger(__name__)

# Inicjalizacja session state
init_session_state()

# Custom CSS dla biznesowego wyglądu
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #4f46e5;
        color: white;
        border-radius: 0.375rem;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #4338ca;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

def main():
    """Główna funkcja aplikacji"""
    
    # Sidebar - Menu
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/4f46e5/ffffff?text=Analytics+AI", 
                 use_container_width=True)
        st.markdown("---")
        
        # Status połączenia z OpenAI
        if st.session_state.get('openai_connected', False):
            st.success("✅ OpenAI Connected")
        else:
            st.warning("⚠️ OpenAI Disconnected")
        
        st.markdown("---")
        
        # Informacje o sesji
        st.markdown("### 📋 Sesja")
        if st.session_state.get('uploaded_files'):
            st.metric("Wgrane pliki", len(st.session_state.uploaded_files))
        if st.session_state.get('analyses'):
            st.metric("Analizy", len(st.session_state.analyses))
        
        st.markdown("---")
        
        # Szybkie akcje
        st.markdown("### ⚡ Szybkie akcje")
        if st.button("🗑️ Wyczyść cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache wyczyszczony!")
        
        if st.button("💾 Zapisz sesję", use_container_width=True):
            st.info("Funkcja w przygotowaniu")
    
    # Header główny
    st.markdown('<h1 class="main-header">🎯 Predictive Analytics Platform</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Inteligentna analiza i prognozowanie biznesowe z AI</p>', 
                unsafe_allow_html=True)
    
    # Dashboard główny
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📁 Pliki",
            value=st.session_state.get('total_files', 0),
            delta="+2 dziś"
        )
    
    with col2:
        st.metric(
            label="📊 Analizy",
            value=st.session_state.get('total_analyses', 0),
            delta="+1 dziś"
        )
    
    with col3:
        st.metric(
            label="🎯 Prognozy",
            value=st.session_state.get('total_predictions', 0),
            delta="Aktywne"
        )
    
    with col4:
        st.metric(
            label="📈 Dokładność",
            value=f"{st.session_state.get('avg_accuracy', 0):.1f}%",
            delta="+2.3%"
        )
    
    st.markdown("---")
    
    # Główny content
    tab1, tab2, tab3 = st.tabs(["🚀 Start", "📚 Dokumentacja", "⚙️ Ustawienia"])
    
    with tab1:
        st.markdown("### 🎬 Jak zacząć?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 1️⃣ Wgraj dane
            - Obsługujemy formaty: CSV, Excel, DOC, PDF, JSON
            - Przejdź do strony **📤 Upload Data**
            - Przeciągnij lub wybierz pliki
            """)
            
            if st.button("📤 Przejdź do uploadu", use_container_width=True, type="primary"):
                st.switch_page("frontend/pages/1_📤_Upload_Data.py")
        
        with col2:
            st.markdown("""
            #### 2️⃣ Analizuj
            - System automatycznie przetworzy dane
            - Przeprowadzi EDA i czyszczenie
            - Wytrenuje odpowiednie modele ML
            """)
            
            if st.button("🔍 Przeglądaj analizy", use_container_width=True):
                st.switch_page("frontend/pages/3_📊_Analysis.py")
        
        st.markdown("---")
        
        # Ostatnie aktywności
        st.markdown("### 📋 Ostatnie aktywności")
        
        if st.session_state.get('recent_activities'):
            for activity in st.session_state.recent_activities[:5]:
                with st.expander(f"🔹 {activity['title']} - {activity['date']}"):
                    st.write(activity['description'])
        else:
            st.info("Brak ostatnich aktywności. Zacznij od wgrania danych!")
    
    with tab2:
        st.markdown("### 📚 Dokumentacja")
        
        st.markdown("""
        #### 🎯 Funkcje platformy
        
        **Automatyczne przetwarzanie:**
        - Wykrywanie typu danych (sprzedaż, zużycie, finanse)
        - Czyszczenie i normalizacja danych
        - Wypełnianie braków danych
        
        **Machine Learning:**
        - Automatyczny wybór najlepszego modelu
        - Trenowanie wielu modeli równolegle
        - Ensemble learning dla lepszej dokładności
        
        **OpenAI Integration:**
        - Inteligentna interpretacja wyników
        - Generowanie rekomendacji biznesowych
        - Automatyczne raporty executive
        
        **Eksport:**
        - PDF z pełną analizą
        - Excel z danymi i prognozami
        - PowerPoint z prezentacją
        """)
        
        with st.expander("🔧 Obsługiwane formaty"):
            st.markdown("""
            - **CSV** - pliki tekstowe z danymi tabelarycznymi
            - **Excel** (.xlsx, .xls) - arkusze kalkulacyjne
            - **DOC/DOCX** - dokumenty Word z danymi
            - **PDF** - dokumenty PDF z tabelami
            - **JSON** - strukturalne dane JSON
            """)
    
    with tab3:
        st.markdown("### ⚙️ Ustawienia")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🤖 OpenAI")
            api_key = st.text_input(
                "API Key",
                type="password",
                value=st.session_state.get('openai_api_key', ''),
                help="Wprowadź swój klucz API OpenAI"
            )
            if st.button("Zapisz API Key"):
                st.session_state.openai_api_key = api_key
                st.success("✅ Zapisano!")
            
            model = st.selectbox(
                "Model GPT",
                ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
                index=0
            )
        
        with col2:
            st.markdown("#### 📊 ML Settings")
            
            auto_ml = st.checkbox(
                "Auto ML",
                value=True,
                help="Automatyczny wybór najlepszego modelu"
            )
            
            n_models = st.slider(
                "Liczba modeli do testowania",
                min_value=1,
                max_value=10,
                value=5
            )
            
            confidence = st.slider(
                "Minimalny poziom pewności (%)",
                min_value=50,
                max_value=99,
                value=80
            )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error(f"❌ Wystąpił błąd: {str(e)}")
        st.info("Sprawdź logi aplikacji dla więcej informacji.")