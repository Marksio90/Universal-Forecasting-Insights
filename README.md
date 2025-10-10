# 🔮 Intelligent Predictor PRO — End‑to‑End Analytics & Forecasting Suite

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-ff4b4b)
![Status](https://img.shields.io/badge/status-active-success)
![Tests](https://img.shields.io/badge/tests-pytest-green)

</div>

> **Cel:** uniwersalna aplikacja klasy **PRO++** do ingestu danych (wiele formatów), EDA, wniosków AI, treningu modeli (AutoML), prognoz szeregów czasowych oraz generowania raportów HTML/PDF — wszystko w jednym interfejsie **Streamlit**.

---

## 📌 Spis treści

- [Funkcje](#-funkcje)
- [Architektura](#-architektura)
- [Struktura katalogów](#-struktura-katalogów)
- [Instalacja](#-instalacja)
- [Konfiguracja](#-konfiguracja)
- [Szybki start](#-szybki-start)
- [Workflow w aplikacji](#-workflow-w-aplikacji)
- [Testy](#-testy)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)
- [Wkład (Contributing)](#-wkład-contributing)
- [Licencja](#-licencja)

---

## ✨ Funkcje

- **Any‑File Ingest**: CSV, XLSX, JSON, DOCX, PDF (parsery w `src/data_processing/file_parser.py`).
- **Walidacja & Czyszczenie**: typy, braki, outliery, rzutowania EU/US; wsparcie dla dat i walut (`validators.py`, `data_cleaner.py`). 
- **EDA PRO**: metryki, histogramy, korelacje, profilowanie (obsługa *ydata‑profiling* jeśli zainstalowane).
- **AI Insights**: wnioski językowe i rekomendacje biznesowe z LLM (`src/ai_engine/*`, integracja OpenAI z retry, rate‑limit, cache).
- **AutoML**: dobór modelu i trening (sklearn/LightGBM/XGBoost), metryki i ważność cech (`src/ml_models/automl_pipeline.py`).
- **Time‑Series**: prognozy, metryki MASE/sMAPE, dekompozycja (`src/ml_models/time_series.py`, `forecasting.py`). 
- **Raporty**: HTML/PDF z szablonów Jinja2, Playwright/WeasyPrint/ReportLab (`src/visualization/reports.py`, `assets/templates/`). 
- **Baza/Cache**: SQLAlchemy + lokalny cache, wektory (Chroma/Redis opcjonalnie) (`src/database/*`). 
- **Testy**: pytest – testy modułów ML, EDA, AI (`tests/*`). 
- **UI**: zakładki **1‑6**: Upload → EDA → AI → Predictions → Forecasting → Reports (`pages/*`).

---

## 🧱 Architektura

```mermaid
flowchart LR
    subgraph UI[Streamlit UI]
        U1[1_📤 Upload_Data]
U2[2_🔍 EDA_Analysis]
U3[3_🤖 AI_Insights]
U4[4_📈 Predictions]
U5[5_📊 Forecasting]
U6[6_📋 Reports]
    end

    subgraph Core[Core Pipeline]
        DP[data_processing]
ML[ml_models]
VIS[visualization]
AI[ai_engine]
DB[database]
UT[utils]
    end

    U1 --> DP
    U2 --> DP
    U3 --> AI
    U4 --> ML
    U5 --> ML
    U6 --> VIS
    DP <--> ML
    ML --> VIS
    AI --> VIS
    DP --> DB
    ML --> DB
    AI --> DB
    UI <-->|st.session_state| UT
```

---

## 📁 Struktura katalogów (skrót)

```
assets/
  images/logo.png
  styles/custom.css
  templates/report_template.html
pages/
  1_📤_Upload_Data.py
  2_🔍_EDA_Analysis.py
  3_🤖_AI_Insights.py
  4_📈_Predictions.py
  5_📊_Forecasting.py
  6_📋_Reports.py
src/
  ai_engine/               # Integracja LLM, wnioski, raporty AI
  data_processing/         # Parsery, walidacja, cleaning, FE
  ml_models/               # AutoML, klasyfikacja/regresja, TS
  visualization/           # Wykresy, dashboardy, PDF/HTML
  database/                # SQLAlchemy, cache, vector store
  utils/                   # logger, helpers, validators
tests/                     # pytest: AI, data, ML
app.py                     # Landing + nawigacja + metryki
config.yaml                # Konfiguracja PRO (środowiska, limity)
requirements.txt
README.md
```

---

## ⚙️ Instalacja

> **Wymagania:** Python **3.10–3.12** (sprawdzone), **pip >= 24**, pakiety systemowe dla kompilacji (Windows: MSVC Build Tools).

```bash
# 1) Klon i wejście
git clone <YOUR_REPO_URL>.git
cd <REPO_FOLDER>

# 2) (Opcjonalnie) wirtualne środowisko
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3) Aktualizacja pip/setuptools
python -m pip install -U pip setuptools wheel

# 4) Instalacja zależności
pip install -r requirements.txt

# 5) (Opcjonalnie) modele/TS
pip install lightgbm xgboost catboost prophet  # w razie potrzeby wsparcia dla pełnego AutoML/TS
```

> 💡 **Windows (Prophet/Playwright):**
> - Prophet bywa wrażliwy na ścieżki – w razie błędu z *Long Paths* włącz długie ścieżki w systemie.
> - Playwright po instalacji: `playwright install chromium` (jeśli używasz eksportu PDF przez przeglądarkę).

---

## 🔧 Konfiguracja

### 1) `.env` / `st.secrets`
Aplikacja **nie hardcoduje** kluczy. Wspierane zmienne (dopasuj do swojego środowiska):

```ini
# .env (lokalnie) lub .streamlit/secrets.toml (Streamlit Cloud)
OPENAI_API_KEY="sk-..."
DATABASE_URL="sqlite:///app.db"            # lub postgres://user:pass@host:5432/db
SECURITY_CODE="twoj_kod"                   # prosty guard aplikacji (wymagany przy starcie)
REGISTRY_PATH="./models/registry.json"     # ścieżka rejestru modeli
VECTORSTORE_DIR="./.vectorstore"           # katalog Chroma/FAISS (jeśli używasz)
ENVIRONMENT="development"                  # development|staging|production
```

> Zgodnie z preferencją bezpieczeństwa **PRO**: klucze i hasła tylko przez **.env/st.secrets**.

### 2) `config.yaml`
Centralne ustawienia (UI, limity, cache, tryb demo, eksport PDF). Przykład:

```yaml
app:
  title: "Intelligent Predictor PRO"
  environment: "development"
ui:
  theme: "dark"
limits:
  max_file_size_mb: 400
  max_rows_preview: 5000
integrations:
  openai_enabled: true
reports:
  pdf_engine: "playwright"   # playwright|weasyprint|reportlab
```

---

## 🏁 Szybki start

```bash
streamlit run app.py
# przeglądarka: http://localhost:8501
```

- **1 — Upload Data:** wgraj pliki (CSV/XLSX/JSON/DOCX/PDF).  
- **2 — EDA:** profilowanie, braki, rozkłady.  
- **3 — AI Insights:** wnioski językowe i rekomendacje (LLM).  
- **4 — Predictions:** AutoML (sklearn/LGBM/XGB), metryki i SHAP.  
- **5 — Forecasting:** serie czasowe, metryki MASE/sMAPE.  
- **6 — Reports:** PDF/HTML, tematy, numeracja, stopki.  

---

## 🔁 Workflow w aplikacji

```mermaid
flowchart TD
    A[Upload] --> B[Walidacja & Cleaning]
    B --> C[EDA]
    C --> D[AI Insights]
    B --> E[Feature Engineering]
    E --> F[AutoML Training]
    F --> G[Explainability (SHAP)]
    F --> H[Predictions Export]
    B --> I[Time‑Series/Forecasting]
    G --> J[Raport PDF/HTML]
    H --> J
    I --> J
```

---

## 🧪 Testy

```bash
pip install -r requirements.txt  # upewnij się, że pytest jest dostępny
pytest -q
```

Zakres: testy AI integracji, przetwarzania danych i modeli ML (`tests/test_*.py`).

---

## 🛠 Troubleshooting

- **Instalacja na Windows:** jeżeli występują błędy kompilacji, doinstaluj *Microsoft C++ Build Tools* oraz trzymaj się pinów `numpy==1.26.x`, `scipy==1.13.x` (zgodne ze scikit‑learn 1.5).  
- **Prophet/Stan:** problemy z długimi ścieżkami → włącz *Long Paths*, rozważ alternatywę (statsmodels, sktime) jeśli build jest kłopotliwy.  
- **Playwright PDF:** po instalacji uruchom `playwright install chromium`.  
- **Brak klucza OpenAI:** moduły AI wyłączą się grzecznie; ustaw `OPENAI_API_KEY` w `.env`/`st.secrets`.  
- **Port 8501 zajęty:** `streamlit run app.py --server.port 8502`.  

---

## 🗺 Roadmap

- [ ] Panel kalibracji klasyfikacji + „reject option” przy niskiej pewności
- [ ] Backtesting TS (rolling‑origin) + porównanie modeli
- [ ] Ensembling lekki (Stacking/Blending) w AutoML
- [ ] Monitorowanie i rejestr modeli (MLflow/registry v2)
- [ ] Scenariusze „What‑If” + porównanie wariantów A/B/C
- [ ] Eksport ZIP: raport + artefakty + logi

---

## 🤝 Wkład (Contributing)

- Fork → branch feat/… → PR.  
- Styl: czysty kod, typowanie, obsługa błędów, brak hardcodów.  
- Dołącz testy i krótką notkę w README, co zmienia PR.

---

## 📄 Licencja

Dodaj plik `LICENSE` (np. **MIT**). Jeżeli repozytorium jest prywatne – doprecyzuj zasady użycia wewnętrznego.

---
## Troubleshooting (Windows / PDF / Prophet / Playwright)

**Prophet / cmdstanpy**
1. `pip install cmdstanpy`
2. `python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"`
3. Dodaj `~/.cmdstan/` do PATH, zrestartuj terminal.

**WeasyPrint (PDF z polskimi znakami)**
- Windows: zainstaluj zależności GTK/cairo.
- Użyj fontu obsługującego PL (np. DejaVu Sans) w CSS.

**Playwright**
- `pip install playwright` + `playwright install`.

