# 🔮 Intelligent Predictor

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

**Uniwersalna aplikacja do analizy danych, AutoML i prognozowania szeregów czasowych**

[Dokumentacja](#dokumentacja) •
[Demo](#szybki-start) •
[Instalacja](#instalacja) •
[Przykłady](#workflow-w-aplikacji) •
[Roadmap](#roadmap--wkład-contributing)

</div>

---

## 📋 Spis treści

- [O projekcie](#-o-projekcie)
- [Kluczowe funkcje](#-kluczowe-funkcje)
- [Technologie](#-technologie)
- [Szybki start](#-szybki-start)
- [Wymagania](#-wymagania)
- [Instalacja](#-instalacja)
- [Konfiguracja](#-konfiguracja)
- [Struktura projektu](#-struktura-projektu)
- [Workflow](#-workflow-w-aplikacji)
- [Testy](#-testy)
- [Docker](#-docker)
- [CI/CD](#-cicd)
- [Bezpieczeństwo](#-bezpieczeństwo)
- [Contributing](#-roadmap--wkład)
- [Licencja](#-licencja)

---

## 🎯 O projekcie

**Intelligent Predictor** to kompleksowe rozwiązanie do zaawansowanej analizy danych, uczenia maszynowego i prognozowania biznesowego. Aplikacja łączy w sobie moc AutoML, AI-powered insights oraz automatyczne generowanie raportów w jednym intuicyjnym interfejsie Streamlit.

### Dla kogo?

- 📊 **Data Scientists** - szybkie prototypowanie i testowanie modeli
- 💼 **Business Analysts** - insighty biznesowe bez kodu
- 🎓 **Badacze** - automatyczna analiza i dokumentacja wyników
- 🏢 **Przedsiębiorstwa** - end-to-end pipeline od danych do decyzji

---

## ✨ Kluczowe funkcje

### 🧩 Multi-format Data Ingestion
- **Formaty**: CSV, XLSX, JSON, DOCX, PDF
- Inteligentne parsowanie i automatyczna detekcja typów
- Obsługa dużych plików (do 400 MB)

### 🧹 AI-Powered Data Cleaning
- Automatyczne czyszczenie i normalizacja
- Inteligentne rzutowanie formatów (EU/US daty, procenty, waluty)
- Feature engineering z dat (dzień tygodnia, miesiąc, kwartał)
- Kodowanie zmiennych kategorycznych

### 🔍 Exploratory Data Analysis (EDA)
- Automatyczne profilowanie danych (ydata-profiling)
- Interaktywne dashboardy Plotly
- Macierze korelacji i heatmapy
- Histogramy z algorytmem Freedmana-Diaconisa
- WebGL scatter plots dla dużych zbiorów

### 🤖 AI Insights & Reports
- Generowanie insightów biznesowych (OpenAI GPT)
- Automatyczne hipotezy i rekomendacje
- Raporty w formacie HTML z szablonami Jinja
- Eksport ZIP z pełnym kontekstem

### 📈 AutoML
- **Algorytmy**: LightGBM → XGBoost → Random Forest (fallback)
- **Zadania**: Regresja i klasyfikacja
- **Metryki**: RMSE, R², Accuracy, F1-weighted
- SHAP values dla interpretowalności
- Atomiczny rejestr modeli (JSON + joblib)

### 📊 Time Series Forecasting
- **Prophet** z konfigurowalnymi parametrami
- Automatyczna detekcja częstotliwości
- Pasma niepewności (90%)
- Sezonowości miesięczne/kwartalne
- Obsługa regresorów zewnętrznych
- Backtesting z konfigurowalnymi foldami

### 🎨 Anomaly Detection
- Isolation Forest
- Wizualizacja PCA 2D
- Histogram anomaly scores

### 📝 Rozbudowany Logging
- Loguru: konsola, plik, JSONL
- Rotacja i retencja logów
- Memory sink widoczny w UI
- Structured logging dla audytu

---

## 🛠 Technologie

### Core Stack
```
Python 3.10-3.11  |  Streamlit  |  Pandas  |  NumPy
```

### Machine Learning
```
LightGBM  |  XGBoost  |  Scikit-learn  |  Prophet  |  SHAP
```

### AI & NLP
```
OpenAI GPT-4  |  LangChain  |  Tiktoken
```

### Data Processing
```
Openpyxl  |  Python-docx  |  PyPDF2  |  ydata-profiling
```

### Visualization
```
Plotly  |  Matplotlib  |  Seaborn
```

### Infrastructure
```
SQLite  |  Redis (opcjonalnie)  |  ChromaDB  |  Docker
```

---

## 🚀 Szybki start

### Podstawowe uruchomienie

```bash
# 1. Instalacja zależności
pip install -r requirements.txt

# 2. Konfiguracja klucza API
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Uruchomienie aplikacji
streamlit run app.py
```

### 🧪 Demo z przykładowymi danymi

1. Otwórz aplikację w przeglądarce
2. W pasku bocznym wybierz **🧪 Dane demo**
3. Kliknij **Wczytaj demo** (timeseries lub klasyfikacja)
4. Eksploruj automatycznie wygenerowane analizy!

---

## 📦 Wymagania

### Systemowe

- **Python**: 3.10 - 3.11
- **RAM**: min. 4 GB (rekomendowane: 8 GB+)
- **Dysk**: min. 2 GB wolnego miejsca

### Kompilatory (dla Prophet/LightGBM)

**Linux/macOS:**
```bash
sudo apt-get install build-essential
```

**Windows:**
- Visual Studio Build Tools
- lub gotowe binaria Prophet/LightGBM

### Rekomendowane

- Wirtualne środowisko (venv/conda)
- Redis dla cache'owania (opcjonalne)
- CUDA dla GPU acceleration (opcjonalne)

---

## 📥 Instalacja

### 1. Klonowanie repozytorium

```bash
git clone https://github.com/<twoja_organizacja>/intelligent-predictor.git
cd intelligent-predictor
```

### 2. Utworzenie środowiska wirtualnego

**venv:**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

**conda:**
```bash
conda create -n intelligent-predictor python=3.11
conda activate intelligent-predictor
```

### 3. Instalacja pakietów

```bash
pip install -U pip
pip install -r requirements.txt
```

### 4. Konfiguracja zmiennych środowiskowych

Utwórz plik `.env` w głównym katalogu:

```env
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Database
DATABASE_URL=sqlite:///data/app.db

# Cache (opcjonalne)
REDIS_URL=redis://localhost:6379

# Vector Store
VECTOR_BACKEND=chroma
```

### 5. Uruchomienie

```bash
streamlit run app.py
```

Aplikacja będzie dostępna pod adresem: `http://localhost:8501`

---

## ⚙️ Konfiguracja

Plik `config.yaml` zawiera wszystkie konfiguracje aplikacji:

```yaml
# Ustawienia aplikacji
app:
  title: "Intelligent Predictor"
  theme: "dark"  # dark | light
  max_file_size_mb: 400
  allow_formats: ["csv", "xlsx", "json", "docx", "pdf"]
  session_timeout_minutes: 60

# Machine Learning
ml:
  test_size: 0.2
  random_state: 42
  default_metric_regression: "rmse"  # rmse | mae | r2
  default_metric_classification: "f1_weighted"  # accuracy | f1_weighted | precision | recall
  enable_shap: true
  cv_folds: 5

# Time Series
ts:
  horizon: 12
  frequency_guess: "auto"  # auto | D | W | M | Q | Y
  backtesting_folds: 3
  seasonality_mode: "multiplicative"  # additive | multiplicative
  growth: "linear"  # linear | logistic

# AI / OpenAI
ai:
  model: "gpt-4o-mini"  # gpt-4o-mini | gpt-4 | gpt-3.5-turbo
  temperature: 0.2
  max_tokens: 900
  timeout_seconds: 30

# Reports
reports:
  include_eda_summary: true
  include_model_cards: true
  include_data_sample: true
  output_format: "html"  # html | pdf
  template: "default"  # default | corporate | minimal

# Logging
logging:
  level: "INFO"  # DEBUG | INFO | WARNING | ERROR
  console: true
  file: true
  json: true
  retention_days: 30
  rotation_size_mb: 10
```

---

## 📁 Struktura projektu

```
intelligent-predictor/
│
├── 📄 app.py                      # Główny punkt wejścia
├── 📄 requirements.txt            # Zależności Python
├── 📄 config.yaml                 # Konfiguracja aplikacji
├── 📄 .env                        # Zmienne środowiskowe (nie commituj!)
├── 📄 README.md                   # Ten plik
├── 📄 LICENSE                     # Licencja MIT
├── 📄 Dockerfile                  # Konteneryzacja
├── 📄 .dockerignore
├── 📄 .gitignore
│
├── 📂 assets/                     # Zasoby statyczne
│   ├── styles/
│   │   └── custom.css             # Niestandardowe style
│   ├── images/
│   │   └── logo.png               # Logo aplikacji
│   └── templates/
│       └── report_template.html   # Szablon raportów
│
├── 📂 pages/                      # Strony Streamlit
│   ├── 1_📤_Upload_Data.py
│   ├── 2_🔍_EDA_Analysis.py
│   ├── 3_🤖_AI_Insights.py
│   ├── 4_📈_Predictions.py
│   ├── 5_📊_Forecasting.py
│   └── 6_📋_Reports.py
│
├── 📂 src/                        # Kod źródłowy
│   ├── 📂 ai_engine/              # Moduł AI
│   │   ├── openai_integrator.py
│   │   ├── insights_generator.py
│   │   └── report_generator.py
│   │
│   ├── 📂 data_processing/        # Przetwarzanie danych
│   │   ├── parsers.py
│   │   ├── cleaners.py
│   │   ├── feature_engineering.py
│   │   ├── validators.py
│   │   └── profiler.py
│   │
│   ├── 📂 ml_models/              # Modele ML
│   │   ├── automl.py
│   │   ├── time_series.py
│   │   ├── anomaly_detection.py
│   │   └── model_registry.py
│   │
│   ├── 📂 visualization/          # Wizualizacje
│   │   ├── charts.py
│   │   ├── dashboards.py
│   │   └── report_builder.py
│   │
│   ├── 📂 database/               # Bazy danych
│   │   ├── db_manager.py
│   │   ├── vector_store.py
│   │   └── cache_manager.py
│   │
│   └── 📂 utils/                  # Narzędzia pomocnicze
│       ├── logger.py
│       ├── helpers.py
│       └── validators.py
│
├── 📂 data/                       # Katalog danych
│   ├── raw/                       # Dane surowe
│   ├── processed/                 # Dane przetworzone
│   ├── exports/                   # Eksporty
│   └── app.db                     # Baza SQLite
│
├── 📂 tests/                      # Testy jednostkowe
│   ├── test_ai_engine.py
│   ├── test_data_processing.py
│   ├── test_ml_models.py
│   └── conftest.py
│
└── 📂 .github/                    # GitHub workflows
    └── workflows/
        └── ci.yml
```

---

## 🔄 Workflow w aplikacji

### 1️⃣ Upload Data
**Wczytaj dane w dowolnym formacie**
- Obsługa CSV, XLSX, JSON, DOCX, PDF
- Drag & drop lub file picker
- Walidacja i preview danych

### 2️⃣ EDA Analysis
**Automatyczna eksploracyjna analiza danych**
- Statystyki opisowe
- Raport profilujący (HTML)
- Wizualizacje rozkładów
- Macierze korelacji
- Analiza brakujących wartości

### 3️⃣ AI Insights
**Insighty biznesowe powered by GPT**
- Automatyczne wykrywanie wzorców
- Hipotezy biznesowe
- Rekomendacje działań
- Analiza trendów

### 4️⃣ Predictions (AutoML)
**Modelowanie predykcyjne**
- Automatyczny wybór algorytmu
- Trening na danych historycznych
- Walidacja krzyżowa
- Metryki wydajności
- SHAP interpretability

### 5️⃣ Forecasting
**Prognozowanie szeregów czasowych**
- Prophet z auto-tuningiem
- Wykrywanie sezonowości
- Pasma niepewności
- Backtesting
- Eksport prognoz

### 6️⃣ Reports
**Generowanie raportów**
- Kompletne raporty HTML
- Eksport ZIP z danymi
- Karty modeli
- Wszystkie wizualizacje
- Historia eksperymentów

---

## 🧪 Testy

Projekt zawiera kompletny zestaw testów jednostkowych:

### Uruchomienie testów

```bash
# Wszystkie testy
pytest

# Testy z pokryciem
pytest --cov=src --cov-report=html

# Testy z detalami
pytest -v

# Szybkie testy (bez Prophet)
pytest -m "not slow"
```

### Struktura testów

```python
tests/
├── test_ai_engine.py          # OpenAI, insights, raporty
├── test_data_processing.py    # Parsery, cleaning, FE
├── test_ml_models.py          # AutoML, forecasting, anomaly
└── conftest.py                # Fixtures i konfiguracja
```

### Pokrycie testami

| Moduł | Pokrycie |
|-------|----------|
| ai_engine | 85%+ |
| data_processing | 90%+ |
| ml_models | 80%+ |
| visualization | 75%+ |
| database | 85%+ |

---

## 🐳 Docker

### Build image

```bash
docker build -t intelligent-predictor:latest .
```

### Uruchomienie kontenera

```bash
docker run -d \
  --name intelligent-predictor \
  -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  intelligent-predictor:latest
```

### Docker Compose

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=sqlite:///data/app.db
    volumes:
      - ./data:/app/data
      - ./exports:/app/exports
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
```

**Uruchomienie:**
```bash
docker-compose up -d
```

---

## ⚡ CI/CD

### GitHub Actions

Projekt zawiera automatyczny workflow CI:

```yaml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          
      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov
          
      - name: Run tests
        run: pytest --cov=src --cov-report=xml
        
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

---

## 🔒 Bezpieczeństwo

### Prywatność danych

- ✅ Dane użytkownika pozostają lokalne
- ✅ Wysyłane do OpenAI: **tylko prompty** (bez danych wrażliwych)
- ✅ Brak zewnętrznych trackerów
- ✅ SQLite lokalnie - brak cloud storage

### Best Practices

1. **Nie commituj:**
   - `.env` (dodane do `.gitignore`)
   - Dane klientów (`data/` w `.gitignore`)
   - Eksporty z raportami
   - Klucze API

2. **Pseudonimizacja:**
   - Rozważ anonimizację w testach
   - Hashowanie identyfikatorów
   - Usuwanie PII przed raportami

3. **Zarządzanie sekretami:**
   - Plik `.env` dla lokalnego devu
   - GitHub Secrets dla CI/CD
   - Docker secrets dla produkcji
   - Streamlit Cloud secrets dla deploymentu

4. **Dependency scanning:**
   ```bash
   pip-audit  # Sprawdzanie luk w pakietach
   ```

---

## 🗺️ Roadmap & Wkład

### Planowane funkcje

- [ ] 🔗 Integracja MLflow (tracking eksperymentów)
- [ ] 🎯 Dodatkowe algorytmy anomalii (LOF, OCSVM)
- [ ] 📊 Template'y raportów branżowych (retail, manufacturing, finance)
- [ ] 🖥️ Tryb batch/CLI dla automatyzacji
- [ ] 🌐 Multi-language support (EN, PL, DE)
- [ ] 📱 Progressive Web App (PWA)
- [ ] 🔐 Autentykacja użytkowników
- [ ] 📤 Export do Power BI / Tableau
- [ ] 🤝 Kolaboracja w czasie rzeczywistym
- [ ] ☁️ Cloud deployment presets (AWS, Azure, GCP)

### Jak pomóc?

Chętnie przyjmujemy wkład społeczności! 🎉

1. **🐛 Zgłoś bug:**
   - Otwórz Issue z tagiem `bug`
   - Dodaj kroki do reprodukcji
   - Dołącz logi i zrzuty ekranu

2. **💡 Zaproponuj funkcję:**
   - Otwórz Issue z tagiem `enhancement`
   - Opisz use case i korzyści

3. **🔧 Dodaj kod:**
   ```bash
   # Fork repo
   git checkout -b feature/amazing-feature
   
   # Twoje zmiany + testy
   pytest
   
   # Commit z konwencją
   git commit -m "feat: add amazing feature"
   
   # Push i PR
   git push origin feature/amazing-feature
   ```

4. **📖 Popraw dokumentację:**
   - Typo w README?
   - Brakujący przykład?
   - Każda pomoc mile widziana!

### Konwencje

- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/)
- **Code style:** Black + isort + flake8
- **Tests:** pytest z min. 80% coverage dla nowego kodu
- **Docs:** docstrings w stylu Google

---

## 📄 Licencja

Projekt udostępniony na licencji **MIT License**.

```
MIT License

Copyright (c) 2024 Intelligent Predictor Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Pełna treść: [LICENSE](LICENSE)

---

## 🤝 Podziękowania

Projekt wykorzystuje następujące wspaniałe biblioteki open-source:

- [Streamlit](https://streamlit.io/) - Framework webowy
- [LightGBM](https://lightgbm.readthedocs.io/) - Gradient boosting
- [Prophet](https://facebook.github.io/prophet/) - Time series forecasting
- [OpenAI](https://openai.com/) - AI insights
- [Plotly](https://plotly.com/) - Interaktywne wykresy
- [ydata-profiling](https://github.com/ydataai/ydata-profiling) - Data profiling

---

## 📞 Kontakt & Wsparcie

- 🐛 **Issues:** [GitHub Issues](https://github.com/<twoja_organizacja>/intelligent-predictor/issues)
- 💬 **Dyskusje:** [GitHub Discussions](https://github.com/<twoja_organizacja>/intelligent-predictor/discussions)
- 📧 **Email:** support@intelligent-predictor.io
- 📚 **Dokumentacja:** [Wiki](https://github.com/<twoja_organizacja>/intelligent-predictor/wiki)

---

## 🌟 Pokaż wsparcie

Jeśli projekt Ci się podoba:

- ⭐ Zostaw **gwiazdkę** na GitHubie
- 🔀 **Fork** i experimentuj
- 🐛 Zgłoś **issue** jeśli znajdziesz bug
- 📢 **Podziel się** projektem

---

<div align="center">

**Zbudowano z ❤️ używając Python & Streamlit**

[⬆ Powrót na górę](#-intelligent-predictor)

</div>