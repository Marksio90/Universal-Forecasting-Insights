# DataGenius PRO — v9.1 ULTRA COMPLETE+

**Zawiera WSZYSTKO z Twoich wymagań:**
- Streamlit UI (role-aware, JWT) + Thematic Chat
- AutoML FUSION (Optuna + ensembling + SMOTE) + MLflow (Postgres) + MinIO
- FastAPI API: JWT/RBAC, batch `/predict`, **stream `/predict/stream`**
- Kolejka (Redis+RQ) do długich treningów
- **Monitoring driftu (APScheduler)** + alerty **Slack/Email**
- **Raporty PDF/HTML** (WeasyPrint/Kaleido)
- **Prosty Feature Store** (Parquet + wersjonowanie) z adapterem S3 (MinIO)
- **Nginx + SSL** reverse proxy
- **CI/CD** (GitHub Actions): lint smoke, build obrazów Dockera

## Quickstart DEV
```bash
mamba env create -f environment.yml
mamba activate datagenius_ultra
streamlit run app.py
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
## Full stack PROD
```bash
cp .env.example .env
./scripts/gen_self_signed.sh
docker compose -f infra/docker-compose.yml --profile prod up -d --build
```
