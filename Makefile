.PHONY: up up-prod down api ui worker gen-cert
up:
	docker compose -f infra/docker-compose.yml --profile dev up -d --build
up-prod:
	docker compose -f infra/docker-compose.yml --profile prod up -d --build
down:
	docker compose -f infra/docker-compose.yml down -v
api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000
ui:
	streamlit run app.py
worker:
	python queue/worker.py
gen-cert:
	bash scripts/gen_self_signed.sh
