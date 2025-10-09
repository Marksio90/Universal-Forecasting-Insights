# ================================================================================================
# INTELLIGENT PREDICTOR PRO++++ — Makefile
# Common development commands
# ================================================================================================

.PHONY: help install install-dev test lint format clean docker-build docker-run docs

# Default target
.DEFAULT_GOAL := help

# Colors
BLUE := \033[0;34m
GREEN := \033[0;32m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Intelligent Predictor PRO++++ - Make Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt
	@echo "$(GREEN)✓ Production dependencies installed$(NC)"

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	pip install --upgrade pip setuptools wheel
	pip install -r requirements-dev.txt
	pre-commit install
	@echo "$(GREEN)✓ Development dependencies installed$(NC)"

test: ## Run tests with coverage
	@echo "$(BLUE)Running tests...$(NC)"
	pytest --cov=src --cov-report=html --cov-report=term -v
	@echo "$(GREEN)✓ Tests completed. Coverage report: htmlcov/index.html$(NC)"

test-fast: ## Run fast tests only
	@echo "$(BLUE)Running fast tests...$(NC)"
	pytest -m "not slow" -v
	@echo "$(GREEN)✓ Fast tests completed$(NC)"

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	pytest-watch

lint: ## Run linters
	@echo "$(BLUE)Running linters...$(NC)"
	flake8 src tests
	pylint src
	mypy src --ignore-missing-imports
	bandit -r src
	@echo "$(GREEN)✓ Linting completed$(NC)"

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	black src tests
	isort src tests
	@echo "$(GREEN)✓ Code formatted$(NC)"

clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build dist .pytest_cache .coverage htmlcov .mypy_cache .tox
	@echo "$(GREEN)✓ Cleaned$(NC)"

run: ## Run the application
	@echo "$(BLUE)Starting Intelligent Predictor...$(NC)"
	streamlit run app.py

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t intelligent-predictor:latest .
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -d -p 8501:8501 --name intelligent-predictor intelligent-predictor:latest
	@echo "$(GREEN)✓ Container running on http://localhost:8501$(NC)"

docker-stop: ## Stop Docker container
	@echo "$(BLUE)Stopping Docker container...$(NC)"
	docker stop intelligent-predictor
	docker rm intelligent-predictor
	@echo "$(GREEN)✓ Container stopped$(NC)"

docker-compose-up: ## Start services with docker-compose
	@echo "$(BLUE)Starting services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ Services started$(NC)"

docker-compose-down: ## Stop services with docker-compose
	@echo "$(BLUE)Stopping services...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Services stopped$(NC)"

docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	cd docs && make html
	@echo "$(GREEN)✓ Documentation built: docs/_build/html/index.html$(NC)"

docs-serve: ## Serve documentation
	@echo "$(BLUE)Serving documentation...$(NC)"
	cd docs/_build/html && python -m http.server 8000

health-check: ## Run system health check
	@echo "$(BLUE)Running health check...$(NC)"
	python scripts/health_check.py
	@echo "$(GREEN)✓ Health check completed$(NC)"

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	pytest tests/test_performance.py --benchmark-only
	@echo "$(GREEN)✓ Benchmarks completed$(NC)"

security-scan: ## Run security vulnerability scan
	@echo "$(BLUE)Running security scan...$(NC)"
	safety check
	bandit -r src
	pip-audit
	@echo "$(GREEN)✓ Security scan completed$(NC)"

update-deps: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	pip-compile requirements.in
	pip-compile requirements-dev.in
	@echo "$(GREEN)✓ Dependencies updated$(NC)"

init-db: ## Initialize database
	@echo "$(BLUE)Initializing database...$(NC)"
	python scripts/seed_db.py
	@echo "$(GREEN)✓ Database initialized$(NC)"

backup-db: ## Backup database
	@echo "$(BLUE)Backing up database...$(NC)"
	python scripts/backup.py
	@echo "$(GREEN)✓ Database backed up$(NC)"

all: clean install-dev lint test ## Run all checks
	@echo "$(GREEN)✓ All checks passed!$(NC)"