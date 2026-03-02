.PHONY: install run test lint format clean docker-build docker-run


install:
	pip install -r requirements.txt
	cp -n .env.example .env || true
	@echo "✅  Dependencies installed. Edit .env and add your GROQ_API_KEY."


run:
	streamlit run src/ui/app.py

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=src --cov-report=html
	@echo "Coverage report: htmlcov/index.html"


lint:
	flake8 src/ tests/ --max-line-length=100

format:
	black src/ tests/ --line-length 100
	isort src/ tests/ --profile black

docker-build:
	docker build -t rag-chatbot:latest .

docker-run:
	docker-compose up --build

docker-down:
	docker-compose down


clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
	@echo "🧹  Clean done."
