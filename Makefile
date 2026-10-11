.PHONY: install install-dev lint test test-unit test-leakage clean run-all

PYTHON := python

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/ configs/

format:
	ruff format src/ tests/

test:
	pytest tests/

test-unit:
	pytest tests/unit/

test-leakage:
	pytest tests/leakage/

test-portfolio:
	pytest tests/portfolio/

test-integration:
	pytest tests/integration/

generate-synthetic:
	$(PYTHON) -m alpha_volatility.cli generate-synthetic-data --config configs/data.yaml

build-features:
	$(PYTHON) -m alpha_volatility.cli build-features --config configs/features.yaml

train-tcn:
	$(PYTHON) -m alpha_volatility.cli train --config configs/experiment.yaml --model tcn

train-transformer:
	$(PYTHON) -m alpha_volatility.cli train --config configs/experiment.yaml --model transformer

train-lightgbm:
	$(PYTHON) -m alpha_volatility.cli train --config configs/experiment.yaml --model lightgbm

evaluate:
	$(PYTHON) -m alpha_volatility.cli evaluate --config configs/experiment.yaml

backtest:
	$(PYTHON) -m alpha_volatility.cli backtest --config configs/portfolio.yaml

report:
	$(PYTHON) -m alpha_volatility.cli report --config configs/experiment.yaml

run-all: generate-synthetic build-features train-tcn train-transformer train-lightgbm evaluate backtest report

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov
