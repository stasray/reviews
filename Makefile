# Simple test runner for this repo

SHELL := /bin/bash

# Prefer virtualenv tools if present
PY := python3
PYTEST := pytest
COVERAGE := coverage

ifneq (,$(wildcard .venv/bin/python))
  PY := .venv/bin/python
  PYTEST := .venv/bin/pytest
  COVERAGE := .venv/bin/coverage
endif

.PHONY: test test-cov test-cov-html profile clean

test:
	$(PYTEST) -q

test-cov:
	$(COVERAGE) run -m pytest
	$(COVERAGE) report -m
	$(COVERAGE) xml -o coverage.xml

test-cov-html:
	$(COVERAGE) run -m pytest
	$(COVERAGE) html -d .coverage_html
	@echo "HTML coverage: .coverage_html/index.html"

clean:
	rm -rf .pytest_cache .coverage .coverage_html coverage.xml profile_results.json

profile:
	$(PY) scripts/profile_analysis.py --sizes "50,200,500"

.PHONY: web-perf
web-perf:
	$(PY) scripts/web_perf_test.py --scenario perfomance/scenario_example.json --rate 5 --concurrency 5 --duration 5
