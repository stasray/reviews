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

.PHONY: test test-cov test-cov-html clean

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
	rm -rf .pytest_cache .coverage .coverage_html coverage.xml
