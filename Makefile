PY ?= python

.PHONY: setup data reproduce test lint nightly all

## Install pinned dependencies and the package (editable)
setup:
	$(PY) -m pip install -r requirements-lock.txt
	$(PY) -m pip install -e . --no-deps

## Download the city's snapshot (and the small portal files used by the fairness audit)
data:
	$(PY) scripts/get_data.py --portal

## Rebuild every number, table and chart in the README
reproduce:
	$(PY) scripts/reproduce.py --update-readme

## Unit tests for the metric and features
test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

## Extension C: one run of the nightly live-scoring job
nightly:
	$(PY) scripts/nightly.py

all: data reproduce test
