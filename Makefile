SHELL := /bin/bash
PYTHON ?= python
PIP ?= pip
NAME := geobias
PACKAGE_NAME := geobias
VERSION := 1.0.0
DIST := dist
UV ?= uv

env:
	$(UV) venv --python=3.12 .venv --clear
	. .venv/bin/activate && $(PYTHON) -m ensurepip --upgrade && $(PYTHON) -m $(PIP) install uv --upgrade && $(UV) $(PIP) install setuptools wheel
	# Manually activate env. Does not work with make

install:
	$(UV) sync

check:
	pre-commit run --all-files

install-dev:
	$(PIP) install -e ".[dev]"
	pre-commit install

clean-build:
	rm -rf ${DIST}