.PHONY: all format lint lint_package help

# Default target executed when no arguments are given to make.
all: help

######################
# LINTING AND FORMATTING
######################

lint:
	python -m ruff check .
	python -m ruff format . --diff
	python -m ruff check --select I .
	python -m mypy --strict . --explicit-package-bases

lint_package:
	python -m ruff check src/
	python -m ruff format src/ --diff
	python -m ruff check --select I src/
	python -m mypy --strict src/ --explicit-package-bases

format:
	ruff format .
	ruff check --select I --fix .

######################
# HELP
######################

help:
	@echo '----'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'lint_package                 - lint only src/ directory'

