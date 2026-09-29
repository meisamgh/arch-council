.PHONY: install dev test lint review-example

install:
	python -m pip install -e .

dev:
	python -m pip install -e '.[dev]'

test:
	pytest -q

lint:
	ruff check .

review-example:
	arch-council review --repo ../semantic_text2sql_ideal --question "Should planning and repair remain deterministic or use bounded agents?"
