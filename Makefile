PY := .venv/bin/python

.PHONY: venv test mock estimate ladder pilot full clean

venv:
	python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt pytest

test:
	$(PY) -m pytest tests -q

# Whole pipeline on synthetic data. No API key, no spend.
mock:
	$(PY) -m seedstudy run --mock --out runs/mock

# Count the API calls the ladder design implies. Calls nothing.
estimate:
	$(PY) -m seedstudy estimate --ladder

# The lean study: one sweep, four analyses. ~480 artifacts per modality, ~$3.
ladder:
	$(PY) -m seedstudy run --ladder

# Small real run: 1 model per modality, 2 prompts, 5 seeds.
pilot:
	$(PY) -m seedstudy run --config configs/pilot.json

full:
	$(PY) -m seedstudy run --config configs/full.json

clean:
	rm -rf runs .pytest_cache **/__pycache__
