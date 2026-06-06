.PHONY: install profile train export api test clean docker-train docker-api

# ── Local development ──────────────────────────────────────────────────────────

install:
	python -m pip install -r requirements.txt -r requirements-api.txt

profile:
	PYTHONPATH=src python -m insiders_loyalty_program.cli profile

train:
	PYTHONPATH=src python -m insiders_loyalty_program.cli train

export:
	python scripts/export_clusters_to_sqlite.py

api:
	PYTHONPATH=src uvicorn insiders_loyalty_program.api:app --reload --host 0.0.0.0 --port 8000

test:
	python -m pytest -v

clean:
	rm -f models/model.joblib reports/metrics.json reports/cluster_assignments.csv
	rm -f reports/insiders_segments.sqlite

# ── Docker ─────────────────────────────────────────────────────────────────────

docker-train:
	docker compose run --rm train

docker-api:
	docker compose up api

# ── Full local pipeline ────────────────────────────────────────────────────────

all: train export
	@echo "Pipeline complete. Results in reports/"
