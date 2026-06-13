.PHONY: setup pipeline test dashboard clean docker-dashboard docker-pipeline docker-quick docker-test

setup:
	pip install -r requirements.txt
	pip install -e .

pipeline:
	python scripts/run_full_pipeline.py

test:
	pytest tests/

dashboard:
	streamlit run streamlit_app.py

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf src/aac_adoption/__pycache__
	rm -rf reports/
	rm -rf models/
	rm -rf data/processed/

docker-dashboard:
	docker compose up dashboard

docker-pipeline:
	docker compose run --rm pipeline-full

docker-quick:
	docker compose run --rm pipeline-quick

docker-test:
	docker compose run --rm test
