.PHONY: setup pipeline test dashboard clean

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
