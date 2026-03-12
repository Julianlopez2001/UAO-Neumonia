install:
	pip install -r requirements.txt

test:
	python -m pytest -v

run:
	python app.py

ui:
	python detector_neumonia.py

check-model:
	python check_model.py

docker-build:
	docker build -t uaoneumonia .

docker-run:
	docker run uaoneumonia