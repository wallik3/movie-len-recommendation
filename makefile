ifeq ($(OS),Windows_NT)
	SETENV = set
	SEP = &
else
	SETENV = export
	SEP = &&
endif

run-server:
	$(SETENV) PYTHONPATH=. $(SEP) python app.py

docker-build:
	docker build -t movie-len-recommendation .

docker-run:
	docker run -p 5000:5000 movie-len-recommendation