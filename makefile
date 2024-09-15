ifeq ($(OS),Windows_NT)
	SETENV = set
	SEP = &
else
	SETENV = export
	SEP = &&
endif

run-server:
	$(SETENV) PYTHONPATH=. $(SEP) python app.py
