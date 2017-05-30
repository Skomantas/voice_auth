python ?= python3
pyenv := pyenv


.PHONY: pyenv
pyenv: requirements.txt
	$(python) -m venv $(pyenv)
	$(pyenv)/bin/pip install -r $^
