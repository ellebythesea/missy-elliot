PY=python3
PIP=$(PY) -m pip
VENV=.venv
ACTIVATE=$(VENV)/bin/activate

.PHONY: help venv install run clean deps-tools lock

help:
	@echo "Common tasks:"
	@echo "  make venv      # Create virtualenv"
	@echo "  make install   # Install requirements"
	@echo "  make run       # Run Streamlit app"
	@echo "  make clean     # Remove virtualenv"
	@echo "  make deps-tools# Install pip-tools for pinning"
	@echo "  make lock      # Compile pinned requirements.txt from requirements.in"

venv:
	$(PY) -m venv $(VENV)
	. $(ACTIVATE) && $(PIP) install --upgrade pip

install: venv
	. $(ACTIVATE) && $(PIP) install -r requirements.txt

run:
	. $(ACTIVATE) && streamlit run app.py

deps-tools: venv
	. $(ACTIVATE) && $(PIP) install pip-tools

lock: deps-tools
	. $(ACTIVATE) && pip-compile --upgrade requirements.in

clean:
	rm -rf $(VENV)
