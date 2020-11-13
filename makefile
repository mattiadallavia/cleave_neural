SHELL := /bin/bash

# Install
# ------------------------------------------------------------------------------
install:
	virtualenv --python=python3.8 ./venv
	source venv/bin/activate; \
	pip install -Ur ./requirements.txt

# Clean
# ------------------------------------------------------------------------------
clean:
	rm -r venv
