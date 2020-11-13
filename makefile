SHELL := /bin/bash

# Install
# ------------------------------------------------------------------------------
install:
	virtualenv --python=python3.8 ./venv
	source venv/bin/activate; \
	pip install -Ur requirements.txt
	mkdir build

# Clean
# ------------------------------------------------------------------------------
clean:
	rm -r venv
	rm -r build
	rm -r plant_metrics
	rm -r controller_metrics
