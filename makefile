SHELL := /bin/bash

# PID controller
# ------------------------------------------------------------------------------

# Plot of the plant silumation controlled by the PID controller
build/controller_pid.pdf: plots/controller_pid.plt build/controller_pid.dat
	gnuplot -e "set terminal pdf font 'Sans,10' size 12cm, 12cm; \
				set output 'build/controller_pid.pdf'; \
				load 'plots/controller_pid.plt'; \
				unset output"

# Plot of the plant silumation controlled by the PID controller
build/response_free.pdf: plots/response_free.plt build/response_free.dat
	gnuplot -e "set terminal pdf font 'Sans,10' size 12cm, 8cm; \
				set output 'build/response_free.pdf'; \
				load 'plots/response_free.plt'; \
				unset output"

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
