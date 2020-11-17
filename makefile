SHELL := /bin/bash

# Free response
# ------------------------------------------------------------------------------

# Plot of the plant free response
build/response_free.pdf: plots/response_free.plt build/response_free.dat
	gnuplot -e "set terminal pdf font 'Sans,10' size 12cm, 8cm; \
				set output 'build/response_free.pdf'; \
				datafile = 'build/response_free.dat'; \
				load 'plots/response_free.plt'; \
				unset output"

# Plot of the plant free response on manuel's computer
build/response_free.manuel.pdf: plots/response_free.plt data/response_free.manuel.dat
	gnuplot -e "set terminal pdf font 'Sans,10' size 12cm, 8cm; \
				set output 'build/response_free.manuel.pdf'; \
				datafile = 'data/response_free.manuel.dat'; \
				load 'plots/response_free.plt'; \
				unset output"

# PID controller
# ------------------------------------------------------------------------------

# Plot of the plant silumation controlled by the PID controller
build/controller_pid.pdf: plots/controller_pid.plt build/controller_pid.dat
	gnuplot -e "set terminal pdf font 'Sans,10' size 12cm, 12cm; \
				set output 'build/controller_pid.pdf'; \
				datafile = 'build/controller_pid.dat'; \
				load 'plots/controller_pid.plt'; \
				unset output"

# Plot of the plant silumation controlled by the PID controller
# using Kp=20, Ki=5, Kd=5 on Manuel's computer
build/controller_pid.20_5_5.manuel.pdf: plots/controller_pid.plt data/controller_pid.20_5_5.manuel.dat
	gnuplot -e "set terminal pdf font 'Sans,10' size 12cm, 12cm; \
				set output 'build/controller_pid.20_5_5.manuel.pdf'; \
				datafile = 'data/controller_pid.20_5_5.manuel.dat'; \
				load 'plots/controller_pid.plt'; \
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
