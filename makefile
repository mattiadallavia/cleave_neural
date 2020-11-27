SHELL := /bin/bash

# Free response
# ------------------------------------------------------------------------------

# Plot of the plant free response
build/response_free.pdf: plots/response_free.plt build/response_free.dat
	gnuplot -e "set terminal pdf font 'Arial,11' size 12cm, 8cm; \
				set output 'build/response_free.pdf'; \
				datafile = 'build/response_free.dat'; \
				load 'plots/response_free.plt'; \
				unset output"

# Plot of the plant free response under high load
build/response_free.stress.pdf: plots/response_free.plt data/response_free.stress.dat
	gnuplot -e "set terminal pdf font 'Arial,11' size 12cm, 8cm; \
				set output 'build/response_free.stress.pdf'; \
				datafile = 'data/response_free.stress.dat'; \
				load 'plots/response_free.plt'; \
				unset output"

# Plot of the plant free response under low load
build/response_free.smooth.pdf: plots/response_free.plt data/response_free.smooth.dat
	gnuplot -e "set terminal pdf font 'Arial,11' size 12cm, 8cm; \
				set output 'build/response_free.smooth.pdf'; \
				datafile = 'data/response_free.smooth.dat'; \
				load 'plots/response_free.plt'; \
				unset output"

# PID controller
# ------------------------------------------------------------------------------

# Plot of the plant silumation controlled by the PID controller
# without artificial noise
build/controller_pid.pdf: plots/controller_pid.plt build/controller_pid.dat
	gnuplot -e "set terminal pdf font 'Arial,11' size 12cm, 8cm; \
				set output 'build/controller_pid.pdf'; \
				datafile = 'build/controller_pid.dat'; \
				load 'plots/controller_pid.plt'; \
				unset output"

# Plot of the plant silumation controlled by the PID controller
# with artificial noise
build/controller_pid_noisy.pdf: plots/controller_pid.plt build/controller_pid_noisy.dat
	gnuplot -e "set terminal pdf font 'Arial,11' size 12cm, 8cm; \
				set output 'build/controller_pid_noisy.pdf'; \
				datafile = 'build/controller_pid_noisy.dat'; \
				load 'plots/controller_pid.plt'; \
				unset output"

# Plot of the plant silumation controlled by the PID controller
# without artificial noise on remote computer
build/controller_pid.noiseless.pdf: plots/controller_pid.plt data/controller_pid.noiseless.dat
	gnuplot -e "set terminal pdf font 'Arial,11' size 12cm, 8cm; \
				set output 'build/controller_pid.noiseless.pdf'; \
				datafile = 'data/controller_pid.noiseless.dat'; \
				load 'plots/controller_pid.plt'; \
				unset output"

# Plot of the plant silumation controlled by the PID controller
# with artificial noise (var = 0.001 N^2) on remote computer
build/controller_pid.noisy.pdf: plots/controller_pid.plt data/controller_pid.noisy.dat
	gnuplot -e "set terminal pdf font 'Arial,11' size 12cm, 8cm; \
				set output 'build/controller_pid.noisy.pdf'; \
				datafile = 'data/controller_pid.noisy.dat'; \
				load 'plots/controller_pid.plt'; \
				unset output"

# Plot of the plant silumation controlled by the PID controller
# with artificial noise (var = 100 N^2) on remote computer
build/controller_pid.very_noisy.pdf: plots/controller_pid.plt data/controller_pid.very_noisy.dat
	gnuplot -e "set terminal pdf font 'Arial,11' size 12cm, 8cm; \
				set output 'build/controller_pid.very_noisy.pdf'; \
				datafile = 'data/controller_pid.very_noisy.dat'; \
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
