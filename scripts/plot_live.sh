#!/bin/bash

output_name=$1 # name of the controller
datafile_name=$2 # name of the datafile
plot_name=$3 # name of the plot

while true
do
	gnuplot -e "set terminal pdf font 'Arial,12' size 12cm, 8cm; \
				set output 'build/$output_name.pdf'; \
				datafile = 'build/$datafile_name.dat'; \
				load 'plots/$plot_name.plt'; \
				unset output"

	sleep 1
done
