#!/bin/bash

controller_name=$1 # name of the controller
output_dir=$2 # path of the output directory
realisation_time=$3 # time length of each realisation (s)
realisation_n=$4 # number of realisations

i=0

while [[ $i -lt $realisation_n ]]
do
	python cleave.py run-controller examples/controller_$controller_name_config.py &
	controller_pid=$!

	sleep 1

	python cleave.py run-plant examples/plant_config.py &
	plant_pid=$!

	echo realisation_i = $i
	echo controller_pid = $controller_pid
	echo plant_pid = $plant_pid

	sleep $realisation_time

	kill $controller_pid
	kill $plant_pid

	cp build/controller_$controller_name.dat $output_dir/realisation_$(printf "%03d" $i).dat

	echo

	(( i++ ))
done
