#!/bin/bash

output_dir=$1 # path of the output directory
realisation_time=$2 # time length of each realisation (s)
realisation_n=$3 # number of realisations

i=0

while [[ $i -lt $realisation_n ]]
do
	python cleave.py run-controller examples/controller_pid_config.py &
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

	mv build/controller_pid.dat $output_dir/realisation_$i.dat

	echo

	(( i++ ))
done
