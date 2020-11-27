#!/bin/bash

controller_name=$1 # name of the controller
realisation_time=$2 # time length of each realisation (s)

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

echo
