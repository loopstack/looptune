#!/bin/bash

sh ../../../kill_autotune_processes.sh

iter=$1
cmd=${@:2}

rm pids.txt

for (( c=1; c<=$iter; c++ ))
do 
    echo $c $cmd
    $cmd
    echo $! >> pids.txt
done


