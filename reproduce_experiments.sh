#!/bin/bash
# conda env create -f environment.yml
# conda activate cyclesl
device=0 # device=-1 stands for CPU

for project in femnist celeba shakespeare
do
    for SL in PSL SFLV1 SFLV2 SGLR FedAvg CyclePSL CycleSFL CycleSGLR # SSL CycleSSL
    do
        for seed in 0 1 2 3 4
        do
            python main.py -P $project -SL $SL -seed $seed -device $device
        done
    done
done