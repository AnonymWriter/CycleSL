#!/bin/bash

# conda env create -f environment.yml
# conda activate cyclesl

device=0 # device=-1 stands for CPU

# for FEMNIST, CelebA, and Shakespeare
for project in femnist celeba shakespeare
do
    for SL in SFLV1 SFLV2 CycleSFL PSL SGLR FedAvg CyclePSL CycleSGLR # SSL CycleSSL
    do
        for seed in 0 1 2 3 4
        do
            python main.py -P $project -SL $SL -seed $seed -device $device
        done
    done
done

# for CIFAR-100
project=cifar100
for cda in -1.0 1.0 0.5 0.1  # cda: alpha value for Dirichlet distribution for non-iid partition. -1.0 stands for iid partition
do
    for SL in SFLV1 SFLV2 CycleSFL PSL SGLR FedAvg CyclePSL CycleSGLR # SSL CycleSSL
    do
        for seed in 0 1 2 3 4
        do
            python main.py -P $project -SL $SL -device $device -seed $seed -cda $cda -T 1001
        done
    done
done

# ablation study: cut layer
for cda in -1.0 1.0 0.5 0.1
do
    for cut in 1 2 3 4 5 6
    do
        for seed in 0 1 2 3 4
        do
            python main.py -P cifar100 -SL CycleSFL -device $device -seed $seed -cda $cda --cut_layer $cut -T 1001
        done
    done
done

# ablation study: server epoch
for cda in -1.0 1.0 0.5 0.1
do
    for epoch in 1 2 4 8
    do
        for seed in 0 1 2 3 4
        do
            python main.py -P cifar100 -SL CycleSFL -device $device -seed $seed -cda $cda --num_server_epoch $epoch -T 1001
        done
    done
done