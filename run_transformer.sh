#!/bin/bash

# Define lists of seeds and algorithms
seeds=(0 1 2 4 8 16 32 40 64 128 256 512 1024 10 20 30 50 42 0 999 123 2025 30000 10000 20000 11000 23333 920083 908461 234815 882415 794609)
algos=("TransformerMTGP" "transformerGP_all_gen_test")     # Modify with your desired algorithm names
datasets=("HH" "HL" "LH" "LL")

for dataset in "${datasets[@]}"; do
    for algo in "${algos[@]}"; do
        for seed in "${seeds[@]}"; do
            echo "Running: python main.py $dataset $seed $algo"
            python main.py "$dataset" "$seed" "$algo" &
        done
        wait
    done
    wait
done
wait
echo "All done!"