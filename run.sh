#!/bin/bash

# Define lists of seeds and algorithms
# seeds=(2 4 8 16 32 64 128 256 512 1024)         # Modify with your desired seed values
# seeds=(2 4 9 32 256)         # Modify with your desired seed values
seeds=(2)         # Modify with your desired seed values
algos=("TransformerMTGP" "transformerGP_all_gen_test")     # Modify with your desired algorithm names

# Iterate over each seed and algorithm combination
for seed in "${seeds[@]}"; do
    for algo in "${algos[@]}"; do
        echo "Running: python main.py HH $seed $algo"
        python main.py HH "$seed" "$algo"
    done
done
