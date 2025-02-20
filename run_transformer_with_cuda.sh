#!/bin/bash

# Define lists of seeds and algorithms
seeds=(0 1 2 4 8 16 32 40 64 128 256 512 1024 10 20 30 50 42)
algos=("TransformerMTGP")     # Modify with your desired algorithm names
# algos=("transformerGP_all_gen_test")     # Modify with your desired algorithm names
# datasets=("HL" "LH" "LL")
datasets=("HH")

for dataset in "${datasets[@]}"; do
    for algo in "${algos[@]}"; do
        for seed in "${seeds[@]}"; do
            echo "Running: python main.py $dataset $seed $algo"
            CUDA_VISIBLE_DEVICES=1 python main.py "$dataset" "$seed" "$algo"
        done
    done
done
echo "All done!"
