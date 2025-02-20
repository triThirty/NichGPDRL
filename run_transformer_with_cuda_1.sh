#!/bin/bash

# Define lists of seeds and algorithms
seeds=(999 123 2025 30000 10000 20000 11000 23333 920083 908461 234815 882415 794609)
algos=("TransformerMTGP")     # Modify with your desired algorithm names
# algos=("transformerGP_all_gen_test")     # Modify with your desired algorithm names
# datasets=("HL" "LH" "LL")
datasets=("HH")

for dataset in "${datasets[@]}"; do
    for algo in "${algos[@]}"; do
        for seed in "${seeds[@]}"; do
            echo "Running: python main.py $dataset $seed $algo"
            CUDA_VISIBLE_DEVICES=2 python main.py "$dataset" "$seed" "$algo"
        done
    done
done
echo "All done!"
