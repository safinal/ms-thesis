#!/bin/bash

# Define the L1 regularization values
L1_VALUES=(0.00 1e-2 1e-3 1e-4 1e-5)

# Loop over seeds 1 through 5
for SEED in {1..5}; do
    echo "Starting runs for seed ${SEED}..."
    
    # Loop over each L1 value
    for L1 in "${L1_VALUES[@]}"; do
        uv run main.py \
            --output_path logs \
            --dataset celeba \
            --dataset_path "datasets_features/celeba/noaug_features_seed${SEED}" \
            --experiment DFR \
            --sample_size 128 \
            --batch_size 512 \
            --learning_rate 0.01 \
            --pretrained_path "checkpoints/dfr-ckpts/celeba/erm_seed${SEED}/final_checkpoint.pt" \
            --l1 "${L1}" \
            --epochs 100 \
            --optimizer adam \
            --step_size 30 \
            --seed "${SEED}" \
            --feature_only True
    done
done

echo "All runs completed!"