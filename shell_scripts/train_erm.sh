for SEED in {1..5}; do
    echo "Starting runs for seed ${SEED}..."
    uv run main.py \
        --output_path logs \
        --experiment ERM \
        --dataset urbancars \
        --dataset_path datasets/urbancars_images \
        --optimizer SGD \
        --learning_rate 1e-3 \
        --step_size 100 \
        --weight_decay 1e-4 \
        --gamma 1.0 \
        --epochs 300 \
        --pretrained_path imagenet \
        --batch_size 128 \
        --seed "${SEED}"
done