uv run edit_images.py \
    --dataset celeba \
    --dataset_path datasets/celeba-dataset \
    --output_dir datasets/celeba_edited_flux2_klein_4b \
    --batch_size 16 \
    --edit_model flux2-klein-4b
uv run edit_images.py \
    --dataset celeba \
    --dataset_path datasets/celeba-dataset \
    --output_dir datasets/celeba_edited_flux2_klein_9b \
    --batch_size 16 \
    --edit_model flux2-klein-9b

