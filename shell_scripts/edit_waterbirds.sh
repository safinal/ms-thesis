uv run edit_images.py \
    --dataset waterbirds \
    --dataset_path datasets/waterbird_complete95_forest2water2 \
    --output_dir datasets/waterbirds_edited_flux2_klein_4b \
    --batch_size 16 \
    --edit_model flux2-klein-4b
uv run edit_images.py \
    --dataset waterbirds \
    --dataset_path datasets/waterbird_complete95_forest2water2 \
    --output_dir datasets/waterbirds_edited_flux2_klein_9b \
    --batch_size 16 \
    --edit_model flux2-klein-9b