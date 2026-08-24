# Master's Thesis

## Training

1. **ERM Training**:
```bash
dataset="urbancars"
seed=1
uv run main.py \
    --output_path logs \
    --experiment ERM \
    --dataset "$dataset" \
    --dataset_path "datasets/${dataset}" \
    --optimizer SGD \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --epochs 300 \
    --pretrained_path imagenet \
    --batch_size 128 \
    --seed "$seed"
```

2. **Edit images**:
```bash
dataset="urbancars"
editing_model="flux2-klein-4b"
uv run edit_images.py \
    --dataset "${dataset}" \
    --dataset_path datasets/${dataset} \
    --output_dir "datasets/${dataset}_edited_${editing_model}" \
    --batch_size 16 \
    --edit_model "${editing_model}"
```

3. **CDA**:
```bash
DATASET="celeba"
EDITING_MODEL="flux2_klein_4b"
SAMPLE_SIZE=128
BATCH_SIZE=128
LR=0.005
OPTIMIZER="adamW"
L1=1e-2
SEED=1
uv run main.py \
    --output_path logs \
    --dataset "$DATASET" \
    --dataset_path "datasets/${DATASET}" \
    --experiment CVA \
    --sample_size "$SAMPLE_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --pretrained_path "checkpoints/${DATASET}/erm_seed${SEED}/final_checkpoint.pt" \
    --l1 "$L1" \
    --epochs 100 \
    --optimizer "$OPTIMIZER" \
    --seed "$SEED" \
    --balanced_dataset_path "datasets/${DATASET}_edited_${EDITING_MODEL}"
```


Note: If the `--feature_only` flag is used, you should provide the pre-computed features of the specified dataset, which can be saved using the `save_features.py` file in the repository. If the flag is not specified, the raw image of the dataset should be provided. Here is an example script:

- For main datasets:
```bash
dataset="urbancars"
seed=1
uv run save_features.py \
    --dataset "$dataset" \
    --dataset_path "datasets/${dataset}" \
    --save_path "datasets_features/${dataset}/noaug_features_seed${seed}" \
    --pretrained_path "checkpoints/${dataset}/erm_seed${seed}/final_checkpoint.pt" \
    --batch_size 512 
```
- For Counterfactual Data Augmented Datasets:
```bash
dataset="urbancars"
sample_size=128
editing_model="flux2_klein_4b"
seed=1
uv run save_features.py \
    --dataset "$dataset" \
    --dataset_path "datasets/${dataset}_edited_${editing_model}" \
    --save_path "datasets_features/${dataset}_edited_${editing_model}_samples${sample_size}/noaug_features_seed${seed}" \
    --pretrained_path "checkpoints/${dataset}/erm_seed${seed}/final_checkpoint.pt" \
    --batch_size 512 \
    --cva True \
    --sample_size "$sample_size"
```

- CDA with `--feature_only` used:
```bash
DATASET="celeba"
EDITING_MODEL="flux2_klein_4b"
SAMPLE_SIZE=128
BATCH_SIZE=128
LR=0.005
OPTIMIZER="adamW"
L1=1e-2
SEED=1
uv run main.py \
    --output_path logs \
    --dataset "$DATASET" \
    --dataset_path "datasets_features/${DATASET}/noaug_features_seed${SEED}" \
    --experiment CVA \
    --sample_size "$SAMPLE_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --pretrained_path "checkpoints/${DATASET}/erm_seed${SEED}/final_checkpoint.pt" \
    --l1 "$L1" \
    --epochs 100 \
    --optimizer "$OPTIMIZER" \
    --seed "$SEED" \
    --balanced_dataset_path "datasets_features/${DATASET}_edited_${EDITING_MODEL}_samples${SAMPLE_SIZE}/noaug_features_seed${SEED}" \
    --feature_only True
```

## Datasets

## ERM Checkpoints

The ERM checkpoints for CelebA, Waterbirds, and UrbanCars are available [here](https://drive.google.com/file/d/1jGc9J4C_Ccy4P1WFfBG1-f4uuOfLzzk7/view?usp=drive_link).


## Acknowledgments

We would like to thank the authors of the following papers and repositories for their valuable contributions:

- [EVaLS](https://github.com/sharif-ml-lab/EVaLS)
- [Deep Feature Reweighting](https://github.com/PolinaKirichenko/deep_feature_reweighting)
- [Spurious Feature Learning](https://github.com/izmailovpavel/spurious_feature_learning)
- [WILDS](https://github.com/p-lambda/wilds)
- [DRO](https://github.com/kohpangwei/group_DRO/)
