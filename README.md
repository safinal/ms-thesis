# Master's Thesis

## Requirements

Our code requires Python 3.10 or higher to run successfully.
Please use either `requirements.txt` with `pip` or `env.yml` with `conda` to create envs and install dependencies.

## Datasets

The following datasets are supported: Waterbirds, CelebA, MultiNLI, Domino, Colored MNIST (CMNIST), CivilComments, and UrbanCars.

Please follow the instructions in the [Data Access](#data-access) section to set up the datasets.

## Training

1. **ERM Training**:
* Waterbirds example
```bash
python main.py --root_dir ./ --experiment ERM --dataset waterbirds --dataset_path /path/to/waterbird_complete95_forest2water2 --optimizer SGD -lr 1e-3 --step_size 100 --weight_decay 1e-4 --gamma 0.5 --epochs 300 --pretrained_path imagenet -b 128
```

* CivilComments and MultiNLI
 The ERM models were trained using the code provided by [this repo](https://github.com/izmailovpavel/spurious_feature_learning).

2. **EVaLS-GL**:
* Celeba example
```bash
python main.py --dataset domino --dataset_path /path/to/data/celeba --experiment loss --sample_size 5 -b 32 -lr 0.0005 --pretrained_path /path/to/resnet50.model --gamma 0.1 --weight_decay 0 --l1 0 --epochs 100 --optimizer adam --step_size 85 --seed 0 --feature_only True
```

3. **EVaLS**:
   Prior to running evals, you should run `EIIL.py`. Here is an example script:
```bash
python EIIL.py --dataset urbancars --dataset_path path/to/urbancars/noaug_features_seed0 --learning_rate 0.01 --num_steps 20000 --batch_size 128 --feature_only True --save_path path/to/save/urbancars/seed0/ --pretrained_path path/to/ckpts/urbancars/erm_seed0/ckpt.pth
```
This will create the new validation environments in the `validation_path`.

* urbancars example 
```bash
python3 main.py --dataset urbancars --dataset_path /path/to/data/urbancars/noaug_features_seed0 --experiment loss --sample_size 10 -b 32 -lr 0.0005 --pretrained_path /path/to/ckpts/urbancars/erm_seed0/ckpt.pth --gamma 0.1 --weight_decay 0 --l1 0 --epochs 100 --optimizer adam --step_size 85 --seed 0 --feature_only True --validation_path /path/to/validation_groups/urbancars/seed1 --for_free True
```

Note: If the `--feature_only` flag is used, you should provide the pre-computed features of the specified dataset, which can be saved using the `save_features.py` file in the repository. If the flag is not specified, the raw image or text files of the dataset should be provided. Here is an example script:

```bash
python3 save_features.py --dataset civilcomments --dataset_path path/to/data/civilcomments --save_path path/to/save/civilcomments/seed1/ --pretrained_path path/to/civilcomments/erm_seed1 --batch_size 64
```

## Data Access

### Waterbirds and CelebA

Follow the instructions in the [DFR repo](https://github.com/PolinaKirichenko/deep_feature_reweighting#data-access) to prepare the Waterbirds and CelebA datasets.

### CelebA

Our code expects the following files/folders in the `[root_dir]/celebA` directory:

- `data/celeba_metadata.csv`
- `data/img_align_celeba/`

You can download these dataset files from [this Kaggle link](https://www.kaggle.com/jessicali9530/celeba-dataset).

Make sure to move `metadata/celeba_metadata.csv`, which contains last layer split, to the data directory.

### Waterbirds

Our code expects the following files/folders in the `[root_dir]/` directory:

- `data/waterbird_complete95_forest2water2/`

You can download a tarball of this dataset [here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz).

Make sure to move `metadata/wb_metadata.csv`, which contains last layer split, to the directory and rename it to `metadata.csv`.

### Civil Comments and MultiNLI

For the CivilComments dataset, we have altered the split column. The version with the last layer split can be downloaded from [this link](https://mega.nz/file/HexUHASa#ktfZdoi_EG5tjrQ25VJQW56PksE9wVRY29rvL20arM4).

To run experiments on the MultiNLI dataset, please manually download and unzip the dataset from [this link](https://nlp.stanford.edu/data/dro/multinli_bert_features.tar.gz). Further, copy the `utils_glue.py` to the root directory of the dataset, and add the metadata with the last layer split from `metadata/multinli/metadata_random.csv` to the dataset directory.

### Dominoes-CMF

You can create and access our modified dominoes dataset in `notebooks/dominoes.ipynb`.

### UrbanCars

For the Urbancars dataset, please refer to [Whac-A-Mole](https://github.com/facebookresearch/Whac-A-Mole/blob/main/README.md#urbancars-experiments) repo. As it is time-consuming and a bit challenging to create the whole dataset, we have uploaded the urbancars images [here](https://mega.nz/file/uaoACADa#pgEH6j8vIL0U6Ys1S5N4O8BPKQiA3b9Ly6rikIFxxyw) for the ease of access and usage.

## Usage
To run an experiment, use the `main.py` script with the appropriate arguments:

```bash
python main.py [--root_dir ROOT_DIR] [--learning_rate LEARNING_RATE] [--optimizer {adam,adamW,SGD}]
               --experiment {ERM,DFR,loss,cluster,entropy,gradcam} --dataset {waterbirds,celeba,multinli,domino,cmnist,civilcomments,metashift,urbancars}
               --dataset_path DATASET_PATH [--comments COMMENTS] [--output_path OUTPUT_PATH] [--bert_ckpt BERT_CKPT]
               [--sample_size SAMPLE_SIZE] [--weight_decay WEIGHT_DECAY] [--l1 L1] [--step_size STEP_SIZE] [--gamma GAMMA]
               [--epochs EPOCHS] [--model {ResNet,BERT}] [--pretrained_path PRETRAINED_PATH] [--batch_size BATCH_SIZE]
               [--num_workers NUM_WORKERS] [--test_only TEST_ONLY] [--log LOG] [--for_free FOR_FREE] [--seed SEED]
               [--random_grouping RANDOM_GROUPING] [--feature_only FEATURE_ONLY] [--num_val NUM_VAL]
               [--fine_tune FINE_TUNE] [--early_stop_val EARLY_STOP_VAL] [--validation_path VALIDATION_PATH]
               [--saved_val SAVED_VAL]
```

### Arguments

- `--root_dir`: Path to the root directory of the project (default: `None`).
- `--learning_rate`, `-lr`: Learning rate for the optimizer (default: `0.001`).
- `--optimizer`: Type of optimizer (choices: `adam`, `adamW`, `SGD`; default: `adam`).
- `--experiment`: Type of experiment (choices: `ERM`, `DFR`, `loss`, `cluster`, `entropy`, `gradcam`; required). `loss` is equivalent to EVaLS.
- `--dataset`: Name of the dataset (choices: `waterbirds`, `celeba`, `multinli`, `domino`, `cmnist`, `civilcomments`, `metashift`, `urbancars`; required).
- `--dataset_path`: Path of the dataset (default: `./waterbird_complete_forest2water2`).
- `--comments`: Additional comments to be included in the log name (default: `''`).
- `--output_path`: Path for logs and checkpoints (default: `/home/logs/`).
- `--bert_ckpt`: Weights of pre-trained BERT for tokenization (default: `bert-base-uncased`).
- `--sample_size`: **Sample size** of each group in the experiment (default: `64`).
- `--weight_decay`: Weight decay coefficient for L2 regularization (default: `0`).
- `--l1`: Coefficient for L1 regularization (default: `0`).
- `--step_size`: Step size for the LR scheduler (default: `10`).
- `--gamma`: Gamma for the LR scheduler (default: `0.1`).
- `--epochs`: Number of epochs (default: `30`).
- `--model`: Name of the model to use (choices: `ResNet`, `BERT`; default: `resnet`).
- `--pretrained_path`: Path of the pre-trained model file (required for some experiments).
- `--batch_size`, `-b`: Batch size for the last layer re-training (default: `128`).
- `--num_workers`: Number of CPU cores to use (default: `8`).
- `--test_only`: Only test the specified model on the specified dataset and report WGA and Avg accuracy (default: `False`).
- `--log`: Whether to log the experiment on wandb (default: `True`).
- `--for_free`: Choose the best model based on group-inferred validation data- and not the ground-truth group annotations (default: `False`).
- `--seed`: Random seed (default: `1`).
- `--random_grouping`: Randomly group validation data (default: `False`).
- `--feature_only`: Load pre-computed features instead of raw data (default: `False`).
- `--num_val`: Number of validation sets (default: `1`).
- `--fine_tune`: Whether to fine-tune the classifier (default: `False`).
- `--early_stop_val`: Use early-stop models for validation grouping (default: `False`).
- `--validation_path`: Path to validation grouping models- inferred from EIIL (default: `None`).
- `--saved_val`: Use a saved validation set (default: `False`).

## Contributing

This project welcomes contributions. To contribute, please follow these steps:

1. Fork the repository
2. Create a new branch
3. Make your changes and commit them
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

We would like to thank the authors of the following papers and repositories for their valuable contributions:

- [Deep Feature Reweighting](https://github.com/PolinaKirichenko/deep_feature_reweighting)
- [Spurious Feature Learning](https://github.com/izmailovpavel/spurious_feature_learning)
- [WILDS](https://github.com/p-lambda/wilds)
- [DRO](https://github.com/kohpangwei/group_DRO/)
