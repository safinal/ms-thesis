import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from .models import model_attributes
from torch.utils.data import Dataset, Subset
from .cub_dataset import CUBDataset
from .dro_dataset import DRODataset
from .multinli import MultiNLIDataset

################
### SETTINGS ###
################

confounder_settings = {
    'CUB':{
        'constructor': CUBDataset
    },
    'MultiNLI':{
        'constructor': MultiNLIDataset
    }
}

########################
### DATA PREPARATION ###
########################
def prepare_confounder_data(root_dir, train=True, return_full_dataset=False):
    full_dataset = confounder_settings['MultiNLI']['constructor'](
        root_dir=root_dir,
        target_name='gold_label_random',
        confounder_names=['sentence2_has_negation'],
        model_type='bert',
        augment_data=False)
    if return_full_dataset:
        return DRODataset(
            full_dataset,
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str)
    if train:
        splits = ['train', 'val', 'test']
    else:
        splits = ['test']
    subsets = full_dataset.get_splits(splits, train_frac=1.0)
    dro_subsets = [DRODataset(subsets[split], process_item_fn=None, n_groups=full_dataset.n_groups,
                              n_classes=full_dataset.n_classes, group_str_fn=full_dataset.group_str) \
                   for split in splits]
    return dro_subsets