"""
- Reference code: https://github.com/AndPotap/afr/blob/main/data/datasets.py
"""

import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader, Subset
from .spurious_dataset import SpuriousCorrelationDataset


def _get_split(split):
    try:
        return ["train", "val", "test", "lastlayer"].index(split)
    except ValueError:
        raise (f"Unknown split {split}")


def _cast_int(arr):
    if isinstance(arr, np.ndarray):
        return arr.astype(int)
    elif isinstance(arr, torch.Tensor):
        return arr.int()
    else:
        raise NotImplementedError

class MultiNLIDataset(SpuriousCorrelationDataset):
    """Adapted from https://github.com/kohpangwei/group_DRO/blob/master/data/multinli_dataset.py
    """

    def __init__(self, basedir, split="train", transform=None):
        assert transform is None, "transfrom should be None"

        # utils_glue module in basedir is needed to load data

        self.basedir = basedir
        self.metadata_df = pd.read_csv(os.path.join(
            self.basedir, "metadata_random.csv"))
        bert_filenames = [
            "cached_train_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli-mm"]
        features_array = sum([torch.load(os.path.join(self.basedir, name))
                              for name in bert_filenames], start=[])
        all_input_ids = torch.tensor([
            f.input_ids for f in features_array]).long()
        all_input_masks = torch.tensor([
            f.input_mask for f in features_array]).long()
        all_segment_ids = torch.tensor([
            f.segment_ids for f in features_array]).long()
        # all_label_ids = torch.tensor([
        #     f.label_id for f in self.features_array]).long()

        split_i = _get_split(split)
        print (split, split_i)
        split_mask = (self.metadata_df["split"] == split_i).values

        self.x_array = torch.stack((
            all_input_ids,
            all_input_masks,
            all_segment_ids), dim=2)[split_mask]
        self.metadata_df = self.metadata_df[split_mask]
        self.y_array = self.metadata_df['gold_label'].values
        self.spurious_array = (
            self.metadata_df['sentence2_has_negation'].values)
        self._count_attributes()
        self._get_class_spurious_groups()
        self._count_groups()

    def __getitem__(self, idx):
        y = self.y_array[idx]
        y_onehot = np.zeros(3)
        y_onehot[y] = 1

        g = self.group_array[idx]
        g_onehot = np.zeros(6)
        g_onehot[g] = 1
        x = self.x_array[idx]
        return x, y_onehot, g_onehot





def get_multinli_loaders(root_dir, batch_size, num_workers=2, train_shuffle=True):
    train_set = MultiNLIDataset(root_dir, split='train')
    print (len(train_set))
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=train_shuffle,
                              num_workers=num_workers)

    val_set = MultiNLIDataset(root_dir, split='val')
    print(len(val_set))
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    test_set = MultiNLIDataset(root_dir, split='test')
    print(len(test_set))
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)


    lastlayer_set = MultiNLIDataset(root_dir, split='lastlayer')
    print(len(lastlayer_set))
    lastlayer_loader = DataLoader(lastlayer_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, lastlayer_loader, val_loader, test_loader




