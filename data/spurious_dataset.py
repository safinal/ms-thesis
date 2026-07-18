"""
SpuriousCorrelation Dataset
- Reference code: https://github.com/AndPotap/afr/blob/main/data/datasets.py
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

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


class SpuriousCorrelationDataset(Dataset):
    def __init__(self, basedir, split="train", transform=None):
        self.basedir = basedir
        self.metadata_df = self._get_metadata(split)

        self.transform = transform
        self.y_array = self.metadata_df["y"].values
        if "spurious" in self.metadata_df:
            self.spurious_array = self.metadata_df["spurious"].values
        else:
            self.spurious_array = self.metadata_df["place"].values
        self._count_attributes()
        if "group" in self.metadata_df:
            self.group_array = self.metadata_df["group"].values
        else:
            self._get_class_spurious_groups()
        self._count_groups()

        self.filename_array = self.metadata_df["img_filename"].values

    def _get_metadata(self, split):
        split_i = _get_split(split)
        metadata_df = pd.read_csv(os.path.join(self.basedir, "metadata.csv"))
        metadata_df = metadata_df[metadata_df["split"] == split_i]
        return metadata_df

    def _count_attributes(self):
        self.n_classes = np.unique(self.y_array).size
        self.n_spurious = np.unique(self.spurious_array).size
        self.y_counts = self._bincount_array_as_tensor(self.y_array)
        self.spurious_counts = self._bincount_array_as_tensor(
            self.spurious_array)

    def _count_groups(self):
        self.group_counts = self._bincount_array_as_tensor(self.group_array)
        # self.n_groups = np.unique(self.group_array).size
        self.n_groups = len(self.group_counts)

    def _get_class_spurious_groups(self):
        self.group_array = _cast_int(
            self.y_array * self.n_spurious + self.spurious_array)

    @staticmethod
    def _bincount_array_as_tensor(arr):
        return torch.from_numpy(np.bincount(arr)).long()

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        s = self.spurious_array[idx]
        x = self._image_getitem(idx)
        return x, y, g, s

    def _image_getitem(self, idx):
        img_path = os.path.join(self.basedir, self.filename_array[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img
