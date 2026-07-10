import os

import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class Subset(torch.utils.data.Dataset):
    """
    Subsets a dataset while preserving original indexing.
    NOTE: torch.utils.dataset.Subset loses original indexing.
    Args:
      dataset (Dataset): Dataset to extract the subset
      indices (np.array): Indices to select
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

        self.group_array = self.get_group_array()
        self.label_array = self.get_label_array()

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def get_group_array(self, re_evaluate=True):
        """Return an array [g_x1, g_x2, ...]"""
        if re_evaluate:
            group_array = self.dataset.group_array[self.indices]
            assert len(group_array) == len(
                self.indices), f"length of self.group_array:{len(group_array)}, length of indices:{len(self.indices)}"
            assert len(self.indices) == len(self)
            assert len(group_array) == len(self)
            return group_array
        else:
            return self.group_array

    def get_label_array(self, re_evaluate=True):
        if re_evaluate:
            label_array = self.dataset.y_array[self.indices]
            assert len(label_array) == len(self)
            return label_array
        else:
            return self.label_array


class MetaDatasetCatDog(Dataset):
    """
    MetaShift data.
    `cat` is correlated with (`sofa`, `bed`), and `dog` is correlated with (`bench`, `bike`);
    In testing set, the backgrounds of both classes are `shelf`.

    Args:
        root_dir (str): Arguments, see run_expt.py
        group_id (int, optional): Select a subset of dataset with the group id

    """

    def __init__(self, root_dir, group_id=None):
        self.root_dir = root_dir
        self.RGB = True

        self.train_data_dir = os.path.join(self.root_dir, "train")
        self.test_data_dir = os.path.join(self.root_dir, 'test')

        # Set training and testing environments
        self.n_classes = 2
        cat_dict = {0: ["sofa"], 1: ["bed"]}
        dog_dict = {0: ['bench'], 1: ['bike']}
        self.test_groups = {"cat": ["shelf"], "dog": ["shelf"]}
        self.train_groups = {"cat": cat_dict, "dog": dog_dict}
        self.train_filename_array, self.train_group_array, self.train_y_array = self.get_data(self.train_groups,
                                                                                              is_training=True)
        self.test_filename_array, self.test_group_array, self.test_y_array = self.get_data(self.test_groups,
                                                                                           is_training=False)

        # split test and validation set
        np.random.seed(100)
        test_idxes = np.arange(len(self.test_group_array))
        val_idxes, _ = train_test_split(np.arange(len(test_idxes)), test_size=0.85, random_state=0)

        # define the split array
        self.train_split_array = np.zeros(len(self.train_group_array))
        self.test_split_array = 2 * np.ones(len(self.test_group_array))
        self.test_split_array[val_idxes] = 1

        self.filename_array = np.concatenate([self.train_filename_array, self.test_filename_array])

        self.group_array = np.concatenate([self.train_group_array, self.test_group_array])
        self.group_array_onehot = torch.zeros(len(self.group_array), 4)
        self.group_array_onehot = self.group_array_onehot.scatter_(1, torch.tensor(self.group_array).unsqueeze(1), 1).numpy()

        self.split_array = np.concatenate([self.train_split_array, self.test_split_array])
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}

        self.y_array = np.concatenate([self.train_y_array, self.test_y_array])
        self.y_array_onehot = torch.zeros(len(self.y_array), self.n_classes)
        self.y_array_onehot = self.y_array_onehot.scatter_(1, torch.tensor(self.y_array).unsqueeze(1), 1).numpy()

        if group_id is not None:
            idxes = np.where(self.group_array == group_id)
            self.filename_array = self.filename_array[idxes]
            self.group_array = self.group_array[idxes]
            self.split_array = self.split_array[idxes]
            self.y_array = self.y_array[idxes]
            self.y_array_onehot = self.y_array_onehot[idxes]

        self.train_transform = get_transform_metashift(train=True)
        self.eval_transform = get_transform_metashift(train=False)

        self.n_groups = len(np.unique(self.group_array))

    def __len__(self):
        return len(self.group_array)

    def __getitem__(self, idx):
        g = self.group_array_onehot[idx]
        y = self.y_array_onehot[idx]
        x = self.get_image(idx)

        return x, y, g

    def get_image(self, idx):
        img_filename = self.filename_array[idx]
        img = Image.open(img_filename)
        if self.RGB:
            img = img.convert("RGB")
        # Figure out split and transform accordingly
        if self.split_array[idx] == self.split_dict['train'] and self.train_transform:
            img = self.train_transform(img)
        elif (self.split_array[idx] in [self.split_dict['val'], self.split_dict['test']] and
              self.eval_transform):
            img = self.eval_transform(img)

        return img

    def get_data(self, groups, is_training):
        filenames = []
        group_ids = []
        ys = []
        id_count = 0
        animal_count = 0
        for animal in groups.keys():
            if is_training:
                for _, group_animal_data in groups[animal].items():
                    for group in group_animal_data:
                        for file in os.listdir(f"{self.train_data_dir}/{animal}/{animal}({group})"):
                            filenames.append(os.path.join(f"{self.train_data_dir}/{animal}/{animal}({group})", file))
                            group_ids.append(id_count)
                            ys.append(animal_count)
                    id_count += 1
            else:
                for group in groups[animal]:
                    for file in os.listdir(f"{self.test_data_dir}/{animal}/{animal}({group})"):
                        filenames.append(os.path.join(f"{self.test_data_dir}/{animal}/{animal}({group})", file))
                        group_ids.append(id_count)
                        ys.append(animal_count)
                    id_count += 1
            animal_count += 1
        return filenames, np.array(group_ids), np.array(ys)

    def get_splits(self, splits, train_frac=1.0):
        subsets = {}
        for split in splits:
            mask = self.split_array == self.split_dict[split]
            indices = np.where(mask)[0]
            if train_frac < 1 and split == 'train':
                num_to_retain = int(np.round(float(len(indices)) * train_frac))
                indices = np.sort(np.random.permutation(indices)[:num_to_retain])
            subsets[split] = Subset(self, indices)
        return subsets

    def group_str(self, group_idx, train=False):
        if not train:
            if group_idx < len(self.test_groups['cat']):
                group_name = f'y = cat'
                group_name += f", attr = {self.test_groups['cat'][group_idx]}"
            else:
                group_name = f"y = dog"
                group_name += f", attr = {self.test_groups['dog'][group_idx - len(self.test_groups['cat'])]}"
        else:
            if group_idx < len(self.train_groups['cat']):
                group_name = f'y = cat'
                group_name += f", attr = {self.train_groups['cat'][group_idx][0]}"
            else:
                group_name = f"y = dog"
                group_name += f", attr = {self.train_groups['dog'][group_idx - len(self.train_groups['cat'])][0]}"
        return group_name



def get_transform_metashift(train):
    scale = 256.0 / 224.0
    target_resolution = (224, 224)
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    if not train:
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0] * scale),
                               int(target_resolution[1] * scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    return transform


def get_metashift_loaders(path, batch_size, num_workers = 2):
    loader_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': False}

    full_dataset = MetaDatasetCatDog(root_dir=path)
    splits = ['train', 'val', 'test']
    subsets = full_dataset.get_splits(splits=splits, train_frac=1.0)
    train_data, val_data, test_data = [subsets[split] for split in splits]

    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_data, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader