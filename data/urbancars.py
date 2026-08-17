# from: https://raw.githubusercontent.com/facebookresearch/Whac-A-Mole/main/dataset/urbancars.py
"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import glob
import torch
import torchvision
import random
import pandas as pd
from PIL import Image

class UrbanCarsDataset(torch.utils.data.Dataset):
    obj_name_list = [
        "urban",
        "country",
    ]

    bg_name_list = [
        "urban",
        "country",
    ]

    co_occur_obj_name_list = [
        "urban",
        "country",
    ]

    def __init__(
        self,
        root: str,
        split: str,
        group_label="both",
        transform=None,
        return_group_index=True,
        return_domain_label=False,
        return_dist_shift=False,
    ):
        if split == "train":
            bg_ratio = 0.95
            co_occur_obj_ratio = 0.95
        elif split in ["lastlayer", "val", "test"]:
            bg_ratio = 0.5
            co_occur_obj_ratio = 0.5
        else:
            raise NotImplementedError
        self.bg_ratio = bg_ratio
        self.co_occur_obj_ratio = co_occur_obj_ratio

        assert os.path.exists(root)

        super().__init__()
        assert group_label in ["bg", "co_occur_obj", "both"]
        if transform == 'train':
            transform = get_transforms("resnet50", is_training=True)
        elif transform == 'test':
            transform = get_transforms("resnet50", is_training=False)
        self.transform = transform
        self.return_group_index = return_group_index
        self.return_domain_label = return_domain_label
        self.return_dist_shift = return_dist_shift
        self.split = split

        ratio_combination_folder_name = f"bg-{bg_ratio}_co_occur_obj-{co_occur_obj_ratio}"
        img_root = os.path.join(root, ratio_combination_folder_name, split)

        self.img_fpath_list = []
        self.obj_bg_co_occur_obj_label_list = []

        for obj_id, obj_name in enumerate(self.obj_name_list):
            for bg_id, bg_name in enumerate(self.bg_name_list):
                for co_occur_obj_id, co_occur_obj_name in enumerate(self.co_occur_obj_name_list):
                    dir_name = f"obj-{obj_name}_bg-{bg_name}_co_occur_obj-{co_occur_obj_name}"
                    dir_path = os.path.join(img_root, dir_name)
                    assert os.path.exists(dir_path)

                    img_fpath_list = glob.glob(os.path.join(dir_path, "*.jpg"))
                    self.img_fpath_list += img_fpath_list

                    self.obj_bg_co_occur_obj_label_list += [(obj_id, bg_id, co_occur_obj_id)] * len(img_fpath_list)

        self.obj_bg_co_occur_obj_label_list = torch.tensor(self.obj_bg_co_occur_obj_label_list, dtype=torch.long)

        self.obj_label = self.obj_bg_co_occur_obj_label_list[:, 0]
        bg_label = self.obj_bg_co_occur_obj_label_list[:, 1]
        co_occur_obj_label = self.obj_bg_co_occur_obj_label_list[:, 2]

        self.y = torch.nn.functional.one_hot(self.obj_label, num_classes=2).type(torch.FloatTensor)

        if group_label == "bg":
            num_shortcut_category = 2
            shortcut_label = bg_label
        elif group_label == "co_occur_obj":
            num_shortcut_category = 2
            shortcut_label = co_occur_obj_label
        elif group_label == "both":
            num_shortcut_category = 4
            shortcut_label = bg_label * 2 + co_occur_obj_label
        else:
            raise NotImplementedError

        self.domain_label = shortcut_label
        self.set_num_group_and_group_array(num_shortcut_category, shortcut_label)

    def _get_subsample_group_indices(self, subsample_which_shortcut):
        bg_ratio = self.bg_ratio
        co_occur_obj_ratio = self.co_occur_obj_ratio

        num_img_per_obj_class = len(self) // len(self.obj_name_list)
        if subsample_which_shortcut == "bg":
            min_size = int(min(1 - bg_ratio, bg_ratio) * num_img_per_obj_class)
        elif subsample_which_shortcut == "co_occur_obj":
            min_size = int(min(1 - co_occur_obj_ratio, co_occur_obj_ratio) * num_img_per_obj_class)
        elif subsample_which_shortcut == "both":
            min_bg_ratio = min(1 - bg_ratio, bg_ratio)
            min_co_occur_obj_ratio = min(1 - co_occur_obj_ratio, co_occur_obj_ratio)
            min_size = int(min_bg_ratio * min_co_occur_obj_ratio * num_img_per_obj_class)
        else:
            raise NotImplementedError

        assert min_size > 1

        indices = []

        if subsample_which_shortcut == "bg":
            for idx_obj in range(len(self.obj_name_list)):
                obj_mask = self.obj_bg_co_occur_obj_label_list[:, 0] == idx_obj
                for idx_bg in range(len(self.bg_name_list)):
                    bg_mask = self.obj_bg_co_occur_obj_label_list[:, 1] == idx_bg
                    mask = obj_mask & bg_mask
                    subgroup_indices = torch.nonzero(mask).squeeze().tolist()
                    random.shuffle(subgroup_indices)
                    sampled_subgroup_indices = subgroup_indices[:min_size]
                    indices += sampled_subgroup_indices
        elif subsample_which_shortcut == "co_occur_obj":
            for idx_obj in range(len(self.obj_name_list)):
                obj_mask = self.obj_bg_co_occur_obj_label_list[:, 0] == idx_obj
                for idx_co_occur_obj in range(len(self.co_occur_obj_name_list)):
                    co_occur_obj_mask = self.obj_bg_co_occur_obj_label_list[:, 2] == idx_co_occur_obj
                    mask = obj_mask & co_occur_obj_mask
                    subgroup_indices = torch.nonzero(mask).squeeze().tolist()
                    random.shuffle(subgroup_indices)
                    sampled_subgroup_indices = subgroup_indices[:min_size]
                    indices += sampled_subgroup_indices
        elif subsample_which_shortcut == "both":
            for idx_obj in range(len(self.obj_name_list)):
                obj_mask = self.obj_bg_co_occur_obj_label_list[:, 0] == idx_obj
                for idx_bg in range(len(self.bg_name_list)):
                    bg_mask = self.obj_bg_co_occur_obj_label_list[:, 1] == idx_bg
                    for idx_co_occur_obj in range(len(self.co_occur_obj_name_list)):
                        co_occur_obj_mask = self.obj_bg_co_occur_obj_label_list[:, 2] == idx_co_occur_obj
                        mask = obj_mask & bg_mask & co_occur_obj_mask
                        subgroup_indices = torch.nonzero(mask).squeeze().tolist()
                        random.shuffle(subgroup_indices)
                        sampled_subgroup_indices = subgroup_indices[:min_size]
                        indices += sampled_subgroup_indices
        else:
            raise NotImplementedError

        return indices

    def set_num_group_and_group_array(self, num_shortcut_category, shortcut_label):
        if self.split in ['train', 'val', 'lastlayer']:
            self.num_group = 4
            self.group_array = (self.obj_label * num_shortcut_category + shortcut_label)//2
            self.group_array = torch.nn.functional.one_hot(self.group_array, num_classes=4).type(torch.FloatTensor)
        else:
            self.num_group = 8
            self.group_array = self.obj_label * num_shortcut_category + shortcut_label
            self.group_array = torch.nn.functional.one_hot(self.group_array, num_classes=8)


    def set_domain_label(self, shortcut_label):
        self.domain_label = shortcut_label

    def __len__(self):
        return len(self.img_fpath_list)

    def __getitem__(self, index):
        img_fpath = self.img_fpath_list[index]
        y = self.y[index]

        img = Image.open(img_fpath).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # data_dict = {
        #     "image": img,
        #     "label": y,
        # }

        # if self.return_group_index:
        #     data_dict["group_index"] = self.group_array[index]

        # if self.return_domain_label:
        #     data_dict["domain_label"] = self.domain_label[index]

        # if self.return_dist_shift:
        #     data_dict["dist_shift"] = 0

        return img, y, self.group_array[index]

    def get_labels(self):
        return self.obj_bg_co_occur_obj_label_list

    def get_sampling_weights(self):
        group_counts = (
            (torch.arange(self.num_group).unsqueeze(1) == self.group_array)
            .sum(1)
            .float()
        )
        group_weights = len(self) / group_counts
        weights = group_weights[self.group_array]
        return weights

class SimpleUrbanCarsDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir_path, sample_size=None):
        self.root_dir_path = root_dir_path
        self.metadata_df = pd.read_csv(os.path.join(root_dir_path, 'metadata.csv'))
        self.y = torch.nn.functional.one_hot(torch.tensor(self.metadata_df["y"].to_numpy()), num_classes=2).type(torch.FloatTensor)
        self.transform = get_transforms("resnet50", is_training=False)

        if sample_size is not None:
            metadata_df = pd.DataFrame(columns=self.metadata_df.columns)
            for label in range(2):
                temp1 = self.metadata_df[self.metadata_df['y'] == label].sample(n=sample_size)
                temp2 = temp1.copy()
                temp2['img_filename'] = temp2['img_filename'].map(lambda x: x.replace('_aug', '') if '_aug' in x else x.split('.')[0] + '_aug' + x[-4:])
                temp2['y'] = int(not label)
                metadata_df = pd.concat((metadata_df, temp1, temp2))
            self.metadata_df = metadata_df

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir_path, self.metadata_df["img_filename"].iloc[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y[idx]

def get_transforms(arch, is_training):
    if arch.startswith("resnet"):
        normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if is_training:
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    normalize,
                ]
            )
    else:
        raise NotImplementedError

    return transform

def _get_train_loader(batch_size, train_set):

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=None,
        )
        return train_loader

def get_urbancars_loaders(root, batch_size, group_label):
    train_set = UrbanCarsDataset(
        root,
        "train",
        group_label=group_label,
        transform='train',
        return_group_index=True,
        return_domain_label=False,
        return_dist_shift=False,
    )
    ll_set = UrbanCarsDataset(
        root,
        "lastlayer",
        transform='test',
    )
    val_set = UrbanCarsDataset(
        root,
        "val",
        transform='test',
    )
    test_set = UrbanCarsDataset(
        root,
        "test",
        transform='test',
    )
    obj_name_list = train_set.obj_name_list
    num_class = len(obj_name_list)

    train_loader = _get_train_loader(batch_size, train_set)
    ll_loader = torch.utils.data.DataLoader(
        ll_set,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, ll_loader, val_loader, test_loader