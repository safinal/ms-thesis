"""
CivilComments Dataset
- Reference code: https://github.com/p-lambda/wilds/blob/472677590de351857197a9bf24958838c39c272b/wilds/datasets/civilcomments_dataset.py
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from wilds.datasets.wilds_dataset import WILDSDataset, WILDSSubset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy
from .spurious_dataset import SpuriousCorrelationDataset

class CivilCommentsDataset(WILDSDataset):
    _dataset_name = 'civilcomments'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/',
            'compressed_size': 90_644_480
        }
    }

    def __init__(self, version=None, root_dir='data', split_scheme='official'):
        self._version = version
        self._data_dir = root_dir

        # Read in metadata
        self._metadata_df = pd.read_csv(
            os.path.join(self._data_dir, 'all_data_with_identities.csv'),
            index_col=0)

        # Get the y values
        self._y_array = torch.LongTensor(self._metadata_df['toxicity'].values >= 0.5)
        self._y_size = 1
        self._n_classes = 2

        self.split_dict_new = {'train':0, 'val':1, 'test': 2, 'lastlayer':3}

        # Extract text
        self._text_array = list(self._metadata_df['comment_text'])

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')
        # metadata_df contains split names in strings, so convert them to ints
        for split in self.split_dict_new:
            split_indices = self._metadata_df['split'] == split
            self._metadata_df.loc[split_indices, 'split'] = self.split_dict_new[split]
        self._split_array = self._metadata_df['split'].values

        # Extract metadata
        self._identity_vars = [
            'male',
            'female',
            'LGBTQ',
            'christian',
            'muslim',
            'other_religions',
            'black',
            'white'
        ]
        self._auxiliary_vars = [
            'identity_any',
            'severe_toxicity',
            'obscene',
            'threat',
            'insult',
            'identity_attack',
            'sexual_explicit'
        ]

        self._metadata_array = torch.cat(
            (
                torch.LongTensor((self._metadata_df.loc[:, self._identity_vars] >= 0.5).values),
                torch.LongTensor((self._metadata_df.loc[:, self._auxiliary_vars] >= 0.5).values),
                self._y_array.reshape((-1, 1))
            ),
            dim=1
        )
        self._metadata_fields = self._identity_vars + self._auxiliary_vars + ['y']

        self._eval_groupers = [
            CombinatorialGrouper(
                dataset=self,
                groupby_fields=[identity_var, 'y'])
            for identity_var in self._identity_vars]

        super().__init__(root_dir, False, split_scheme)

    def get_input(self, idx):
        return self._text_array[idx]




class CivilComments(SpuriousCorrelationDataset):
    def __init__(self, basedir, split, transform):
        self.split_dict = {'train': 0, 'val': 1, 'test': 2, 'lastlayer': 3}
        self.basedir = basedir
        self.root_dir = "/".join(self.basedir.split("/")[:-2])
        base_dataset = CivilCommentsDataset(root_dir=basedir, split_scheme='official')
        self.dataset = self.get_subset(base_dataset, split, transform)

        attributes = ["male", "female", "LGBTQ", "black", "white", "christian",
                      "muslim", "other_religions"]
        column_names = self.dataset.metadata_fields
        y_idx = column_names.index('y')
        self.y_array = self.dataset.metadata_array[:, y_idx]
        column_names = self.dataset.metadata_fields
        self.spurious_cols = [column_names.index(a) for a in attributes]
        self.spurious_array = self.get_spurious(self.dataset.metadata_array)

        self.group_dict = {
            (0, 0): 0,
            (0, 1): 1,
            (0, 2): 2,
            (0, 3): 3,
            (0, 4): 4,
            (0, 5): 5,
            (0, 6): 6,
            (0, 7): 7,
            (1, 0): 8,
            (1, 1): 9,
            (1, 2): 10,
            (1, 3): 11,
            (1, 4): 12,
            (1, 5): 13,
            (1, 6): 14,
            (1, 7): 15,
        }

        # print (torch.max(self.spurious_array))
        # self._count_attributes()
        # self._get_class_spurious_groups()
        # self._count_groups()

    def get_subset(self, base_dataset, split, transform):
        split_mask = base_dataset.split_array == self.split_dict[split]
        split_idx = np.where(split_mask)[0]

        return WILDSSubset(base_dataset, split_idx, transform)


    def get_spurious(self, metadata):
        if len(metadata.shape) == 1:
            return metadata[self.spurious_cols]
        else:
            return metadata[:, self.spurious_cols]

    def __getitem__(self, idx):
        g = np.zeros(16)
        x, y, metadata = self.dataset[idx]
        attrs = self.spurious_array[idx]
        if torch.sum(attrs)>0:
            for i in range (8):
                if attrs[i]>0:
                    g[self.group_dict[(y.item(), i)]] = 1


        y_onehot = np.zeros(2)
        y_onehot[y]=1

        # print (torch.max(g))
        # g_onehot = torch.nn.functional.one_hot(g, num_classes = torch.max(g).item())
        #  self.dataset.dataset._metadata_df['id'][self.dataset.indices[idx]]

        return x, y_onehot, g

    def __len__(self):
        return len(self.dataset)


class TokenizeTransform:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text):
        tokens = self.tokenizer(
            str(text),
            padding="max_length",
            truncation=True,
            max_length=220,
            return_tensors="pt",
        )

        return torch.squeeze(torch.stack((
            tokens["input_ids"], tokens["attention_mask"],
            tokens["token_type_ids"]), dim=2), dim=0)


class BertTokenizeTransform(TokenizeTransform):
    def __init__(self):
        super().__init__(
                tokenizer=BertTokenizer.from_pretrained("bert-base-uncased"))



def get_civil_comments_loaders(model_name, root_dir, batch_size, num_workers=2, train_shuffle=True):
    """
    Actually load CivilComments
    """

    transform = BertTokenizeTransform()

    train_set = CivilComments(root_dir, split='train', transform=transform)
    print (len(train_set))
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=train_shuffle,
                              num_workers=num_workers)

    val_set = CivilComments(root_dir, split='val', transform=transform)
    print(len(val_set))
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    test_set = CivilComments(root_dir, split='test', transform=transform)
    print(len(test_set))
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)


    lastlayer_set = CivilComments(root_dir, split='lastlayer', transform=transform)
    print(len(lastlayer_set))
    lastlayer_loader = DataLoader(lastlayer_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, lastlayer_loader, val_loader, test_loader