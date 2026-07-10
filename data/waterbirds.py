import os
import torch
from torch import nn
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

class WaterbirdDataset(Dataset):
    def __init__(self, split, transform, dataset_dir='waterbird_complete_forest2water2', spuriousity=95, **kwargs):
        assert 'num_classes' in kwargs.keys(), 'num_classes missing in class arguments'
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2,
            'last_layer': 3,
        }
        self.env_dict = {
            (0, 0): torch.Tensor(np.array([1,0,0,0])),
            (0, 1): torch.Tensor(np.array([0,1,0,0])),
            (1, 0): torch.Tensor(np.array([0,0,1,0])),
            (1, 1): torch.Tensor(np.array([0,0,0,1]))
        }
        self.split = split
        self.dataset_dir = dataset_dir
        if not os.path.exists(self.dataset_dir):
            raise ValueError(
                f'{self.dataset_dir} does not exist yet. Please generate the dataset first.'
                )
        self.metadata_df = pd.read_csv(
            os.path.join(self.dataset_dir, 'metadata.csv') 
            # f'balanced_metadata{spuriousity}_with_additional_last_layer.csv')
            # wb_official_metadata_20LL
            )
        self.metadata_df = self.metadata_df[self.metadata_df['split']==self.split_dict[self.split]]

        y_array = torch.Tensor(np.array(self.metadata_df['y'].values)).type(torch.LongTensor)
        self.y_array = self.metadata_df['y'].values

        self.place_array = self.metadata_df['place'].values
        self.filename_array = self.metadata_df['img_filename'].values
        self.transform = transform

        self.y_one_hot = nn.functional.one_hot(y_array, num_classes=kwargs['num_classes']).type(torch.FloatTensor)
    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        place = self.place_array[idx]
        img_filename = os.path.join(
            self.dataset_dir,
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        img = self.transform(img)

        label = self.y_one_hot[idx]

        return img, label, self.env_dict[(y, place)]

    def get_raw_image(self,idx):
        scale = 256.0/224.0
        target_resolution = [224, 224]
        img_filename = os.path.join(
            self.dataset_dir,
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        transform = transforms.Compose([
          transforms.Resize(
              (int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
          transforms.CenterCrop(target_resolution),
          transforms.ToTensor(),
        ])
        return transform(img)


def get_waterbird_dataloader(dataset_dir, split, transform, batch_size, spuriousity):
    kwargs = {'pin_memory': True, 'num_workers': 8, 'drop_last': False}
    dataset = WaterbirdDataset(split=split, transform=transform, dataset_dir=dataset_dir, num_classes=2, spuriousity=spuriousity)
    if split in ['train', 'last_layer']:
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, **kwargs)
    else:
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return dataloader

def get_waterbird_dataset(split, transform, spuriousity):
    kwargs = {'pin_memory': True, 'num_workers': 4, 'drop_last': True}
    dataset = WaterbirdDataset(split=split, transform=transform, spuriousity=spuriousity)
    return dataset


def get_transform_cub(train):
    scale = 256.0/224.0
    target_resolution = [224, 224]
    assert target_resolution is not None

    if (not train):

        transform = transforms.Compose([
                transforms.Resize(
                    (int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
                transforms.CenterCrop(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                ),
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
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),
        ])

    return transform

def get_waterbirds_loaders(dataset_dir, spuriousity=95, **kwargs):
    assert 'batch_size' in kwargs.keys(), 'Unspecified batch_size in kwargs'
    t_train =  get_transform_cub(True)
    t_tst = get_transform_cub(False)
    trainloader = get_waterbird_dataloader(dataset_dir, 'train', t_train, kwargs['batch_size'], spuriousity)
    lastlayerloader = get_waterbird_dataloader(dataset_dir, 'last_layer', t_tst, kwargs['batch_size'], spuriousity)
    valloader = get_waterbird_dataloader(dataset_dir, 'val', t_tst, kwargs['batch_size'], spuriousity)
    testloader = get_waterbird_dataloader(dataset_dir, 'test', t_tst, kwargs['batch_size'], spuriousity)
    return trainloader, lastlayerloader, valloader, testloader
