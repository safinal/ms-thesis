import torch
import os

from data.celeba import CelebADataset
from data.waterbirds import WaterbirdDataset
from .experiment import Experiment


class CVA(Experiment):
    def __init__(self):
        super().__init__('CVA')
        
    def create_balanced_dataloader_ll(self, balanced_dataset_path, sample_size, batch_size, feature_only, dataset, **kwargs):
        if feature_only:
            features = torch.load(os.path.join(balanced_dataset_path, "cva_features.pt"))
            labels = torch.load(os.path.join(balanced_dataset_path, "cva_labels.pt"))
            dataset = torch.utils.data.TensorDataset(features, labels)
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        if dataset == 'celeba':
            dataset = CelebADataset(phase=None, dataset_dir=balanced_dataset_path, spuriousity=95, transform='test', sample_size=sample_size)
        elif dataset == 'waterbirds':
            dataset = WaterbirdDataset(split='last_layer', transform='test', dataset_dir=balanced_dataset_path, num_classes=2, spuriousity=95, sample_size=sample_size)
        elif dataset == 'urbancars':
            pass
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)