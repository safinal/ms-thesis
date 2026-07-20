import os
import torch


def get_feature_dataset(root_dir, split, validation_path=None):
    if validation_path and split == 'val':
        root_dir = validation_path
    features = torch.load(os.path.join(root_dir, f"{split}_features.pt"))
    labels = torch.load(os.path.join(root_dir, f"{split}_labels.pt"))
    groups = torch.load(os.path.join(root_dir, f"{split}_envs.pt"))

    return torch.utils.data.TensorDataset(features, labels, groups)

def get_feature_loaders(root_dir, batch_size, num_workers = 2, validation_path=None):
    # train_loader = DataLoader(get_feature_dataset(root_dir, 'train'), batch_size = batch_size, shuffle = True, num_workers = num_workers)
    lastlayer_dataset = get_feature_dataset(root_dir, 'lastlayer')
    lastlayer_loader = torch.utils.data.DataLoader(lastlayer_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataset = get_feature_dataset(root_dir, 'val', validation_path=validation_path)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=num_workers)
    test_dataset = get_feature_dataset(root_dir, 'test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=num_workers)
    return None, lastlayer_loader, val_loader, test_loader

def get_feature_loader (root_dir, split, batch_size=512, num_workers = 2, shuffle = False):
    loader = torch.utils.data.DataLoader(get_feature_dataset(root_dir, split), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader
