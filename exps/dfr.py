import random
import torch

from .experiment import Experiment

class DFR(Experiment):
    def __init__(self):
        super().__init__('DFR')
        
    def create_balanced_dataloader_ll(self, miscls_envs, corrcls_envs, sample_size, **kwargs):
        assert 'batch_size' in kwargs.keys(), 'Missing batch_size in arguments'
        balanced_data = []
        envs = []
        all_envs = {}
        for env_id, miscls_tensors in miscls_envs.items():
            all_envs[env_id] = miscls_tensors + corrcls_envs[env_id]
        for env_id, tensors in all_envs.items():
            random.shuffle(tensors)
            selected_tensors = tensors[:sample_size]
            balanced_data.extend(selected_tensors)
            envs.extend([env_id]*len(selected_tensors))
        features = torch.stack([tensor[0] for tensor in balanced_data])
        labels = torch.stack([tensor[1] for tensor in balanced_data])
        envs = torch.Tensor(envs).to(torch.int64)
        envs = torch.nn.functional.one_hot(envs, num_classes=8)
        balanced_dataset = torch.utils.data.TensorDataset(features, labels, envs)
        balanced_dataloader = torch.utils.data.DataLoader(balanced_dataset, batch_size=kwargs['batch_size'], shuffle=True)
        return balanced_dataloader