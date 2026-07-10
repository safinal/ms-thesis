import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from .experiment import Experiment 
import torch.nn.functional as F

class EntropyBasedExp(Experiment):
    def __init__(self):
        super().__init__('EntropyBased')

    def calculate_entropy(self, envs_samples_dict, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        model.eval()

        envs_entropy = {}

        with torch.no_grad():
            for env, samples in envs_samples_dict.items():
                env_entropy = []
                for sample in samples:
                    input_tensor = torch.Tensor(sample[0]).to(device)

                    output = model(input_tensor.unsqueeze(0))
                    probs = F.softmax(output, dim=1)
                    entropy = -(probs*torch.log(probs)).sum(dim=1)
                    env_entropy.append(entropy.item())

                envs_entropy[env] = env_entropy
                print(f'Completed env{env}')

        return envs_entropy


    def select_samples(self, envs_samples_dict, envs_entropy_values, sample_size, top=True):
        selected_samples = {}

        for env, entropy_values in envs_entropy_values.items():
            sorted_indices = sorted(range(len(entropy_values)), key=lambda i: entropy_values[i])
            top_samples = [envs_samples_dict[env][idx] for idx in sorted_indices[:sample_size]]
            selected_samples[env] = top_samples

        return selected_samples

    def merge_dicts(self, dict1, dict2):
        merged_x = []
        merged_y = []

        for env, samples in dict1.items():
            merged_x.extend([sample[0].unsqueeze(0) for sample in samples])
            merged_y.extend([sample[1] for sample in samples])

        for env, samples in dict2.items():
            merged_x.extend([sample[0].unsqueeze(0) for sample in samples])
            merged_y.extend([sample[1] for sample in samples])

        merged_x_tensor = torch.vstack(merged_x)
        merged_y_tensor = torch.vstack(merged_y)

        return merged_x_tensor, merged_y_tensor

    def create_balanced_dataloader(self, miscls_envs, corrcls_envs, sample_size, model, **kwargs):
        assert 'batch_size' in kwargs.keys(), 'Missing batch_size in arguments'
        miscls_loss_values = self.calculate_entropy(miscls_envs, model)
        corrcls_loss_values = self.calculate_entropy(corrcls_envs, model)
        miscls_label_loss_dict = {0: miscls_loss_values[0]+miscls_loss_values[1],
                                1: miscls_loss_values[2]+miscls_loss_values[3]}
        corrcls_label_loss_dict = {0: corrcls_loss_values[0]+corrcls_loss_values[1],
                                1: corrcls_loss_values[2]+corrcls_loss_values[3]}
        miscls_data_dict = {0: miscls_envs[0]+miscls_envs[1],
                            1: miscls_envs[2]+miscls_envs[3]}
        corrcls_data_dict = {0: corrcls_envs[0]+corrcls_envs[1],
                            1: corrcls_envs[2]+corrcls_envs[3]}
        miscls_selected_samples = self.select_samples(miscls_data_dict, miscls_label_loss_dict, sample_size, top=True)
        corrcls_selected_samples = self.select_samples(corrcls_data_dict, corrcls_label_loss_dict, sample_size, top=False)
        X, y = self.merge_dicts(miscls_selected_samples, corrcls_selected_samples)
        labels_tensor = torch.argmax(y, 1)
        dummy_envs = torch.zeros((X.shape[0], 4))
        dataset = TensorDataset(X, y, dummy_envs)
        balanced_loader = DataLoader(dataset, batch_size=kwargs['batch_size'], shuffle=True)
        return balanced_loader
