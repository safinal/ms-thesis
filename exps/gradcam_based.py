from pytorch_grad_cam import GradCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from .experiment import Experiment 
import torchvision

from matplotlib import pyplot as plt
from tqdm import tqdm 

class GradCAMExp(Experiment):
    def __init__(self, model):
        super().__init__('GradCAMBased')
        self.model = model
        self.heat_map_generator = XGradCAM(
          model=model,
          target_layers=[model.model.layer4[-1]],
        )
        
    def calculate_cross_entropy(self, envs_samples_dict):
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device)
        self.model.eval()

        envs_loss = {}

        with torch.no_grad():
            for env, samples in envs_samples_dict.items():
                env_loss = []
                for sample in samples:
                    input_tensor = torch.Tensor(sample[0]).to(device)
                    label = torch.Tensor(sample[1]).float().to(device)

                    output = self.model(input_tensor.unsqueeze(0))
                    loss = loss_fn(output[0], label)
                    env_loss.append(loss.item())

                envs_loss[env] = env_loss
                print(f'Completed env{env}')

        return envs_loss


    def select_samples(self, envs_samples_dict, envs_loss_values, sample_size, top=True):
        selected_samples = {}

        for env, loss_values in envs_loss_values.items():
            sorted_indices = sorted(range(len(loss_values)), key=lambda i: loss_values[i], reverse=top)
            top_samples = [envs_samples_dict[env][idx] for idx in sorted_indices[:min(sample_size, len(sorted_indices))]]
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


    
    def visualize_heatmap(self, image, label, mode='prediction'):
        '''
        inputs:
            image: image which is normalized according to the statistics of the dataset. shape: [1, channels, width, height]
            unnormalized_image: image that you are going to visualize. the value of its pixels must be in [0,1]. shape: [width, height, channels]

        returns: visualization of the heatmap on the image
        '''
        unnormalized_image = image.cpu()*torch.Tensor(np.array([[[[0.229]], [[0.224]], [[0.225]]]]))+torch.Tensor(np.array([[[[0.485]], [[0.456]], [[0.406]]]])).cpu().detach().numpy().astype(np.float32)
        unnormalized_image = unnormalized_image.squeeze().permute(1,2,0).numpy()

        if mode == 'prediction':
            grayscale_cam = self.heat_map_generator(image)
        if mode == 'correct_class':
            targets = [ClassifierOutputTarget(torch.argmax(label).cpu().item())]
            grayscale_cam = self.heat_map_generator(image, targets)
        if mode == 'incorrect_class':
            targets = [ClassifierOutputTarget(1-torch.argmax(label).cpu().item())]
            grayscale_cam = self.heat_map_generator(image, targets)

        grayscale_cam = np.expand_dims(grayscale_cam[0, :],0)

        vis = show_cam_on_image(unnormalized_image.squeeze(), grayscale_cam.squeeze(), use_rgb=True)

        return vis
    
    
    def get_quantile_masks(self, heat_map, probs):
        masks = []
        quantiles = np.vsplit(np.quantile(heat_map, probs, axis=(1,2)), len(probs))
        for quantile in quantiles:
            quantile = quantile.reshape(quantile.shape[1], 1,1)
            masks.append(self.mask_heatmap_using_threshold(heat_map, quantile))
        return masks

    
    def mask_heatmap_using_threshold(self, heat_maps, k):
        ret = heat_maps < k
        return np.expand_dims(ret, 1)
    
    
    def gradually_mask(self, image, label, range=[0.9, 0.8, 0.7, 0.6, 0.4, 0.2], mode='prediction'):
        criterion = nn.CrossEntropyLoss()
        if mode == 'prediction':
            heat_map = self.heat_map_generator(image)
        if mode == 'correct_class':
            targets = [ClassifierOutputTarget(torch.argmax(label).cpu().item())]
            heat_map = self.heat_map_generator(image, targets)
        if mode == 'incorrect_class':
            targets = [ClassifierOutputTarget(1-torch.argmax(label).cpu().item())]
            heat_map = self.heat_map_generator(image, targets)


        plt.imshow(self.visualize_heatmap(image, label, mode))
        plt.show()

        masks = self.get_quantile_masks(heat_map, range)
        losses = []
        vis = []

        for mask in masks:
            masked = image * torch.Tensor(mask).to(image.device)

            loss = criterion(self.model(masked), label)
            losses.append(loss.detach().cpu().numpy())
            vis.append(self.visualize_heatmap(masked, label, mode))

        plt.plot(1-np.array(range), losses)
        plt.show()

        fig, axarr = plt.subplots(1, len(range), figsize=(15, 3))

        for i, image_data in enumerate(vis):
            axarr[i].imshow(image_data)
            axarr[i].axis('off')

        plt.show()

    def check_incr_true_conf(self, x, y, masked_base_loss):
        criterion = nn.CrossEntropyLoss(reduction='none')

        # base_loss = 4
        low_threshold = 0.1

        r = [0.3, 0.4, 0.5, 0.6, 0.8]
        self.model.eval()

        xs = [None, None]
        ys = [None, None]
        for i in range(2):
            xs[i] = x[y[:, i] == 1]
            ys[i] = y[y[:, i] == 1]
            with torch.no_grad():
                loss = criterion(self.model(xs[i]), ys[i])
            xs[i] = xs[i][loss > masked_base_loss[i]]
            ys[i] = ys[i][loss > masked_base_loss[i]]

        x = torch.cat(xs, 0)
        y = torch.cat(ys, 0)

        flag = torch.zeros(x.shape[0])

        for i in range(x.shape[0]):
            image = x[i].unsqueeze(0).to(device)
            label = y[i].unsqueeze(0).to(device)

            heat_map = self.heat_map_generator(image)
            masks = self.get_quantile_masks(heat_map, r)

            for mask in masks:
                masked = image * torch.Tensor(mask).to(device)

                loss = criterion(self.model(masked), label).item()
                if loss < low_threshold:
                    flag[i] = 1

        return x[flag == 1], y[flag == 1]

    def get_confidence_based_minority(self, dataloader, sample_size, masked_base_loss):
        model.eval()
        labels = [torch.Tensor(np.array([[1, 0]])).to(device), torch.Tensor(
            np.array([[0, 1]])).to(device)]
        all_samples = {0: [], 1: []}

        with torch.no_grad():
            for (batch, (inputs, labels, envs)) in enumerate(tqdm(dataloader)):
                inputs = inputs.to(device)
                labels = labels.to(device)

            selected_inputs, selected_labels = self.check_incr_true_conf(inputs, labels, masked_base_loss)

            for i in range(2):
                all_samples[i].extend(
                    list(zip(selected_inputs[selected_labels[:, i] == 1], selected_labels[selected_labels[:, i] == 1])))

            loss_values = calculate_cross_entropy(all_samples, model)
            ret = select_samples(all_samples, loss_values, sample_size, top=True)

        return ret

    def create_balanced_dataloader(self, corrcls_envs, sample_size, dataloader, batch_size=32, **kwargs):
        model.eval()
        masked_base_loss = []
        labels = [torch.Tensor(np.array([[1, 0]])).to(device), torch.Tensor(
            np.array([[0, 1]])).to(device)]
        criterion = nn.CrossEntropyLoss(reduction='none')
        null = torch.zeros((1, 3, 256, 256)).to(device)

        for label in labels:
            masked_base_loss.append(criterion(model(null), label).item())

        print(masked_base_loss)

        corrcls_loss_values = calculate_cross_entropy(corrcls_envs, model)
        corrcls_label_loss_dict = {0: corrcls_loss_values[0] + corrcls_loss_values[1],
                                   1: corrcls_loss_values[2] + corrcls_loss_values[3]}
        corrcls_data_dict = {0: corrcls_envs[0] + corrcls_envs[1],
                             1: corrcls_envs[2] + corrcls_envs[3]}

        high_loss_selected_samples = self.get_confidence_based_minority(dataloader, sample_size, masked_base_loss)
        corrcls_selected_samples = self.select_samples(corrcls_data_dict, corrcls_label_loss_dict, sample_size, top=False)

        X, y = self.merge_dicts(high_loss_selected_samples, corrcls_selected_samples)
        dummy_envs = torch.zeros((X.shape[0], 4))
        dataset = TensorDataset(X, y, dummy_envs)
        balanced_loader = DataLoader(dataset, batch_size, shuffle=True)
        return balanced_loader
