import umap
import numpy as np
from collections import defaultdict
import random
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from .experiment import Experiment

class ClusterBasedExp(Experiment):
    def __init__(self, method):
        super().__init__('ClusterBased')
        assert method in ['kmeans', 'gmm'], 'Unknown method specified!'
        self.method = method 
        
    def extract_features(self, model, dataloader):
        inputs_list = []
        features = []
        labels = []
        environments = []
        model.model.fc = torch.nn.Identity()
        with torch.no_grad():
            model.eval()
            for batch, (inputs, batch_labels, batch_envs) in enumerate(tqdm(dataloader)):
                inputs_list.extend(inputs.cpu().numpy().tolist())

                inputs = inputs.to(model.device)
                features_batch = model(inputs).cpu().numpy()

                features.extend(features_batch)
                labels.extend(batch_labels)
                environments.extend(batch_envs)
        return features, labels, environments, inputs_list

    def cluster_features(self, features, seed=42):
        if self.method == 'kmeans':
            kmeans = KMeans(n_clusters=4, random_state=seed)
            cluster_labels = kmeans.fit_predict(features)
        if self.method == 'gmm':
            gmm = GaussianMixture(n_components=4, random_state=seed)
            cluster_labels = gmm.fit_predict(features)
        if self.method == 'dbscan':
            dbscan = DBSCAN(min_samples=2)
            cluster_labels = dbscan.fit_predict(n_components=4, random_state=seed)
        return cluster_labels

    def cluster_label_based(self, validation_features, validation_labels):
        gmm_models = {}
        kmeans_models = {}
        num_clusters_per_label = 2

        for label_value in [0, 1]:
            label_features = [f for i, f in enumerate(validation_features) if (validation_labels[i].argmax().item() == label_value)]
            if self.method == 'gmm':
                gmm = GaussianMixture(n_components=num_clusters_per_label, random_state=42)
                gmm.fit(label_features)
                gmm_models[label_value] = gmm
            if self.method == 'kmeans':
                kmeans = KMeans(n_clusters=num_clusters_per_label, random_state=42)
                kmeans.fit(label_features)
                kmeans_models[label_value] = kmeans

        cluster_labels_per_label = {}
        models = kmeans_models if self.method == 'kmeans' else gmm_models
        for label_value, model in models.items():
            label_features = [f for i, f in enumerate(validation_features) if validation_labels[i].argmax().item() == label_value]
            cluster_labels = model.predict(label_features)
            cluster_labels_per_label[label_value] = cluster_labels
        return cluster_labels_per_label


    def create_balanced_dataloader(self, valloader, model, sample_size, **kwargs):
        assert 'batch_size' in kwargs.keys(), 'Missing batch_size in arguments'

        validation_features, validation_labels, validation_environments, validation_data = self.extract_features(model, valloader)
        cluster_labels_per_label = self.cluster_label_based(validation_features, validation_labels)
        cluster_samples = defaultdict(list)

        for label_value, cluster_labels in cluster_labels_per_label.items():
            for i, cluster_label in enumerate(cluster_labels):
                cluster_samples[(label_value, cluster_label)].append(i)

        balanced_samples = []

        for cluster_samples_list in cluster_samples.values():
            samples = random.sample(cluster_samples_list, sample_size)
            balanced_samples.extend(samples)

        balanced_labels = [validation_labels[i] for i in balanced_samples]
        balanced_environments = [validation_environments[i] for i in balanced_samples]
        balanced_data = [torch.Tensor(validation_data[i]) for i in balanced_samples]
        balanced_dataset = torch.utils.data.TensorDataset(torch.stack(balanced_data), 
                                                          torch.stack(balanced_labels), 
                                                          torch.stack(balanced_environments))
        balanced_dataloader = torch.utils.data.DataLoader(balanced_dataset, batch_size=kwargs['batch_size'], shuffle=True)
        return balanced_dataloader

    def plot_features_umap(self, cluster_labels_per_label, validation_features, validation_labels, validation_environments):
        sorted_features, sorted_envs = [], []
        sorted_cluster_labels = cluster_labels_per_label[0].tolist() + (cluster_labels_per_label[1]+2).tolist()
        sorted_labels = np.concatenate((np.zeros(len(cluster_labels_per_label[0])), np.ones(len(cluster_labels_per_label[1]))))
        for label_value in [0, 1]:
            sorted_envs.extend([f.argmax().item() for i, f in enumerate(validation_environments) if validation_labels[i].argmax().item() == label_value])
        for label_value in [0, 1]:
            label_features = [f for i, f in enumerate(validation_features) if validation_labels[i].argmax().item() == label_value]
            sorted_features.extend(label_features)
        pca = PCA(n_components=50)
        pca_features = pca.fit_transform(sorted_features)
        umap_emb = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='euclidean').fit_transform(pca_features)
        plt.figure(figsize=(13, 4))
        plt.subplot(1,3,1)
        plt.scatter(umap_emb[:, 0], umap_emb[:, 1], c=sorted_cluster_labels, cmap='viridis', alpha=0.6)
        plt.colorbar(label='Cluster Labels')
        plt.title('UMAP Visualization of Clusters for Each Label')
        plt.subplot(1,3,2)
        labels = torch.stack(validation_labels).argmax(axis=1).numpy()
        plt.scatter(umap_emb[:, 0], umap_emb[:, 1], c=sorted_labels, cmap='viridis', alpha=0.6)
        plt.title('Labels')
        plt.subplot(1,3,3)
        envs = torch.stack(validation_environments).argmax(axis=1).numpy()
        plt.scatter(umap_emb[:, 0], umap_emb[:, 1], c=sorted_envs, cmap='viridis', alpha=0.6)
        plt.title('Environments')
        plt.tight_layout()
        plt.show()