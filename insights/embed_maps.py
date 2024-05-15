# Initial Look...!!!

# IDEA -> simple plot of the PCA of the positional embeddings

import torch
import matplotlib.pyplot as plt
import logging
import numpy as np
from sklearn.decomposition import PCA
from torch import nn
import seaborn as sns

class EmbeddingMaps:
    """Class to extract the positional embeddings from a transformer model and
    perform PCA on them, to visualise class separability.


    Current Ideas and thoughts
    --------------------------

    We can compare different embedding operations via the clustering of the
    PCA embeddings, tightly clustered embeddings will tell use that the model
    could do better in capturing discriminative information.
    (-> Measure of spread/clustering metric??)

    Examine embeddings using different features to understand how much the model
    cpatures important variance in the data.

    Implement slicing for different parts of the phase-space

    """
    def __init__(self, model, network_type, config_dict):
        self.network_type = network_type
        self.config_dict = config_dict
        self.model = model

    # load the model
    def load_model(self):

        model = self.model

        state_dict = torch.load("/scratch4/levans/tth-network/plots/Archives/transformer5/07_05_24_part2/models/model_state_dict.pt")
        model.load_state_dict(state_dict)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return model

    def get_embeddings(self, x, stage):
        """Extract embeddings from specified stage."""
        with torch.no_grad():
            x = self.model.input_embedding(x)
            if stage == 'post_input':
                return x.unsqueeze(1)

            x = self.model.pos_encoder(x)
            if stage == 'post_positional_encoding':
                return x.unsqueeze(1)

            x = self.model.transformer_encoder(x)
            if stage == 'post_encoder':
                return x.unsqueeze(1)

        return x

    def normalize_embeddings(self, embeddings):
        # calc the mean and std along each feature dimension
        mean = torch.mean(embeddings, dim=0)
        std = torch.std(embeddings, dim=0)
        # norm of the embeddings
        normalized_embeddings = (embeddings - mean) / std
        return normalized_embeddings

    def perform_pca(self, embeddings, n_components=2):
        # reshape embeddings to a 2d array
        flattened_embeddings = embeddings.reshape(embeddings.shape[0], -1)

        pca = PCA(n_components=n_components)
        pca_embeddings = pca.fit_transform(flattened_embeddings)
        return pca_embeddings

    def plot_pca(self, pca_embeddings, labels, title, filename):
        plt.figure(figsize=(8, 6))

        # reshape labels
        labels = labels.reshape(-1)

        class_1_indices = labels == 0
        class_2_indices = labels == 1

        plt.scatter(pca_embeddings[class_1_indices, 0], pca_embeddings[class_1_indices, 1], color='blue', label='Class 1')
        plt.scatter(pca_embeddings[class_2_indices, 0], pca_embeddings[class_2_indices, 1], color='red', label='Class 2')

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(title)
        plt.legend()
        plt.savefig(filename)
        plt.close()

    def plot_heatmap(embeddings, stage, vmax=None, vmin=None):
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(embeddings, vmin=vmin, vmax=vmax, cmap="viridis", cbar=True)
        ax.set_title(f"Heatmap of Embeddings - {stage}")
        plt.savefig(f'heatmap_{stage}.png')

    def analyse_embeddings(self, embeddings):
        stages = ['post_input', 'post_positional_encoding', 'post_encoder']
        for stage in stages:
            normalized_embeddings = self.normalize_embeddings(embeddings)
            mean_embeddings = normalized_embeddings.mean(dim=0)
            mean_embeddings = mean_embeddings.squeeze(0)
            self.plot_heatmap(mean_embeddings.detach().cpu().numpy(), stage)

    def run(self, x, labels):
        """Execute embedding extraction, PCA, and plotting for different stages."""
        stages = ['post_input',
                  'post_positional_encoding',
                  'post_encoder',
        ]

        for stage in stages:
            embeddings = self.get_embeddings(x, stage)
            pca_embeddings = self.perform_pca(embeddings)

            if len(labels) > len(pca_embeddings):
                labels = labels[:len(pca_embeddings)]
            elif len(labels) < len(pca_embeddings):
                pad_length = len(pca_embeddings) - len(labels)
                labels = np.pad(labels, (0, pad_length), mode='constant')

            title = f'PCA of Embeddings {stage}'
            filename = f'pca_embeddings_{stage}.png'

            self.plot_pca(pca_embeddings, labels, title, filename)
            self.analyse_embeddings(embeddings)