# Initial Look...
import torch
import matplotlib.pyplot as plt
import logging
import numpy as np


class AttentionMap:
    def __init__(self, network_type, config_dict):
        self.network_type = network_type
        self.config_dict = config_dict

    def load_model(self):
        loaded_model = torch.load(
            "/scratch4/levans/tth-network/plots/Archives/transformer3/04_04_24/model.pt"
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded_model.eval()
        return loaded_model

    def get_attention_weights(self, data_loader, loaded_model):

        input_data, _ = next(iter(data_loader))
        device = next(loaded_model.parameters()).device
        input_data = input_data.to(device)

        # register forward hook to capture the attention weights
        attention_weights = {}

        def save_attention_weights(module, input, output):
            attention_weights["weights"] = output[1]

        loaded_model.attention_pooling.register_forward_hook(save_attention_weights)

        with torch.no_grad():
            _ = loaded_model(input_data)

        return attention_weights["weights"]

    def calculate_entropy(self, attention_weights):
        attention_weights_np = attention_weights.cpu().numpy()
        entropy = -np.sum(
            attention_weights_np * np.log(attention_weights_np + 1e-9), axis=1
        )  # Add a small number to avoid log(0)
        return entropy

    def plot_attention_weights(
        self,
        attention_weights,
        feature_names,
        layer_index=None,
    ):
        num_heads = attention_weights.size(0)
        feature_names = {}

        if layer_index is not None:
            attention_weights = attention_weights[layer_index]

        # loop over each head and plot att. weights
        for head_index in range(num_heads):
            attention_weights_head = attention_weights[head_index, :, :].cpu().numpy()

            plt.figure(figsize=(10, 8))
            plt.imshow(attention_weights_head, cmap="hot", interpolation="nearest")

            tick_marks = np.arange(len(feature_names))
            plt.xticks(tick_marks, feature_names, rotation=90)
            plt.yticks(tick_marks, feature_names)

            plt.xlabel("Object 4-vectors and b-tagging scores")
            plt.ylabel("Object 4-vectors and b-tagging scores")
            plt.title(
                f"Layer {layer_index} Attention Weights (Head {head_index})"
                if layer_index is not None
                else f"Attention Weights (Head {head_index})"
            )
            plt.colorbar()
            plt.tight_layout()

            plot_filename = (
                f"attention_weights_layer{layer_index}_head{head_index}.png"
                if layer_index is not None
                else f"attention_weights_head{head_index}.png"
            )
            plot_path = f"/scratch4/levans/tth-network/plots/Archives/transformer3/04_04_24/{plot_filename}"
            plt.savefig(plot_path)
            plt.close()

    def plot_attention_entropy(
        self, attention_weights, feature_names, layer_index=None
    ):
        entropy = self.calculate_entropy(attention_weights)
        entropies = self.calculate_entropy(attention_weights[0])

        average_attention = torch.mean(attention_weights, dim=0)

        # average attention heatmap with bicubic interpolation
        plt.figure(figsize=(10, 8))
        plt.imshow(average_attention.cpu(), cmap="hot", interpolation="bicubic")
        plt.colorbar()
        plt.title("Average Attention Weights")
        plt.xlabel("Sequence Position")
        plt.ylabel("Sequence Position")
        plt.tight_layout()
        plt.savefig(
            f"/scratch4/levans/tth-network/plots/Archives/transformer3/04_04_24/average_attention.png"
        )
        plt.close()
        # netropy of attention distributions
        plt.figure(figsize=(10, 8))
        plt.plot(entropies)
        plt.title("Entropy of Attention Distributions")
        plt.xlabel("Sequence Position")
        plt.ylabel("Entropy")
        plt.savefig(
            f"/scratch4/levans/tth-network/plots/Archives/transformer3/04_04_24/attention_entropy.png"
        )
        plt.close()


# NOTE:
# attention rollout or attention flow provide methods to aggregate attention across multiple layers
# Alain, G., & Bengio, Y. (2016). Understanding intermediate layers using linear classifier probes. arXiv preprint arXiv:1610.01644.
# Hewitt, J., & Manning, C. D. (2019). A Structural Probe for Finding Syntax in Word Representations. NAACL-HLT 2019
