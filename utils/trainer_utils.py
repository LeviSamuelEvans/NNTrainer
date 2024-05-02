import torch
import torch.nn as nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import logging

"""
A collection of utility functions for training neural networks with the Trainer class.
"""


def gather_all_labels(loader, device):
    """Gathers all labels from the given data loader and concatenates them into a single tensor.

    Parameters:
    -----------
    loader : torch.utils.data.DataLoader
        The data loader containing the labeled data.
    device : torch.device
        The device to move the labels to.

    Returns:
    --------
    torch.Tensor
        A tensor containing all the labels concatenated along the specified dimension.
    """
    all_labels = []
    for i, data in enumerate(loader):
        if isinstance(data, (list, tuple)):
            # data is a tuple or list of tensors (FFN)
            labels = data[1].to(device)
        else:
            # data is a PyTorch Geometric Data object
            labels = data.y.to(device)
        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        all_labels.append(labels)

    print("Data object attributes:")  # FOR DEBUGGING
    print(data)

    concatenated_labels = torch.cat(all_labels, dim=0)
    return concatenated_labels


# ===========================================================================================


def compute_class_weights(all_labels):
    """Compute class weights for binary classification.

    Parameters:
    -----------
        all_labels : torch.Tensor
            Tensor containing all the labels.

    Returns:
    --------
        torch.Tensor
            Tensor containing the class weights.
    """
    positive_examples = float((all_labels == 1).sum())
    negative_examples = float((all_labels == 0).sum())
    pos_weight = negative_examples / positive_examples
    return torch.tensor([pos_weight], dtype=torch.float32)


# ===========================================================================================


def get_class_weights(y) -> torch.Tensor:
    """Compute class weights based on the given labels.

    Parameters
    ----------
    y : torch.Tensor
        Labels tensor.

    Returns
    -------
    torch.Tensor
        Computed class weights.
    """

    # logging.info("Computing class weights...")
    # Convert tensor to numpy array
    y_np = y.numpy().squeeze()
    # Compute class weights using sklearns compute_class_weight
    classes = np.unique(y_np)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_np)

    # Convert class weights to a tensor and move to the same device as y
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(y.device)

    return class_weights_tensor


# ===========================================================================================


def initialise_weights(model) -> None:
    """Initialise the weights of the given model using Xavier Uniform initialization for linear layers.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model whose weights need to be initialized.

    Notes:
            - https://arxiv.org/abs/1706.03762 uses Xavier Uniform initialisation for linear layers.
            - another method is applying a normal distribution with mean 0 and std 0.02
    """
    if isinstance(model, nn.TransformerEncoder) or isinstance(
        model, nn.TransformerDecoder
    ):
        # initialise weights for transformer encoder
        for name, param in model.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
    else:
        # initialise weights for other layers
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# ===========================================================================================


def validate(model, val_loader, device, criterion, network_type) -> float:
    """Perform validation on the model.

    Returns
    -------
    float
        The validation loss.
    """
    model.eval()
    val_epoch_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for data in val_loader:
            if isinstance(data, tuple):
                # Standard data loaders return tuples (inputs, labels)
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
            elif isinstance(data, list) and all(
                isinstance(x, torch.Tensor) for x in data
            ):
                # Assuming data is a list of tensors
                data = [x.to(device) for x in data]
                inputs, labels = data
            else:
                # Assuming data is a PyTorch Geometric Data object for GNNs
                data = data.to(device)
                inputs, edge_index = data.x, data.edge_index
                labels = data.y

            if hasattr(data, "batch"):
                batch = data.batch

            # Check if the model is a type of Transformer that uses input twice
            if model.__class__.__name__ in [
                "TransformerClassifier2",
                "SetsTransformerClassifier",
                "TransformerClassifier5",
            ]:
                outputs = model(inputs, inputs)
            elif network_type in ["GNN", "LENN", "TransformerGCN"]:
                # Handle graph-based models which might require edge indices and possibly batch vectors
                if hasattr(data, "batch"):
                    outputs = model(inputs, edge_index, batch)
                else:
                    outputs = model(inputs, edge_index)
            else:
                outputs = model(inputs)

            # Calculate validation accuracy
            probabilities = torch.sigmoid(outputs).squeeze()
            predicted = torch.round(probabilities)

            if labels.dim() > 1:
                labels = labels.squeeze(-1)

            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

            outputs = outputs.squeeze(-1)

            loss = criterion(outputs, labels)
            val_epoch_loss += loss.item() * len(inputs)

    val_epoch_loss /= len(val_loader.dataset)
    val_accuracy = val_correct / val_total

    return val_epoch_loss


# ===========================================================================================


def separator() -> None:
    """Print a separator line."""
    logging.info("=" * 50)
    logging.info("=" * 50)


# ===========================================================================================
