import time
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import re
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
import torch.nn.init as init
from torch_geometric.loader import DataLoader as GeoDataLoader
from sklearn.utils.class_weight import compute_class_weight
from modules.config_utils import log_with_separator
from modules.scheduler import CosineRampUpDownLR

"""
TODO:
    - Add model saving (state_dict)
    - Add model loading (state_dict)
    - Add model comparison feature
"""
logging.basicConfig(level=logging.INFO)


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
    for data in loader:
        labels = data.y.to(device)
        all_labels.append(labels)
    return torch.cat(all_labels, dim=0)


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


class Trainer:
    """A class used to train a PyTorch model(s).

    Attributes
    ----------
    model : torch.nn.Module
        The PyTorch model to be trained.
    train_loader : torch.utils.data.DataLoader
        The data loader for the training set.
    val_loader : torch.utils.data.DataLoader
        The data loader for the validation set.
    num_epochs : int
        The number of epochs to train the model for.
    lr : float
        The learning rate for the optimizer.
    weight_decay : float
        The weight decay for the optimizer.
    patience : int
        The number of epochs to wait before reducing the learning rate.
    criterion : torch.nn.Module
        The loss function to be used for training.
    early_stopping : bool, optional
        Whether to use early stopping. Defaults to False.
    use_scheduler : bool, optional
        Whether to use a learning rate scheduler. Defaults to False.
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        The learning rate scheduler to be used. Defaults to None.
    factor : float, optional
        The factor by which to reduce the learning rate. Defaults to None.
    initialise_weights : bool, optional
        Whether to initialise the weights of the model. Defaults to False.
    class_weights : torch.Tensor, optional
        The class weights to use for the loss function. Defaults to None.
    balance_classes : bool, optional
        Whether to balance the classes during training. Defaults to False.
    network_type : str, optional
        The type of neural network being trained. Defaults to None.
    use_cosine_burnin : bool, optional
        Whether to use cosine annealing with burn-in for learning rate scheduling. Defaults to False.
    lr_init : float, optional
        The initial learning rate for cosine annealing. Defaults to 1e-8.
    lr_max : float, optional
        The maximum learning rate for cosine annealing. Defaults to 1e-3.
    lr_final : float, optional
        The final learning rate for cosine annealing. Defaults to 1e-5.
    burn_in : int, optional
        The number of burn-in epochs for cosine annealing. Defaults to 5.
    ramp_up : int, optional
        The number of ramp-up epochs for cosine annealing. Defaults to 10.
    plateau : int, optional
        The number of plateau epochs for cosine annealing. Defaults to 15.
    ramp_down : int, optional
        The number of ramp-down epochs for cosine annealing. Defaults to 20.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        num_epochs,
        lr,
        weight_decay,
        patience,
        criterion,
        early_stopping=False,
        use_scheduler=False,
        scheduler=None,
        factor=None,
        initialise_weights=False,
        class_weights=None,
        balance_classes=False,
        network_type=None,
        use_cosine_burnin=False,
        lr_init=1e-8,
        lr_max=1e-3,
        lr_final=1e-5,
        burn_in=5,
        ramp_up=10,
        plateau=15,
        ramp_down=20,
    ):

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )  # Use GPU if available
        logging.info(f"Device: {self.device}")
        # If using a GNN or the LEGNNs, we need to use the GeoDataLoader!
        if network_type in ["GNN", "LENN"]:
            self.train_loader = GeoDataLoader(train_loader, shuffle=True)
            self.val_loader = GeoDataLoader(val_loader, shuffle=False)
            # print(f"Initial train_loader type: {type(self.train_loader)}")  # DEBUG
        else:
            self.train_loader = train_loader
            self.val_loader = val_loader

        self.model = model
        self.model.to(self.device)  # Move model to GPU if available
        self.num_epochs = num_epochs
        self.lr = lr
        self.class_weights = class_weights
        self.criterion = criterion
        self.optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.early_stopping = early_stopping
        self.weight_decay = weight_decay
        self.use_scheduler = use_scheduler
        self.use_cosine_burnin = use_cosine_burnin
        self.lr_init = lr_init
        self.lr_max = lr_max
        self.lr_final = lr_final
        self.burn_in = burn_in
        self.ramp_up = ramp_up
        self.plateau = plateau
        self.ramp_down = ramp_down
        self.patience = patience
        self.factor = factor
        self.initialise_weights = initialise_weights
        self.balance_classes = balance_classes
        self.network_type = network_type

        if self.balance_classes == True:
            if self.network_type in ["GNN", "LENN"]:
                all_labels = gather_all_labels(train_loader, self.device)
                pos_weight = compute_class_weights(all_labels)
                pos_weight = pos_weight.to(self.device)
                self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                all_labels = torch.cat([labels for *_, labels in self.train_loader])
                class_weights = self.get_class_weights(all_labels)
                positive_examples = float((all_labels == 1).sum())
                negative_examples = float((all_labels == 0).sum())
                pos_weight = torch.tensor([negative_examples / positive_examples])
                pos_weight = pos_weight.to(self.device)
                self.criterion = torch.nn.BCEWithLogitsLoss(weight=pos_weight)
        else:
            self.criterion = torch.nn.BCELoss()

        if torch.cuda.is_available():
            logging.info("GPU is available. Using GPU for training.")
        else:
            logging.info("GPU not available. Using CPU for training.")

        if self.use_scheduler == True:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=factor,
                patience=patience,
                verbose=True,
            )
        else:
            self.scheduler = None

        if self.use_cosine_burnin == True:
            logging.info("Initialising Cosine Scheduler...")
            cosine_scheduler = CosineRampUpDownLR(
                self.optimizer,
                lr_init,
                lr_max,
                lr_final,
                burn_in,
                ramp_up,
                plateau,
                ramp_down,
                last_epoch=-1,
            )
            self.cosine_scheduler = cosine_scheduler
            logging.info(
                f"Initialised Cosine Scheduler with params: {self.cosine_scheduler.__dict__}"
            )
        else:
            self.cosine_scheduler = None

    def get_class_weights(self, y) -> torch.Tensor:
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

        logging.info("Computing class weights...")
        # Convert tensor to numpy array
        y_np = y.numpy().squeeze()
        # Compute class weights using sklearns compute_class_weight
        classes = np.unique(y_np)
        class_weights = compute_class_weight("balanced", classes=classes, y=y_np)

        # Convert class weights to a tensor and move to the same device as y
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(y.device)

        return class_weights_tensor

    @staticmethod
    def initialise_weights(model) -> None:
        """Initialise the weights of the given model using He initialisation for linear layers and sets bias to 0.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model whose weights need to be initialized.
        """
        for m in model.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def validate(self) -> float:
        """Perform validation on the model.

        Returns
        -------
        float
            The validation loss.
        """
        self.model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if (
                    self.model.__class__.__name__ == "TransformerClassifier2"
                    or self.model.__class__.__name__ == "SetsTransformerClassifier"
                ):
                    outputs = self.model(inputs, inputs)
                else:
                    outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_epoch_loss += loss.item() * inputs.size(0)
        val_epoch_loss /= len(self.val_loader.dataset)
        logging.info(f"Validation Loss: {val_epoch_loss:.4f}")
        return val_epoch_loss

    def train_model(self) -> None:
        """Train the model using the specified criterion, optimizer, and data loaders for a given number of epochs.

        Logs training and validation loss and accuracy, and stops early if validation loss doesn't improve for a given
        number of epochs.
        """
        # A loada of logging stuff
        logging.info(log_with_separator("Training Configuration"))
        logging.info(f"Model: {self.model}")
        logging.info(f"Loss function: {self.criterion}")
        logging.info(f"Number of epochs: {self.num_epochs}")
        logging.info(f"scheduler: {self.scheduler}")
        logging.info(f"patience: {self.patience}")
        logging.info(f"factor: {self.factor}")
        logging.info(f"weight_decay: {self.weight_decay}")
        logging.info(f"initialise_weights: {self.initialise_weights}")
        logging.info(f"criterion: {self.criterion}")
        logging.info(f"optimizer: {self.optimizer}")
        logging.info(f"balance_classes: {self.balance_classes}")
        logging.info(f"Burn-in Cosine Scheduler: {self.cosine_scheduler}")
        logging.info(f"Initial learning-rate: {self.lr}")
        logging.info(f"Early Stopping: {self.early_stopping}")
        logging.info("Training model...")
        start_time = time.time()
        logging.info(f"Training for {self.num_epochs} epochs...")
        logging.info(
            "Training on {} samples, validating on {} samples".format(
                len(self.train_loader.dataset), len(self.val_loader.dataset)
            )
        )
        logging.info(
            "Start time: {}".format(time.strftime("%H:%M:%S", time.localtime()))
        )
        # Onwards with the training...
        best_val_loss = float("inf")
        patience_counter = 0
        self.val_labels_list = []
        self.val_outputs_list = []

        if self.initialise_weights:
            Trainer.initialise_weights(self.model)

        if self.network_type in ["GNN", "LENN"]:
            data_iter = self.train_loader.dataset
        else:
            data_iter = self.train_loader

        for epoch in range(self.num_epochs):
            for data in data_iter:
                if self.network_type in ["GNN", "LENN"]:
                    data = torch.tensor(data).to(self.device)
                    outputs = self.model(data.x, data.edge_index, data.global_features)
                    logging.info(
                        f"Max node index in edge_index: {data.edge_index.max()}, Number of nodes: {data.x.size(0)}"
                    )
                    labels = data.y
                else:
                    break

            self.model.train()

            if self.cosine_scheduler:
                self.cosine_scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                logging.info(
                    f"Epoch {epoch+1}: Cosine Scheduler sets learning rate to {current_lr:.9f}"
                )
            self.learning_rates.append(
                current_lr
            )  # movce to after optimiser for >PyT v1.2
            running_loss = 0.0
            correct = 0  # Counter for correct predictions
            total = 0  # Counter for total predictions

            for batch_idx, (inputs, labels) in enumerate(data_iter):
                if self.network_type in ["GNN", "LENN"]:
                    outputs = self.model(data.x, data.edge_index, data.batch)
                    labels = data.y  # Access labels directly from the batch
                    # mean over sequence_length label adjustments for transformer # TEMP
                    # if labels.dim() > 2:
                    #     labels = labels.mean(dim=1, keepdim=True)
                    loss = self.criterion(outputs, labels)
                    print(
                        f"Batch {batch_idx}: Max node index in edge_index: {data.edge_index.max()}, Number of nodes: {data.x.size(0)}"
                    )
                    print(f"Max node index: {data.edge_index.max()}")
                    print(f"Number of nodes: {data.x.size(0)}")
                elif self.network_type in ["FFNN"]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    if (
                        self.model.__class__.__name__ == "TransformerClassifier2"
                        or self.model.__class__.__name__ == "SetsTransformerClassifier"
                    ):
                        outputs = self.model(inputs, inputs)
                    else:
                        outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                else:
                    raise ValueError(f"Unsupported network type: {self.network_type}")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)

                # Compute accuracy for this batch
                # predicted = torch.round(outputs) # no need for sigmoid here as in the final layer of the models we have sigmoid (need to add safeguard for this!)
                probabilities = torch.sigmoid(outputs)
                predicted = torch.round(probabilities)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Log batch progress
                if (batch_idx + 1) % 50 == 0:
                    logging.info(
                        f"Epoch [{epoch+1}/{self.num_epochs}], Batch [{batch_idx+1}/{len(self.train_loader)}], Loss: {loss.item():.6f}, Accuracy: {correct/total:.4f}"
                    )

            epoch_loss = running_loss / len(self.train_loader.dataset)
            self.train_losses.append(epoch_loss)  # Store training loss for this epoch

            epoch_accuracy = correct / total  # Compute training accuracy for this epoch
            self.train_accuracies.append(epoch_accuracy)
            # Validation loss for the epoch
            val_epoch_loss = self.validate()
            if self.scheduler:
                self.scheduler.step(
                    val_epoch_loss
                )  # Adjust learning rate based on validation loss, using ReduceLROnPlateau
                current_lr = self.optimizer.param_groups[0]["lr"]
                logging.info(
                    f"Epoch {epoch+1}: ReduceLROnPlateau sets learning rate to {current_lr:.6f}"
                )

            val_correct = 0
            val_total = 0
            with torch.no_grad():
                if self.network_type in ["GNN", "LENN"]:
                    for data in self.val_loader:
                        data = data.to(self.device)
                        outputs = self.model(data.x, data.edge_index, data.batch)
                        labels = data.y
                else:
                    for inputs, labels in self.val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        if (
                            self.model.__class__.__name__ == "TransformerClassifier2"
                            or self.model.__class__.__name__
                            == "SetsTransformerClassifier"
                        ):
                            outputs = self.model(inputs, inputs)
                        else:
                            outputs = self.model(inputs)
                        # squeeze the output to remove the extra singleton dimensions
                        probabilities = torch.sigmoid(
                            outputs
                        ).squeeze()  # Convert logits to probabilities!
                        predicted = torch.round(probabilities)
                        val_correct += (predicted == labels).sum().item()
                        val_total += labels.size(0)

                # Store the predicted scores and true labels
                self.val_labels_list.extend(labels.cpu().numpy())
                self.val_outputs_list.extend(outputs.cpu().numpy())

            self.val_losses.append(
                val_epoch_loss
            )  # Store validation loss for this epoch
            val_epoch_accuracy = (
                val_correct / val_total
            )  # Compute validation accuracy for this epoch
            self.val_accuracies.append(
                val_epoch_accuracy
            )  # Store validation accuracy for this epoch
            logging.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, "
                f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.4f}"
            )

            # Check for early stopping
            if self.early_stopping == True:
                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logging.info(
                            f"Validation loss hasn't improved for {self.patience} epochs. Stopping early."
                        )
                        break
        end_time = time.time()
        training_time = end_time - start_time

        logging.info("Training complete.")

        logging.info(f"Training time: {training_time:.2f} seconds")

        # Print CPU/GPU usage
        if torch.cuda.is_available():
            logging.info(
                f"GPU Usage: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB"
            )
        else:
            logging.info("GPU not available. Using CPU.")

        # Save our model for further use
        # saving just the state dictionary with torch.save(model.state_dict(), 'path_to_model_state_dict.pt') [MOVE TO THIS]
        torch.save(self.model, "/scratch4/levans/tth-network/models/outputs/model.pt")
        logging.info(
            "Model saved to /scratch4/levans/tth-network/models/outputs/model.pt"
        )
