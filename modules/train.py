import json
import os
import time
import logging

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GeoDataLoader
import torch.optim as optim

from modules import CosineRampUpDownLR
from utils import (
    gather_all_labels,
    compute_class_weights,
    initialise_weights,
    separator,
    _print_model_summary,
    _print_cosine_scheduler_params,
)

logging.basicConfig(level=logging.INFO)

class Trainer:
    """A class used to steer the training of a PyTorch NN.

    Attributes
    ----------
        config : dict
            The configuration dictionary for training.
        model : torch.nn.Module
            The PyTorch model to be trained.
        train_loader : torch.utils.data.DataLoader
            The data loader for training data.
        val_loader : torch.utils.data.DataLoader
            The data loader for validation data.
        device : torch.device
            The device (GPU or CPU) to be used for training.
        gradient_clipping : bool
            Whether to apply gradient clipping during training.
        max_norm : float
            The maximum norm value for gradient clipping.
        lr : float
            The learning rate for the optimizer.
        train_losses : list
            A list to store the training losses.
        val_losses : list
            A list to store the validation losses.
        train_accuracies : list
            A list to store the training accuracies.
        val_accuracies : list
            A list to store the validation accuracies.
        learning_rates : list
            A list to store the learning rates.
        current_epoch : int
            The current epoch number.

    Methods
    -------
        __len__():
            Return the total number of training samples.
        __iter__():
            Return an iterator over the training batches.
        __next__():
            Return the next training batch.
        state_dict():
            Return the state dictionary of the Trainer instance.
        load_state_dict(state_dict):
            Load the state dictionary of the Trainer instance.
        get_losses():
            Return the training and validation losses.
        get_accuracies():
            Return the training and validation accuracies.
        save_checkpoint(epoch):
            Save the training checkpoint at a specific epoch.
        load_checkpoint(checkpoint_path):
            Load a training checkpoint from a given path.
        log_metrics(epoch):
            Log the training and validation metrics for each epoch.
        get_hyperparameters():
            Return the hyperparameters used for training.
        get_total_parameters():
            Return the total number of trainable parameters in the model.
        save_metrics_to_json(file_path):
            Save the training metrics and metadata to a JSON file.
        save_model(model):
            Save the model to the specified path.
        _validate_loaders():
            Validate the train_loader and val_loader for specific network types.
        _initialise_model():
            Initialise the model and optimiser.
        _initialise_attributes():
            Initialise the attributes of the Trainer instance.
        _initialise_criterion():
            Initialise the loss criterion.
        _compute_pos_weight(all_labels):
            Compute the positive weight for imbalanced classes.
        _initialise_scheduler():
            Initialise the learning rate scheduler.
        _print_optimizer_params():
            Print the optimiser parameters.
        _early_stopping(val_loss, best_val_loss, patience_counter):
            Check if early stopping criteria are met.
        _train_epoch(epoch):
            Perform a single training epoch.
        train_model():
            Train the model using the specified criterion, optimiser, and data loaders
            for a given number of epochs.
    """

    def __init__(self, config, model, train_loader, val_loader, **kwargs):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gradient_clipping = kwargs.get("gradient_clipping", False)
        self.max_norm = kwargs.get("max_norm", None)
        self.lr = float(kwargs.get("lr", 0.001))
        logging.info(f"Device: {self.device}")

        self._validate_loaders()
        self._initialise_model()
        self._initialise_attributes()
        self._initialise_criterion()
        self._initialise_scheduler()

    def __len__(self):
        """Return the total number of training samples."""
        return len(self.train_loader.dataset)

    def __iter__(self):
        """Return an iterator over the training batches."""
        return iter(self.train_loader)

    def __next__(self):
        """Return the next training batch."""
        return next(self.train_loader)

# ======================================================================================

    def _validate_loaders(self):
        if self.config["network_type"] in ["GNN", "LENN", "TransformerGCN"]:
            assert isinstance(
                self.train_loader, GeoDataLoader
            ), "For GNN, LENN, and TransformerGCN, train_loader must be an instance of GeoDataLoader!"
            assert isinstance(
                self.val_loader, GeoDataLoader
            ), "For GNN, LENN, and TransformerGCN, val_loader must be an instance of GeoDataLoader!"

    def _initialise_model(self):
        self.model.to(self.device)
        lr = float(self.config["training"]["learning_rate"])
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.config["training"]["weight_decay"],
        )

    def _initialise_attributes(self):
        for key, value in self.config.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if hasattr(self, subkey):
                        current_type = type(getattr(self, subkey))
                        setattr(self, subkey, current_type(subvalue))
                    else:
                        setattr(self, subkey, subvalue)
            else:
                if hasattr(self, key):
                    current_type = type(getattr(self, key))
                    setattr(self, key, current_type(value))
                else:
                    setattr(self, key, value)

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []

    def _initialise_criterion(self):
        if self.balance_classes:
            all_labels = gather_all_labels(self.train_loader, self.device)
            pos_weight = (
                compute_class_weights(all_labels)
                if self.network_type in ["GNN", "LENN", "TransformerGCN"]
                else self._compute_pos_weight(all_labels)
            )
            pos_weight = pos_weight.to(self.device)
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = torch.nn.BCELoss()

    def _initialise_scheduler(self):
        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.factor,
                patience=self.patience,
                verbose=True,
            )
            logging.info("Initialised ReduceLROnPlateau scheduler.")
        else:
            self.scheduler = None

        if self.use_cosine_burnin:
            self.cosine_scheduler = CosineRampUpDownLR(
                self.optimizer,
                float(self.lr_init),
                float(self.lr_max),
                float(self.lr_final),
                self.burn_in,
                self.ramp_up,
                self.plateau,
                self.ramp_down,
                last_epoch=-1,
            )
            _print_cosine_scheduler_params(self.cosine_scheduler)
        else:
            self.cosine_scheduler = None

# ======================================================================================

    def state_dict(self):
        """Return the state dictionary of the Trainer instance."""
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
            "learning_rates": self.learning_rates,
            "epoch": self.current_epoch,
        }

    def load_state_dict(self, state_dict):
        """Load the state dictionary of the Trainer instance."""
        self.model.load_state_dict(state_dict["model_state_dict"])
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.train_losses = state_dict["train_losses"]
        self.val_losses = state_dict["val_losses"]
        self.train_accuracies = state_dict["train_accuracies"]
        self.val_accuracies = state_dict["val_accuracies"]
        self.learning_rates = state_dict["learning_rates"]
        self.current_epoch = state_dict["epoch"]

    def save_checkpoint(self, epoch):
        """Save the training checkpoint at a specific epoch."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
            "learning_rates": self.learning_rates,
        }
        torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pt")  # add savepath

    def load_checkpoint(self, checkpoint_path):
        """Load a training checkpoint from a given path."""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.train_accuracies = checkpoint["train_accuracies"]
        self.val_accuracies = checkpoint["val_accuracies"]
        self.learning_rates = checkpoint["learning_rates"]
        self.current_epoch = checkpoint["epoch"]

    def save_metrics_to_json(self, file_path):
        """Save the training metrics and metadata to a JSON file."""
        metrics_data = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
            "learning_rates": self.learning_rates,
            "hyperparameters": self.get_hyperparameters(),
            "total_parameters": self.get_total_parameters(),
        }

        with open(file_path, "w") as json_file:
            json.dump(metrics_data, json_file, indent=4)

    def save_model(self, model, model_save_path):
        """save the model state dictionary to the specified path."""
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_save_path, "model_state_dict.pt"))
        logging.info(f"Model state dictionary saved to {os.path.join(model_save_path, 'model_state_dict.pt')}")

# ======================================================================================

    def get_losses(self):
        """Return the training and validation losses."""
        return self.train_losses, self.val_losses

    def get_accuracies(self):
        """Return the training and validation accuracies."""
        return self.train_accuracies, self.val_accuracies

    def get_hyperparameters(self):
        """Return the hyperparameters used for training."""
        return {
            "learning_rate": self.lr,
            "batch_size": self.train_loader.batch_size,
            "num_epochs": self.num_epochs,
            "optimizer": type(self.optimizer).__name__,
            "criterion": type(self.criterion).__name__,
            "model": type(self.model).__name__,
        }

    def get_total_parameters(self):
        """Return the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _compute_pos_weight(self, all_labels):
        """Compute the positive weight for imbalanced classes."""
        all_labels = torch.cat([labels for *_, labels in self.train_loader])
        positive_examples = float((all_labels == 1).sum())
        negative_examples = float((all_labels == 0).sum())
        return torch.tensor([negative_examples / positive_examples])

    def _print_optimizer_params(self):
        """Print the optimisers parameters."""
        optimizer_params = self.optimizer.state_dict()["param_groups"][0]
        for param, value in optimizer_params.items():
            if isinstance(value, (float, int, bool)):
                logging.info(f"{param:<20}   {value:<15}")
        if "betas" in optimizer_params:
            betas = optimizer_params["betas"]
            logging.info(f"{'betas[0]':<20}   {betas[0]:<15.5f}")
            logging.info(f"{'betas[1]':<20}   {betas[1]:<15.5f}")

# ======================================================================================

    def _train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        tot_samples = 0

        for batch_idx, batch_data in enumerate(self.train_loader):
            inputs, labels, edge_index, batch = self._process_batch_data(batch_data)

            self.optimizer.zero_grad()

            if self.network_type in ["GNN", "LENN", "TransformerGCN"]:
                outputs = self._forward_pass(inputs, edge_index, batch)
            else:
                outputs = self._forward_pass(inputs)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            predicted = self._get_predictions(outputs)
            correct += (predicted == labels.view_as(predicted)).sum().item()
            total += labels.size(0)
            tot_samples += inputs.size(0)

            if (batch_idx + 1) % 50 == 0:
                self._log_batch_progress(epoch, batch_idx, loss.item(), correct, total)

        if self.network_type in ["GNN", "LENN", "TransformerGCN"]:
            epoch_loss = running_loss / tot_samples
            epoch_accuracy = correct / total
        else:
            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_accuracy = correct / total

        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_accuracy)

    def _validate_epoch(self, epoch):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        tot_samples = 0

        with torch.no_grad():
            for batch_data in self.val_loader:
                # if self.network_type in ["GNN", "LENN", "TransformerGCN"]:
                #     inputs, labels, edge_index, batch = self._process_batch_data(batch_data)
                #     outputs = self._forward_pass(inputs, edge_index, batch)
                # else:
                inputs, labels, edge_index, batch = self._process_batch_data(batch_data)
                outputs = self._forward_pass(inputs)

                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                predictions = self._get_predictions(outputs)
                correct += (predictions == labels.view_as(predictions)).sum().item()

                total += labels.size(0)
                tot_samples += inputs.size(0)

        if self.network_type in ["GNN", "LENN", "TransformerGCN"]:
            val_loss /= tot_samples
            val_accuracy = correct / total
        else:
            val_loss /= total
            val_accuracy = correct / total

        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)

        return val_loss

    def _early_stopping(self, val_loss, best_val_loss, patience_counter):
        """Check if early stopping criteria are met."""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= self.patience:
                logging.info(
                    f"Validation loss hasn't improved for {self.patience} epochs. Stopping early."
                )
                return True, best_val_loss, patience_counter
        return False, best_val_loss, patience_counter

    def _process_batch_data(self, batch_data):
        """Process the batch data based on the network type."""
        # check if the batch data is a tuple of inputs and labels

        if hasattr(batch_data, 'x') and hasattr(batch_data, 'edge_index'):
            # for our graph networks
            batch_data = batch_data.to(self.device)
            inputs = batch_data.x
            labels = batch_data.y
            edge_index = batch_data.edge_index
            batch = batch_data.batch if hasattr(batch_data, 'batch') else None
            return inputs, labels, edge_index, batch
        elif isinstance(batch_data, tuple):
            # for our ff networks
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device).squeeze() if len(batch_data) > 1 else None
            return inputs, labels, None, None
        elif isinstance(batch_data, torch.Tensor):
            # handle the case where batch_data is a single tensor
            inputs = batch_data.to(self.device)
            labels = None
            return inputs, labels, None, None
        elif isinstance(batch_data, list) and all(isinstance(x, torch.Tensor) for x in batch_data):
            # Assume batch_data is a list of tensors
            batch_data = [x.to(self.device) for x in batch_data]
            inputs = batch_data[0]
            labels = batch_data[1] if len(batch_data) > 1 else None
            return inputs, labels, None, None
        else:
            raise ValueError("Unsupported batch data format." + str(type(batch_data)))


    def _forward_pass(self, inputs, edge_index=None, batch=None):
        """
        Perform a forward pass through the model.

        This function is conditioned to handle different network types, where
        different inputs might be necessary, such as edge indices for graph networks
        or additional batch tensors for batched operations.

        Parameters:
        - inputs (torch.Tensor): The input features to the model.
        - edge_index (torch.Tensor): The edge indices in a COO format (if applicable).
        - batch (torch.Tensor): The batch index for each node (if applicable).

        Returns:
        - torch.Tensor: The output from the model.
        """
        if self.network_type in ["GNN", "LENN", "TransformerGCN"]:
            # here, the model expects inputs, edge_index, and possibly batch for graph-level pooling
            outputs = self.model(inputs, edge_index, batch)
        elif self.model.__class__.__name__ in ["TransformerClassifier2", "SetsTransformerClassifier", "TransformerClassifier5"]:
            # for some models, our inputs are duplicated or transformed differently
            outputs = self.model(inputs, inputs)
        else:
            # For standard models with basic inputs
            outputs = self.model(inputs)

        return outputs

    def _get_predictions(self, outputs):
        """Get the predictions from the model outputs."""
        probabilities = torch.sigmoid(outputs).squeeze()
        return torch.round(probabilities)

    def _log_batch_progress(self, epoch, batch_idx, loss, correct, total):
        """Log the progress of the current batch."""
        logging.info(
            f"Epoch: [{epoch+1}/{self.num_epochs}] | Batch: [{batch_idx+1}/{len(self.train_loader)}], Loss = {loss:.4f}, Acc = {100 * correct/total:.4f}%"
        )

    def _log_epoch_metrics(self, epoch):
        """Log the training and validation metrics for each epoch."""
        if hasattr(self, "train_losses") and hasattr(self, "val_losses"):
            train_loss = self.train_losses[-1] if self.train_losses else 0.0
            train_acc = self.train_accuracies[-1] if self.train_accuracies else 0.0
            val_loss = self.val_losses[-1] if self.val_losses else 0.0
            val_acc = self.val_accuracies[-1] if self.val_accuracies else 0.0

            logging.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
        else:
            logging.warning("Attempted to log metrics without complete data.")

    def log_metrics(self, epoch):
        """Log the training and validation metrics for each epoch."""
        logging.info(
            f"Epoch {epoch+1}/{self.num_epochs} - "
            f"Train Loss: {self.train_losses[-1]:.4f}, Train Acc: {100 * self.train_accuracies[-1]:.4f}%, "
            f"Val Loss: {self.val_losses[-1]:.4f}, Val Acc: {100 * self.val_accuracies[-1]:.4f}%"
        )
# ======================================================================================
# ======================================================================================
# MAIN TRAINING METHOD

    def train_model(self):
        """Train the model using the specified criterion, optimiser, and data loaders for a given number of epochs.

        Logs training and validation loss and accuracy, and stops early if validation loss doesn't improve for a given number of epochs.

        Saves the model and training metrics to a JSON file.
        """
        _print_model_summary(self.model)
        self._print_optimizer_params()
        separator()
        start_time = time.time()
        logging.info(
            "Start time: {}".format(time.strftime("%H:%M:%S", time.localtime()))
        )
        separator()

        best_val_loss = float("inf")
        patience_counter = 0

        if self.initialise_weights:
            initialise_weights(self.model)

        for epoch in range(self.num_epochs):
            self._train_epoch(epoch)  # Train for one epoch
            val_loss = self._validate_epoch(epoch)  # Validate after training

            # scheduler updates
            if self.scheduler and isinstance(
                self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.scheduler.step(
                    val_loss
                )  # using ReduceLROnPlateau scheduler to update learning rate
            elif self.cosine_scheduler:
                 # the CosineRampUpDownLR updates each epoch unconditionally,
                 # based on configurations in config file made by user
                self.cosine_scheduler.step()

            self._log_epoch_metrics(epoch)

            # log the current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            logging.info(f"Epoch {epoch+1}: Current LR = {current_lr:.11f}")

            # update learning rate history
            self.learning_rates.append(current_lr)

            # check if we want to stop early, based on validation loss
            stop_early, best_val_loss, patience_counter = self._early_stopping(
                val_loss, best_val_loss, patience_counter
            )
            if stop_early:
                logging.info(
                    "Trainer :: Stopping early due to lack of improvement in validation loss."
                )
                break

        end_time = time.time()
        # log the total training time
        training_time = end_time - start_time
        logging.info(f"Trainer :: Training complete. Total time: {training_time:.2f} seconds")

        # log the memory usage if GPU is used
        if torch.cuda.is_available():
            logging.info(
                f"GPU Usage: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB"
            )
            logging.info(
                f"GPU Usage (cached): {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB"
            )
        else:
            logging.info("GPU not available. Using CPU.")

        # save the final model state dictionary
        self.save_model(self.model, self.model_save_path)

# ======================================================================================