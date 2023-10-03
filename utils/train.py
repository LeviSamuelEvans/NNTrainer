import torch
import torch.nn as nn
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
import torch.nn.init as init  # Add this import for weight initialization
from sklearn.utils.class_weight import compute_class_weight # Add this import for computing class weights
from utils.config_utils import log_with_separator
#plt.rcParams['agg.path.chunksize'] = 10000
'''
TODO:
    - Add model saving
    - Add model loading
    - Add model evaluation/comparisons
'''

class Trainer:
    """
    A class used to train a PyTorch model(s).

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_loader (torch.utils.data.DataLoader): The data loader for the training set.
        val_loader (torch.utils.data.DataLoader): The data loader for the validation set.
        num_epochs (int): The number of epochs to train the model for.
        lr (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.
        patience (int): The number of epochs to wait before reducing the learning rate.
        early_stopping (bool, optional): Whether to use early stopping. Defaults to False.
        use_scheduler (bool, optional): Whether to use a learning rate scheduler. Defaults to False.
        factor (float, optional): The factor by which to reduce the learning rate. Defaults to None.
        initialise_weights (bool, optional): Whether to initialise the weights of the model. Defaults to False.
        class_weights (torch.Tensor, optional): The class weights to use for the loss function. Defaults to None.
    """

    def __init__(self, model, train_loader, val_loader, num_epochs, lr, weight_decay, patience,
                 early_stopping=False, use_scheduler=False, factor=None, initialise_weights=False,
                 class_weights=None, balance_classes=False):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.lr = lr
        self.criterion = torch.nn.BCEWithLogitsLoss(weight=class_weights)
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=factor, patience=patience, verbose=True)
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []  # List to store training accuracies
        self.val_accuracies = []
        self.early_stopping = early_stopping
        self.weight_decay = weight_decay
        self.use_scheduler = use_scheduler
        self.patience = patience
        self.factor = factor
        self.initialise_weights = initialise_weights
        self.balance_classes = balance_classes


    def get_class_weights(self, y):
        """
        Compute class weights based on the given labels.

        Args:
        - y (torch.Tensor): Labels tensor.

        Returns:
        - class_weights (torch.Tensor): Computed class weights.
        """
        y_np = y.numpy().squeeze()  # Convert tensor to numpy array
        class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_np)
        return torch.tensor(class_weights, dtype=torch.float32)

    def initialise_weights(model):
            """
            Initializes the weights of the given model using He initialization for linear layers
            and sets bias to 0.

            Args:
            - model: PyTorch model whose weights need to be initialized

            Returns:
            - None
            """
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                    if m.bias is not None:
                        init.constant_(m.bias, 0)

    def train_model(self):
        """
        Trains the model using the specified criterion, optimizer, and data loaders for a given number of epochs.
        Logs training and validation loss and accuracy, and stops early if validation loss doesn't improve for a given
        number of epochs.

        Returns:
            None
        """
        # A loada of logging stuff

        logging.info(log_with_separator("Training Configuration"))
        logging.info(f"Model: {self.model}")
        logging.info(f"Loss function: {self.criterion}")
        logging.info(f"Number of epochs: {self.num_epochs}")
        logging.info(f"Learning rate: {self.lr}")
        logging.info(f"scheduler: {self.scheduler}")
        logging.info(f"patience: {self.patience}")
        logging.info(f"factor: {self.factor}")
        logging.info(f"weight_decay: {self.weight_decay}")
        logging.info(f"initialise_weights: {self.initialise_weights}")
        logging.info(f"optimizer: {self.optimizer}")
        logging.info(f"balance_classes: {self.balance_classes}")

        logging.info("Training model...")
        # Onwards with the training...
        best_val_loss = float('inf')
        patience_counter = 0
        self.val_labels_list = []
        self.val_outputs_list = []
        if self.balance_classes == True:
            self.criterion = torch.nn.BCEWithLogitsLoss(weight=Trainer.class_weights)
        if Trainer.initialise_weights == True:
            self.model.initialise_weights()
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0  # Counter for correct predictions
            total = 0  # Counter for total predictions
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)

                # Compute accuracy for this batch
                predicted = torch.round(torch.sigmoid(outputs))
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Log batch progress
                if (batch_idx + 1) % 50 == 0:
                    logging.info(f"Epoch [{epoch+1}/{self.num_epochs}], Batch [{batch_idx+1}/{len(self.train_loader)}], Loss: {loss.item():.6f}, Accuracy: {correct/total:.4f}")

            epoch_loss = running_loss / len(self.train_loader.dataset)
            self.train_losses.append(epoch_loss)  # Store training loss for this epoch

            epoch_accuracy = correct / total  # Compute training accuracy for this epoch
            self.train_accuracies.append(epoch_accuracy)  # Store training accuracy for this epoch
            logging.info(f"Epoch [{epoch+1}/{self.num_epochs}] Average Loss: {epoch_loss:.4f} Avergae Accuracy: {epoch_accuracy:.4f}")

            # Validation loss for the epoch
            self.model.eval()
            val_epoch_loss = 0.0
            val_correct = 0  # Counter for correct predictions on validation set
            val_total = 0
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_epoch_loss += loss.item() * inputs.size(0)
                if self.use_scheduler == True:
                    self.scheduler.step(val_epoch_loss) # Adjust learning rate based on validation loss, using ReduceLROnPlateau

                    # Compute accuracy for this batch on validation set
                    predicted = torch.round(torch.sigmoid(outputs))
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)

                    # Store the predicted scores and true labels
                    self.val_labels_list.extend(labels.cpu().numpy())
                    self.val_outputs_list.extend(torch.sigmoid(outputs).cpu().numpy())
            print("Number of validation labels:", len(self.val_labels_list))
            print("Number of validation outputs:", len(self.val_outputs_list))


            val_epoch_loss /= len(self.val_loader.dataset)
            self.val_losses.append(val_epoch_loss)  # Store validation loss for this epoch
            val_epoch_accuracy = val_correct / val_total  # Compute validation accuracy for this epoch
            self.val_accuracies.append(val_epoch_accuracy)  # Store validation accuracy for this epoch
            logging.info(f"Epoch [{epoch+1}/{self.num_epochs}] Validation Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_accuracy:.4f}")

            # Check for early stopping
            if self.early_stopping == True:
                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logging.info(f"Validation loss hasn't improved for {self.patience} epochs. Stopping early.")
                        break

        logging.info("Training complete.")

# Add model saving

# put into class when time
def plot_losses(trainer):
    """
    Plots the training and validation losses for a given trainer object.

    Args:
        trainer (Trainer): The trainer object containing the training and validation losses.

    Returns:
        None
    """
    # Determine the number of epochs based on the length of train_losses (e.g if Early Stopping is used, this will be less than num_epochs)
    actual_epochs = len(trainer.train_losses)
    trainer.num_epochs = actual_epochs

    plt.style.use(hep.style.ATLAS)
    plt.plot(np.arange(trainer.num_epochs), trainer.train_losses, label='Training loss')
    plt.plot(np.arange(trainer.num_epochs), trainer.val_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    hep.atlas.label(loc=0, label="Internal",)
    plt.tight_layout()
    plt.savefig('plots/Validation/loss.png')
    logging.info("Loss plot saved to plots/Validation/loss.png")

def plot_accuracy(trainer):
    """
    Plots the training and validation accuracy over the number of epochs.

    Args:
        trainer: An instance of the Trainer class containing the training and validation accuracies.

    Returns:
        None
    """
    plt.clf()  # Clear previous plot
    plt.style.use(hep.style.ATLAS)
    plt.plot(np.arange(trainer.num_epochs), trainer.train_accuracies, label='Training accuracy')
    plt.plot(np.arange(trainer.num_epochs), trainer.val_accuracies, label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    hep.atlas.label(loc=0, label="Internal",)
    plt.tight_layout()
    plt.savefig('plots/Validation/accuracy.png')
    logging.info("Accuracy plot saved to plots/Validation/accuracy.png")