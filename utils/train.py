import time
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
import torch.nn.init as init  #  weight initialization
from sklearn.utils.class_weight import compute_class_weight  # computing class weights
from utils.config_utils import log_with_separator
from utils.scheduler import CosineRampUpDownLR

"""
TODO:
    - Add model saving (state_dict)
    - Add model loading (state_dict)
    - Add model comparison feature
"""
logging.basicConfig(level=logging.INFO)


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
        use_cosine_burnin=False
        ):

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )  # Use GPU if available
        self.model = model
        self.model.to(self.device)  # Move model to GPU if available
        self.train_loader = train_loader
        self.val_loader = val_loader
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
        self.learning_rates = []            # list to store learning rates for each epoch 
        self.early_stopping = early_stopping
        self.weight_decay = weight_decay
        self.use_scheduler = use_scheduler
        self.use_cosine_burnin = use_cosine_burnin
        self.patience = patience
        self.factor = factor
        self.initialise_weights = initialise_weights
        self.balance_classes = balance_classes
        self.network_type = network_type

        if self.balance_classes == True:
            all_labels = torch.cat([labels for *_, labels in self.train_loader])
            class_weights = self.get_class_weights(all_labels)
            # Assuming binary classification and labels are either 0 or 1
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
                lr_init=1e-8,
                lr_max=1e-3,
                lr_final=1e-5,
                burn_in=5,
                ramp_up=10,
                plateau=15,
                ramp_down=17,
                last_epoch=-1,
            )
            self.cosine_scheduler = cosine_scheduler
            logging.info(
                f"Initialised Cosine Scheduler with params: {self.cosine_scheduler.__dict__}"
            )
        else:
            self.cosine_scheduler = None

    def get_class_weights(self, y):
        """
        Compute class weights based on the given labels.

        Args:
        - y (torch.Tensor): Labels tensor.

        Returns:
        - class_weights (torch.Tensor): Computed class weights.
        """
        print("Computing class weights...")
        y_np = y.numpy().squeeze()  # Convert tensor to numpy array
        classes_array = np.array([0, 1])  # Explicitly create a numpy array for classes
        class_weights = compute_class_weight("balanced", classes=classes_array, y=y_np)
        return torch.tensor(class_weights, dtype=torch.float32).to(y.device)

    @staticmethod
    def initialise_weights(model):
        """
        Initialises the weights of the given model using He initialisation for linear layers
        and sets bias to 0.

        Args:
        - model: PyTorch model whose weights need to be initialized

        Returns:
        - None
        """
        for m in model.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def validate(self):
        """
        Perform validation on the model.

        Returns:
            float: The validation loss.
        """
        self.model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_epoch_loss += loss.item() * inputs.size(0)
        val_epoch_loss /= len(self.val_loader.dataset)
        logging.info(f"Validation Loss: {val_epoch_loss:.4f}")
        return val_epoch_loss

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
        logging.info(f"criterion: {self.criterion}")
        logging.info(f"optimizer: {self.optimizer}")
        logging.info(f"balance_classes: {self.balance_classes}")
        logging.info(f"Burn-in Cosine Scheduler: {self.cosine_scheduler}")
        logging.info(f"Initial learning-rate: {self.lr}")
        logging.info(f"Early Stopping: {self.early_stopping}")

        logging.info("Training model...")
        start_time = time.time()  # Start time of training
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

        for epoch in range(self.num_epochs):
            self.model.train()

            if self.cosine_scheduler:
                self.cosine_scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                logging.info(f"Epoch {epoch+1}: Cosine Scheduler sets learning rate to {current_lr:.9f}")
            self.learning_rates.append(current_lr)
            running_loss = 0.0
            correct = 0  # Counter for correct predictions
            total = 0  # Counter for total predictions

            for batch_idx, data in enumerate(self.train_loader):
                if self.network_type == ["GNN"]:
                    node_features, edge_features, global_features, labels = data
                    print("Node Features Shape:", node_features.shape)  # DEBUG
                    print("Edge Features Shape:", edge_features.shape)
                    print("Global Features Shape:", global_features.shape)
                    print("Labels Shape:", labels.shape)
                    node_features = node_features.to(self.device)
                    edge_features = edge_features.to(self.device)
                    global_features = global_features.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(node_features, edge_features, global_features)
                elif self.network_type == ["FFNN"]:
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                else:
                    raise ValueError(f"Unsupported network type: {self.network_type}")

                self.optimizer.zero_grad()
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)

                # Compute accuracy for this batch
                predicted = torch.round(outputs) # no need for sigmoid here as in the final layer of the models we have sigmoid (need to add safeguard for this!)
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
            self.train_accuracies.append(
                epoch_accuracy
            )  # Store training accuracy for this epoch
            # logging.info(f"Epoch [{epoch+1}/{self.num_epochs}] Average Loss: {epoch_loss:.4f} Avergae Accuracy: {epoch_accuracy:.4f}")

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
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    predicted = torch.round(outputs)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)

                    # Store the predicted scores and true labels
                    self.val_labels_list.extend(labels.cpu().numpy())
                    self.val_outputs_list.extend(torch.sigmoid(outputs).cpu().numpy())

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
    plt.plot(np.arange(trainer.num_epochs), trainer.train_losses, label="Training loss")
    plt.plot(np.arange(trainer.num_epochs), trainer.val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    hep.atlas.label(
        loc=0,
        label="Internal",
    )
    plt.tight_layout()
    plt.savefig("/scratch4/levans/tth-network/plots/Validation/loss.png")
    logging.info(
        "Loss plot saved to /scratch4/levans/tth-network/plots/Validation/loss.png"
    )


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
    plt.plot(
        np.arange(trainer.num_epochs),
        trainer.train_accuracies,
        label="Training accuracy",
    )
    plt.plot(
        np.arange(trainer.num_epochs),
        trainer.val_accuracies,
        label="Validation accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    hep.atlas.label(
        loc=0,
        label="Internal",
    )
    plt.tight_layout()
    plt.savefig("/scratch4/levans/tth-network/plots/Validation/accuracy.png")
    logging.info(
        "Accuracy plot saved to /scratch4/levans/tth-network/plots/Validation/accuracy.png"
    )
    
def plot_lr(trainer):
    """
    Plots the learning rate over the number of epochs.

    Args:
        trainer: An instance of the Trainer class containing the learning rates.

    Returns:
        None
    """
    plt.clf()  # Clear previous plot
    plt.style.use(hep.style.ATLAS)
    plt.plot(
        np.arange(trainer.num_epochs),
        trainer.learning_rates,
        label="Learning Rate",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.legend()
    hep.atlas.label(
        loc=0,
        label="Internal",
    )
    plt.tight_layout()
    plt.savefig("/scratch4/levans/tth-network/plots/Validation/learning_rate.png")
    logging.info(
        "Learning rate plot saved to /scratch4/levans/tth-network/plots/Validation/learning_rate.png"
    )
