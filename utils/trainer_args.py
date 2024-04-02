import torch


class TrainerArgs:
    """Class to prepare arguments for the Trainer class."""

    def __init__(self, config, model, train_loader, val_loader):
        """Initialises the TrainerArgs class.

        Attributes
        ----------
        - config: Configuration dictionary.
        - model: Model to be trained.
        - train_loader: DataLoader for training data.
        - val_loader: DataLoader for validation data.

        The class prepares all the arguments required for the Trainer class, which
        is then passes to the class to train the model via **kwargs, where
        the arguments are unpacked.

        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    def prepare_trainer_args(self):
        """Prepares the arguments to pass to the Trainer class.

        The method prepares the arguments required for the Trainer class, which
        include:
        - model
        - train_loader
        - val_loader
        - num_epochs
        - lr
        - weight_decay
        - patience
        - early_stopping
        - use_scheduler
        - factor
        - initialise_weights
        - balance_classes
        - use_cosine_burnin
        - lr_init
        - lr_max
        - lr_final
        - burn_in
        - ramp_up
        - plateau
        - ramp_down
        - network_type
        - criterion

        """
        trainer_args = {
            "model": self.model,
            "train_loader": self.train_loader,
            "val_loader": self.val_loader,
            "num_epochs": self.config["training"]["num_epochs"],
            "lr": self.config["training"]["learning_rate"],
            "weight_decay": self.config["training"]["weight_decay"],
            "patience": self.config["training"]["patience"],
            "early_stopping": self.config["training"]["early_stopping"],
            "use_scheduler": self.config["training"]["use_scheduler"],
            "factor": self.config["training"]["factor"],
            "initialise_weights": self.config["training"]["initialise_weights"],
            "balance_classes": self.config["training"]["balance_classes"],
            "use_cosine_burnin": self.config["training"]["use_cosine_burnin"],
            "lr_init": float(self.config["training"]["lr_init"]),
            "lr_max": float(self.config["training"]["lr_max"]),
            "lr_final": float(self.config["training"]["lr_final"]),
            "burn_in": self.config["training"]["burn_in"],
            "ramp_up": self.config["training"]["ramp_up"],
            "plateau": self.config["training"]["plateau"],
            "ramp_down": self.config["training"]["ramp_down"],
            "network_type": (
                self.config["Network_type"][0]
                if isinstance(self.config["Network_type"], list)
                else self.config["Network_type"]
            ),
        }

        if self.config["training"]["criterion"] == "BCELoss":
            trainer_args["criterion"] = torch.nn.BCELoss()

        return trainer_args
