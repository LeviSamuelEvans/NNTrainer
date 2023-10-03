import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import logging

class DataPreparation:
    """
    A class for preparing data for training and validation by converting the input dataframes to tensors, splitting the data into
    training and validation sets, shuffling the data, normalising the data, and creating data loaders for each set.

    Args:
    - df_sig (pandas.DataFrame): A pandas dataframe containing the signal data.
    - df_bkg (pandas.DataFrame): A pandas dataframe containing the background data.
    - batch_size (int): The batch size for the data loaders.
    - features (list): A list of strings containing the names of the features to be used for training.
    - train_ratio (float): The ratio of data to be used for training. Default is 0.8.

    Returns:
    - train_loader (torch.utils.data.dataloader.DataLoader): A PyTorch DataLoader object containing the training data.
    - val_loader (torch.utils.data.dataloader.DataLoader): A PyTorch DataLoader object containing the validation data.
     """
    def __init__(self, df_sig, df_bkg, batch_size, features, train_ratio=0.8):
        self.df_sig = df_sig
        self.df_bkg = df_bkg
        self.batch_size = batch_size
        self.features = features
        self.train_ratio = train_ratio


    def convert_to_tensors(self):
        """
        Converts the input dataframes into PyTorch tensors and creates corresponding target tensors.

        Returns:
        - X: PyTorch tensor containing concatenated signal and background data
        - y: PyTorch tensor containing concatenated target values (1 for signal, 0 for background)
        """
        logging.info("1. Converting to Tensors...")
        X_sig = torch.tensor(self.df_sig[self.features].values, dtype=torch.float32) # Convert the dataframes to tensors
        X_bkg = torch.tensor(self.df_bkg[self.features].values, dtype=torch.float32) # Convert the dataframes to tensors
        y_sig = torch.ones((X_sig.shape[0], 1), dtype=torch.float32) # Create a tensor of ones for signal label
        y_bkg = torch.zeros((X_bkg.shape[0], 1), dtype=torch.float32) # Create a tensor of zeros for background label
        X = torch.cat([X_sig, X_bkg]) # Concatenate signal and background data
        y = torch.cat([y_sig, y_bkg]) # Concatenate signal and background targets/labels
        return X, y

    def split_data(self, X, y):
        """
        Splits the input data into training and validation datasets based on the given train_ratio.
        Also implements shuffling of the data, in order to avoid any potential bias in the training
        and validation sets.

        Args:
        - X (torch.Tensor): Input data tensor.
        - y (torch.Tensor): Target data tensor.

        Returns:
        - train_dataset (torch.utils.data.dataset.TensorDataset): Training dataset.
        - val_dataset (torch.utils.data.dataset.TensorDataset): Validation dataset.
        """
        logging.info("2. Creating the training and validation datasets...")

        # Shuffle the data
        indices = torch.randperm(X.size(0))
        X = X[indices]
        y = y[indices]

        dataset = TensorDataset(X, y)
        train_size = int(self.train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        return train_dataset, val_dataset

    def normalize_data(self, train_dataset, val_dataset):
        """
        Normalizes the data in the train and validation datasets using mean and standard deviation.

        Args:
        train_dataset (list): A list of tuples containing the training data and labels.
        val_dataset (list): A list of tuples containing the validation data and labels.

        Returns:
        tuple: A tuple containing the normalized training and validation datasets.
        """
        logging.info("3. Normalizing the input data...")
        X_train = [x[0] for x in train_dataset]
        mean = torch.stack(X_train).mean(dim=0, keepdim=True)
        std = torch.stack(X_train).std(dim=0, keepdim=True)

        train_dataset = [(((x - mean) / (std + 1e-7)).squeeze(), y) for x, y in train_dataset]
        val_dataset = [(((x - mean) / (std + 1e-7)).squeeze(), y) for x, y in val_dataset]
        return train_dataset, val_dataset


    def prepare_data(self):
        """
        Prepares the data for training and validation by converting the input dataframes to tensors, splitting the data into
        training and validation sets, normalizing the data, and creating data loaders for each set.

        Returns:
            tuple: A tuple containing the training data loader and validation data loader.
        """

        np.random.seed(42)
        torch.manual_seed(42)

        X, y = self.convert_to_tensors()
        train_dataset, val_dataset = self.split_data(X, y)
         # before normalization DEBUG
        sample_data, sample_label = train_dataset[0]

        logging.debug(f"Sample data shape before (train_dataset): {sample_data.shape}")
        logging.debug(f"Sample label shape before (train_dataset): {sample_label.shape}")

        train_dataset, val_dataset = self.normalize_data(train_dataset, val_dataset)
        # After normalization DEBUG
        sample_data, sample_label = train_dataset[0]

        logging.debug(f"Sample data shape after (train_dataset): {sample_data.shape}")
        logging.debug(f"Sample label shape after (train_dataset): {sample_label.shape}")

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        # After creating the DataLoader
        for batch_data, batch_label in train_loader:
            logging.debug(f"Batch data shape (train_loader): {batch_data.shape}")
            logging.debug(f"Batch label shape (train_loader): {batch_label.shape}")
            break  # Only check the first batch

        logging.info(f"Total samples: {len(X)}")
        logging.info(f"Training samples: {len(train_dataset)}")
        logging.info(f"Validation samples: {len(val_dataset)}")
        logging.info(f"Batch size: {self.batch_size}")

        return train_loader, val_loader

# Future class to perform data augmentation on the original data
class DataAugmentation:
    def __init__(self) -> None:
        pass