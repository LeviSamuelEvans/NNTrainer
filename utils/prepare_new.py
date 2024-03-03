from logging import config
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch_geometric.data as geo_data
import logging
from tqdm import tqdm


class DataPreparationFactory:
    @staticmethod
    def prep_data(network_type, loaded_data, config):
        batch_size = config["training"]["batch_size"]
        features = config["features"]

        if network_type == "FFNN":
            df_sig, df_bkg = loaded_data
            preparer = FFDataPreparation(df_sig, df_bkg, batch_size, features)
            return preparer.prepare_data()
        elif network_type == "GNN" or network_type == "LENN":
            (
                node_features_sig,
                edge_features_sig,
                global_features_sig,
                node_features_bkg,
                edge_features_bkg,
                global_features_bkg,
                df_sig,
                df_bkg,
            ) = loaded_data
            preparer = GraphDataPreparation(df_sig, df_bkg, batch_size, features)
            return preparer.prepare_GNN_data()
        else:
            raise ValueError("Invalid network type")


class BaseDataPreparation:

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
        logging.info("Converting to Tensors...")
        X_sig = torch.tensor(self.df_sig[self.features].values, dtype=torch.float32)
        X_bkg = torch.tensor(self.df_bkg[self.features].values, dtype=torch.float32)
        y_sig = torch.ones((X_sig.shape[0], 1), dtype=torch.float32)
        y_bkg = torch.zeros((X_bkg.shape[0], 1), dtype=torch.float32)
        X = torch.cat([X_sig, X_bkg])
        y = torch.cat([y_sig, y_bkg])
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
        logging.info("Creating the training and validation datasets...")

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
        logging.info(" Normalizing the input data...")
        X_train = [x[0] for x in train_dataset]
        mean = torch.stack(X_train).mean(dim=0, keepdim=True)
        std = torch.stack(X_train).std(dim=0, keepdim=True)

        train_dataset = [
            (((x - mean) / (std + 1e-7)).squeeze(), y) for x, y in train_dataset
        ]
        val_dataset = [
            (((x - mean) / (std + 1e-7)).squeeze(), y) for x, y in val_dataset
        ]
        return train_dataset, val_dataset

    def prepare_data(self):
        """
        Prepares the data for training a neural network.

        Returns:
        - train_loader (DataLoader): A PyTorch DataLoader object for the training data.
        - val_loader (DataLoader): A PyTorch DataLoader object for the validation data.
        """
        np.random.seed(42)
        torch.manual_seed(42)

        X, y = self.convert_to_tensors()
        train_dataset, val_dataset = self.split_data(X, y)
        sample_data, sample_label = train_dataset[0]

        logging.info(f"Sample data shape before (train_dataset): {sample_data.shape}")
        logging.info(f"Sample label shape before (train_dataset): {sample_label.shape}")

        train_dataset, val_dataset = self.normalize_data(train_dataset, val_dataset)
        # After normalization DEBUG
        sample_data, sample_label = train_dataset[0]

        logging.info(f"Sample data shape after (train_dataset): {sample_data.shape}")
        logging.info(f"Sample label shape after (train_dataset): {sample_label.shape}")

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
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


class FFDataPreparation(BaseDataPreparation):
    """
    A class for preparing data for feed-forward neural networks.

    Args:
        df_sig (pandas.DataFrame): The signal dataset.
        df_bkg (pandas.DataFrame): The background dataset.
        batch_size (int): The batch size for training.
        features (list): A list of feature names.
        train_ratio (float, optional): The ratio of training data to total data. Defaults to 0.8.
    """

    def __init__(self, df_sig, df_bkg, batch_size, features, train_ratio=0.8):
        super().__init__(df_sig, df_bkg, batch_size, features, train_ratio)


class GraphDataPreparation(BaseDataPreparation):
    """
    A class for preparing data in graph format for GNN training.

    Args:
        df_sig (pandas.DataFrame): A pandas dataframe containing the signal data.
        df_bkg (pandas.DataFrame): A pandas dataframe containing the background data.
        batch_size (int): The batch size for training.
        features (dict): A dictionary containing the node, edge, and global features.
        train_ratio (float): The ratio of data to use for training (default is 0.8).
    """

    def __init__(self, df_sig, df_bkg, batch_size, features, train_ratio=0.8):
        super().__init__(df_sig, df_bkg, batch_size, features, train_ratio)

    def _construct_edge_index(self, event):
        """
        Constructs edge indices for a fully connected graph of jets in each event.

        Args:
            event (pandas.Series): A pandas Series containing the jet data for a single event.

        Returns:
            edge_index (torch.Tensor): A tensor containing the edge indices.
        """
        # Count the number of jets with non-null data in the event
        num_jets = event.count() // 4  # Assuming 4 features per jet
        # num_jets = event.shape[1] # for future move to multi-dim h5 file

        edge_indices = []

        for i in range(num_jets):
            for j in range(num_jets):
                if i != j:  # Avoid self-loops
                    edge_indices.append((i, j))

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        # print("Edge Index Shape for a Graph:", edge_index.shape) # DEBUG
        return edge_index

    def convert_to_graphs(self):
        logging.info("Proceeding to convert inputs to graphs...")

        data_list = []

        # Iterate over each row in df_sig and df_bkg
        logging.info("Converting signal and background data to graphs...")
        for event_data_sig, event_data_bkg in tqdm(
            zip(self.df_sig.iterrows(), self.df_bkg.iterrows()), total=len(self.df_sig)
        ):
            _, event_data_sig = event_data_sig
            _, event_data_bkg = event_data_bkg

            # Extract features for this event
            node_features_sig = (
                event_data_sig[self.features["node_features"]].dropna().values
            )
            node_features_bkg = (
                event_data_bkg[self.features["node_features"]].dropna().values
            )

            edge_features_sig = (
                event_data_sig[self.features["edge_features"]].dropna().values
            )
            edge_features_bkg = (
                event_data_bkg[self.features["edge_features"]].dropna().values
            )

            global_features_sig = (
                event_data_sig[self.features["global_features"]].dropna().values
            )
            global_features_bkg = (
                event_data_bkg[self.features["global_features"]].dropna().values
            )

            # Construct edge indices for this event
            edge_index_sig = self._construct_edge_index(event_data_sig)
            edge_index_bkg = self._construct_edge_index(event_data_bkg)

            # Create Data object for signal and background
            #logging.info("Creating Data objects for signal and background...")
            n_features = len(self.features['node_features'])
            #n_features = 3  # DEBUG
            data_sig = geo_data.Data(
                x=torch.tensor(node_features_sig, dtype=torch.float32).view(
                    -1, n_features
                ),
                edge_index=edge_index_sig,
                edge_attr=torch.tensor(edge_features_sig, dtype=torch.float32).view(
                    -1, 1
                ),
                y=torch.tensor([1], dtype=torch.float32),
                global_features=torch.tensor(global_features_sig, dtype=torch.float32),
            )

            data_bkg = geo_data.Data(
                x=torch.tensor(node_features_bkg, dtype=torch.float32).view(
                    -1, n_features
                ),
                edge_index=edge_index_bkg,
                edge_attr=torch.tensor(edge_features_bkg, dtype=torch.float32).view(
                    -1, 1
                ),
                y=torch.tensor([0], dtype=torch.float32),
                global_features=torch.tensor(global_features_bkg, dtype=torch.float32),
            )

            # Append the Data objects to the list
            data_list.extend([data_sig, data_bkg])
        logging.info("Graphs made, moving to the next step :D.")
        return data_list

    def split_data(self, data_list):
        """
        Splits the list of Data objects into training and validation datasets.

        Args:
            data_list (list): A list of torch_geometric.data.Data objects.

        Returns:
            train_dataset (list): A list of Data objects for training.
            val_dataset (list): A list of Data objects for validation.
        """
        logging.info("Splitting the data into training and validation datasets...")

        # Shuffle the data list
        shuffled_data_list = [data_list[i] for i in torch.randperm(len(data_list))]

        # Split the data list
        train_size = int(self.train_ratio * len(data_list))
        train_dataset = shuffled_data_list[:train_size]
        val_dataset = shuffled_data_list[train_size:]

        return train_dataset, val_dataset

    def normalize_data(self, train_dataset, val_dataset):
        """
        Normalizes the input features in the training and validation datasets.

        Args:
            train_dataset (torch.utils.data.TensorDataset): The training dataset.
            val_dataset (torch.utils.data.TensorDataset): The validation dataset.

        Returns:
            train_dataset (torch.utils.data.TensorDataset): The normalized training dataset.
            val_dataset (torch.utils.data.TensorDataset): The normalized validation dataset.
        """

        logging.info("Now normalising the input features...")

        # Extract input features from training dataset
        X_train = [x[0] for x in train_dataset]

        # Calculate mean and standard deviation of input features
        mean = torch.stack(X_train).mean(dim=0, keepdim=True)
        std = torch.stack(X_train).std(dim=0, keepdim=True)

        # Normalise input features in training and validation datasets
        train_dataset = [
            (((node - mean) / (std + 1e-7)), edge, global_feature, y)
            for node, edge, global_feature, y in train_dataset
        ]
        val_dataset = [
            (((node - mean) / (std + 1e-7)), edge, global_feature, y)
            for node, edge, global_feature, y in val_dataset
        ]

        return train_dataset, val_dataset

    def prepare_GNN_data(self):
        """
        Prepares the data for GNN training.

        Returns:
            tuple: A tuple containing the DataLoader for the training and validation data.
        """

        # Convert dataframes to graph data format
        graph_data_list = self.convert_to_graphs()

        # Split the graph data into training and validation datasets
        train_dataset, val_dataset = self.split_data(graph_data_list)

        # Normalise the data
        train_dataset, val_dataset = self.normalize_data(train_dataset, val_dataset)

        # Create DataLoaders for training and validation datasets
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Logging for debugging
        logging.info(f"Total graph samples: {len(graph_data_list)}")
        logging.info(f"Training graph samples: {len(train_dataset)}")
        logging.info(f"Validation graph samples: {len(val_dataset)}")
        logging.info(f"Batch size: {self.batch_size}")

        pre_training_dir = config["data"]["pre_training_dir"]

        # Save the datasets
        torch.save(train_dataset, "train_dataset.pt")
        torch.save(val_dataset, "val_dataset.pt")

        logging.info(f"Graphs saved to {pre_training_dir}.")

        return train_loader, val_loader
