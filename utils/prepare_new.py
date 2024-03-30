from logging import config
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch_geometric.data as geo_data
from torch_geometric.loader import DataLoader as GeoDataLoader
import logging
from tqdm import tqdm
import os


class DataPreparationFactory:
    @staticmethod
    def prep_data(network_type,
                  loaded_data,
                  config,
                  signal_fvectors=None,
                  background_fvectors=None,
                  ):

        batch_size = config["training"]["batch_size"]
        #features = config["features"]
        train_ratio = config["data"]["train_ratio"]
        value_threshold = float(config["data"]["value_threshold"])
        # Pytorch's usual DataLoader used for the standard FFNNs
        if network_type == "FFNN":
            df_sig, df_bkg = loaded_data
            if config.get("preparation", {}).get("use_four_vectors", False):
                if signal_fvectors is None or background_fvectors is None:
                    raise ValueError("Four-vectors are required when use_four_vectors is true!")

                preparer = FFDataPreparation(
                    signal_fvectors,
                    background_fvectors,
                    batch_size,
                    train_ratio,
                    value_threshold,
                    features_or_fvectors=signal_fvectors
                )

            else:
                features = config["features"]

                preparer = FFDataPreparation(
                    df_sig,
                    df_bkg,
                    batch_size,
                    train_ratio,
                    value_threshold,
                    features_or_fvectors=features,
                )

            train_dataset, val_dataset = preparer.prepare_data()
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
            )

            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            return train_loader, val_loader

        # We use the GeoDataLoader for the GNNs
        elif network_type in ["GNN", "LENN"]:
            if config["data"]["use_saved_graph_data"] is True:
                logging.info("Using pre-converted graph data...")
                graph_data_path = config["data"]["path_to_save_graph"]
                if os.path.exists(graph_data_path):
                    train_data_path = os.path.join(graph_data_path, "train_dataset.pt")
                    val_data_path = os.path.join(graph_data_path, "val_dataset.pt")
                    logging.info(
                        f"Loading pre-converted graph data from {graph_data_path}"
                    )
                    train_dataset = torch.load(train_data_path)
                    val_dataset = torch.load(val_data_path)
                    train_loader = GeoDataLoader(
                        train_dataset, batch_size=batch_size, shuffle=True
                    )
                    val_loader = GeoDataLoader(
                        val_dataset, batch_size=batch_size, shuffle=False
                    )
                    return train_loader, val_loader
                else:
                    raise ValueError("path_to_save_graph not found in config")
            else:
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
                train_dataset, val_dataset = preparer.prepare_GNN_data(config)
                train_loader = GeoDataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )
                val_loader = GeoDataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False
                )
                return train_loader, val_loader
        else:
            raise ValueError("Invalid network type")


class BaseDataPreparation:

    def __init__(
        self, df_sig, df_bkg, batch_size, train_ratio, value_threshold, features_or_fvectors
    ):
        self.df_sig = df_sig
        self.df_bkg = df_bkg
        self.features_or_fvectors = features_or_fvectors
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.value_threshold = value_threshold

    def mask_small_values(self, X, value_threshold):
        """
        Thresholds small values in the input array.

        Args:
            X (ndarray): The input array.
            threshold (float, optional): The threshold value. Default is 1e-6.

        Returns:
            ndarray: The input array with small values set to 0.
        """
        # raise error if threshold is not a scalar value
        if not isinstance(self.value_threshold, (float, int)):
            logging.error("Threshold must be a scalar value")
        X[X < value_threshold] = 0
        return X

    def convert_to_tensors(self):
        """
        Converts the input dataframes into PyTorch tensors and creates corresponding target tensors.

        Returns:
        - X: PyTorch tensor containing concatenated signal and background data
        - y: PyTorch tensor containing concatenated target values (1 for signal, 0 for background)
        """
        logging.info("Converting to Tensors...")

        if isinstance(self.features_or_fvectors, list):
            logging.info("Using the pre-defined features...")
            # use the pre-defined features
            X_sig = torch.tensor(self.df_sig[self.features_or_fvectors].values, dtype=torch.float32)
            X_bkg = torch.tensor(self.df_bkg[self.features_or_fvectors].values, dtype=torch.float32)
        else:
            logging.info("Using the constructed four-vectors...")
            # Use the constructed four-vectors
            X_sig = torch.tensor(self.df_sig, dtype=torch.float32)
            X_bkg = torch.tensor(self.df_bkg, dtype=torch.float32)


        logging.info(
            f"Removing small values at {self.value_threshold} threshold value..."
        )
        # veto small values for training stability
        X_sig = self.mask_small_values(X_sig, self.value_threshold)
        X_bkg = self.mask_small_values(X_bkg, self.value_threshold)

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

    def prepare_data(self) -> tuple:  # returns actual datasets (better for GNNs)
        """
        Prepares the data for training a neural network.

        Returns:
        - train_dataset (Dataset): A PyTorch Dataset object for the training data.
        - val_dataset (Dataset): A PyTorch Dataset object for the validation data.
        """
        np.random.seed(42)
        torch.manual_seed(42)

        X, y = self.convert_to_tensors()
        train_dataset, val_dataset = self.split_data(X, y)
        sample_data, sample_label = train_dataset[0]

        logging.info(f"Sample data shape before (train_dataset): {sample_data.shape}")
        logging.info(f"Sample label shape before (train_dataset): {sample_label.shape}")

        train_dataset, val_dataset = self.normalize_data(train_dataset, val_dataset)

        sample_data, sample_label = train_dataset[0]

        logging.info(f"Sample data shape after (train_dataset): {sample_data.shape}")
        logging.info(f"Sample label shape after (train_dataset): {sample_label.shape}")

        logging.info(f"Total samples: {len(X)}")
        logging.info(f"Training samples: {len(train_dataset)}")
        logging.info(f"Validation samples: {len(val_dataset)}")
        logging.info(f"Batch size: {self.batch_size}")

        return train_dataset, val_dataset


class FFDataPreparation(BaseDataPreparation):
    """
    A class for preparing data for feed-forward neural networks.

    Args:
        df_sig (pandas.DataFrame): The signal dataset.
        df_bkg (pandas.DataFrame): The background dataset.
        batch_size (int): The batch size for training.
        train_ratio (float, optional): The ratio of training data to total data.
        value_threshold (float, optional): The threshold for removing small values in the input data.
        features_or_fvectors (list or numpy.ndarray): A list of feature names or four-vectors.
    """

    def __init__(
        self, df_sig, df_bkg, batch_size, train_ratio, value_threshold, features_or_fvectors
    ):
        super().__init__(
            df_sig, df_bkg, batch_size, train_ratio, value_threshold, features_or_fvectors
        )


class GraphDataPreparation(BaseDataPreparation):
    """
    A class for preparing data in graph format for GNN training.

    Args:
        df_sig (pandas.DataFrame): A pandas dataframe containing the signal data.
        df_bkg (pandas.DataFrame): A pandas dataframe containing the background data.
        batch_size (int): The batch size for training.
        features (dict): A dictionary containing the node, edge, and global features.
        train_ratio (float): The ratio of data to use for training.
    """

    def __init__(
        self, df_sig, df_bkg, batch_size, features, train_ratio, value_threshold
    ):
        super().__init__(
            df_sig, df_bkg, batch_size, features, train_ratio, value_threshold
        )

    def _construct_edge_index(self, event, num_nodes):
        """
        Constructs edge indices for a fully connected graph of jets in each event.

        Args:
            event (pandas.Series): A pandas Series containing the jet data for a single event.
            num_nodes (int): The number of nodes in the graph.

        Returns:
            edge_index (torch.Tensor): A tensor containing the edge indices.
        """

        edge_indices = []
        # Construct edge indices for a fully connected graph
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # Avoid self-loops
                    edge_indices.append((i, j))

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        return edge_index

    def prepare_graph_components(
        self, node_features, edge_features=None, global_features=None
    ):
        """
        Prepares the components necessary to construct a PyTorch Geometric Data object for each graph.

        Args:
            node_features (numpy.ndarray): Node features for the current graph.
            edge_features (numpy.ndarray, optional): Edge features for the current graph.
            global_features (numpy.ndarray, optional): Global features for the current graph.

        Returns:
            Data: A PyTorch Geometric Data object representing the graph.
        """
        num_nodes = node_features.shape[0]

        x = torch.tensor(node_features, dtype=torch.float)

        edge_index = self._construct_edge_index(num_nodes)

        edge_attr = None
        if edge_features is not None:
            edge_attr = torch.tensor(edge_features, dtype=torch.float)

        u = None
        if global_features is not None:
            u = torch.tensor(global_features, dtype=torch.float)

        # construct the Pytorch geometric data object
        data = geo_data.Data(x=x, edge_index=edge_index)
        if edge_attr is not None:
            data.edge_attr = edge_attr
        if u is not None:
            data.u = u

        return data

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
            edge_index_sig = self._construct_edge_index(
                event_data_sig, len(node_features_sig)
            )
            edge_index_bkg = self._construct_edge_index(
                event_data_bkg, len(node_features_bkg)
            )

            unique_nodes = torch.unique(edge_index_sig)

            assert len(node_features_sig) == len(
                unique_nodes
            ), "Mismatch between number of nodes and node features"

            max_node_index = torch.max(edge_index_sig)
            assert max_node_index < len(
                node_features_sig
            ), "Node index exceeds number of nodes"

            assert edge_index_sig.max() < len(
                node_features_sig
            ), "Edge index exceeds number of signal nodes"
            assert edge_index_sig.max() < len(
                node_features_bkg
            ), "Edge index exceeds number of bkg nodes"

            # Create data object for signal and background
            n_features = len(self.features["node_features"])

            # correcting edge indices to be zero-based to prevent out-of-bounds errors when batched
            edge_index_sig = edge_index_sig - 1  # BUG
            edge_index_bkg = edge_index_bkg - 1  # BUG

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

    def normalize_data(self, data_list):
        """
        Normalizes the node features (`x`) in each graph in the dataset.

        Args:
            data_list (list of torch_geometric.data.Data): The dataset containing graph data.

        Returns:
            data_list (list of torch_geometric.data.Data): The dataset with normalized node features.
        """
        logging.info("Now normalising the input features...")

        # Concatenate all node features to calculate the mean and std
        all_x = torch.cat([data.x for data in data_list], dim=0)
        mean = all_x.mean(dim=0, keepdim=True)
        std = all_x.std(dim=0, keepdim=True)

        # Normalize node features in each Data object
        for data in data_list:
            data.x = (data.x - mean) / (std + 1e-7)

        return data_list

    def prepare_GNN_data(self, config, preloaded_data=None):
        if preloaded_data is not None:
            graph_data_list = preloaded_data
        else:
            graph_data_list = self.convert_to_graphs()

        # Normalise the data first, before splitting, to ensure consistent normalization across datasets
        normalised_data_list = self.normalize_data(graph_data_list)

        # Split the normalized graph data into training and validation datasets
        train_dataset, val_dataset = self.split_data(normalised_data_list)

        # Create DataLoaders for training and validation datasets
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        logging.info(f"Total graph samples: {len(graph_data_list)}")
        logging.info(f"Training graph samples: {len(train_dataset)}")
        logging.info(f"Validation graph samples: {len(val_dataset)}")
        logging.info(f"Batch size: {self.batch_size}")

        # Save the datasets
        # Ensure pre_training_dir is defined and accessible at this scope
        pre_training_dir = config["data"]["path_to_save_graph"]
        torch.save(train_dataset, f"{pre_training_dir}/train_dataset.pt")
        torch.save(val_dataset, f"{pre_training_dir}/val_dataset.pt")

        logging.info(f"Graphs saved to {pre_training_dir}.")

        return train_dataset, val_dataset


def verify_data(self, data_list):
    """
    Verify if the edge indices in each graph in the data list are within bounds.

    Args:
        data_list (list): A list of graphs.

    Returns:
        bool: True if all graphs have valid edge indices, False otherwise.
    """
    for i, data in enumerate(data_list):
        max_edge_index = data.edge_index.max().item()
        num_nodes = data.x.size(0)
        if max_edge_index >= num_nodes:
            print(
                f"Graph {i} has out-of-bounds edge indices: Max edge index {max_edge_index}, Number of nodes {num_nodes}"
            )
            return False
    print("All graphs have valid edge indices.")
    return True
