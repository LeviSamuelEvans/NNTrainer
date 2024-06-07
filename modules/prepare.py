from logging import config
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch_geometric.data as geo_data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.data import Batch
import logging
from tqdm import tqdm
import os


# ================================================================================================
# THE DATA PREPARATION FACTORY


class DataPreparationFactory:
    """Factory class for the type of data preparation to use.

    Attributes
    ----------
        network_type : str
            The type of network.
        loaded_data : tuple
            A tuple containing the loaded data.
        config : dict
            The configuration settings.
        signal_fvectors: list (optional)
            The signal four-vectors. Defaults to None.
        background_fvectors : list (optional)
            The background four-vectors. Defaults to None.

    Returns
    -------
        tuple
            A tuple containing the train loader and validation loader.

    Raises
    ------
        ValueError
            If an invalid network type is provided.
            If four-vectors are required but not provided.
            If the path to save graph data is not found in the configuration.
    """

    @staticmethod
    def prep_data(
        network_type,
        loaded_data,
        config,
        signal_fvectors=None,
        background_fvectors=None,
        signal_edges=None,
        signal_edge_attr=None,
        background_edges=None,
        background_edge_attr=None,
    ):

        batch_size = config["training"]["batch_size"]
        train_ratio = config["data"]["train_ratio"]
        value_threshold = float(config["data"]["value_threshold"])

        # Pytorch's usual DataLoader used for the standard FFNNs
        if network_type == "FFNN":
            df_sig, df_bkg = loaded_data
            if config.get("preparation", {}).get("use_four_vectors", False):
                if signal_fvectors is None or background_fvectors is None:
                    raise ValueError(
                        "Four-vectors are required when use_four_vectors is true!"
                    )

                preparer = FFDataPreparation(
                    signal_fvectors,
                    background_fvectors,
                    batch_size,
                    train_ratio,
                    value_threshold,
                    features_or_fvectors=signal_fvectors,
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

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
            )

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

        elif network_type == "TransformerGCN":
            df_sig, df_bkg = loaded_data
            batch_size = config["training"]["batch_size"]
            train_ratio = config["data"]["train_ratio"]
            value_threshold = float(config["data"]["value_threshold"])
            preparer = TransformerGCNDataPreparation(
                signal_fvectors,
                background_fvectors,
                batch_size,
                train_ratio,
                value_threshold,
                features_or_fvectors=signal_fvectors,
            )

            train_dataset, val_dataset = preparer.prepare_transformer_gcn_data(
                signal_edges, signal_edge_attr, background_edges, background_edge_attr
            )

            train_loader = GeoDataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = GeoDataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )

            return train_loader, val_loader

        else:
            raise ValueError("Invalid network type")


# ================================================================================================
# BASE CLASS FOR DATA PREPARATION


class BaseDataPreparation:
    """The base class for data preparation.

    Attributes
    ----------
    df_sig : pandas.DataFrame
        The signal dataset.
    df_bkg : pandas.DataFrame
        The background dataset.
    batch_size : int
        The batch size for training.
    train_ratio : float
        The ratio of training data to total data.
    value_threshold : float
        The threshold for removing small values in the input data.
    features_or_fvectors : list or numpy.ndarray
        A list of feature names or four-vectors.
    """

    def __init__(
        self,
        df_sig,
        df_bkg,
        batch_size,
        train_ratio,
        value_threshold,
        features_or_fvectors,
    ):
        self.df_sig = df_sig
        self.df_bkg = df_bkg
        self.features_or_fvectors = features_or_fvectors
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.value_threshold = value_threshold

    def mask_small_values(self, X, value_threshold):
        """Set small values below threshold in the input array to 0.

        Parameters
        ----------
        X : ndarray
            The input array.
        value_threshold : float, optional
            The threshold value. Default is 1e-6.

        Returns
        -------
        ndarray
            The input array with small values set to 0.

        Raises
        ------
        ValueError
            If the threshold is not a scalar value.
        """
        # raise error if threshold is not a scalar value
        if not isinstance(self.value_threshold, (float, int)):
            logging.error("Threshold must be a scalar value")
        X[X < value_threshold] = 0
        return X

    def convert_to_tensors(self):
        """Converts the input dataframes into PyTorch tensors and creates corresponding target tensors.

        Returns
        -------
        tuple
            A tuple containing:
            - X: PyTorch tensor containing concatenated signal and background data
            - y: PyTorch tensor containing concatenated target values
        """
        logging.info("PreparationFactory :: Converting to Tensors...")

        if isinstance(self.features_or_fvectors, list):
            logging.info("Using the pre-defined features...")
            # Use the pre-defined features
            X_sig = torch.tensor(
                self.df_sig[self.features_or_fvectors].values, dtype=torch.float32
            )
            X_bkg = torch.tensor(
                self.df_bkg[self.features_or_fvectors].values, dtype=torch.float32
            )
        else:
            logging.info("PreparationFactory :: Using the constructed four-vectors...")
            # Use the constructed four-vectors
            X_sig = torch.tensor(self.df_sig, dtype=torch.float32)
            X_bkg = torch.tensor(self.df_bkg, dtype=torch.float32)

        logging.info(
            f"PreparationFactory :: Removing small values at {self.value_threshold} threshold value..."
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
        """Splits the input data into training and validation datasets based on the given train_ratio.

        Also implements shuffling of the data, in order to avoid any potential bias in the training
        and validation sets.

        Parameters
        ----------
        X : torch.Tensor
            Input data tensor.
        y : torch.Tensor
            Target data tensor.

        Returns
        -------
        tuple
            A tuple containing:
            - train_dataset (torch.utils.data.dataset.TensorDataset): Training dataset.
            - val_dataset (torch.utils.data.dataset.TensorDataset): Validation dataset.
        """
        logging.info(
            "PreparationFactory :: Creating the training and validation datasets..."
        )

        # Shuffle the data
        indices = torch.randperm(X.size(0))
        X = X[indices]
        y = y[indices]

        dataset = TensorDataset(X, y)
        train_size = int(self.train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        return train_dataset, val_dataset

    def normalize_features(self, X, mean, std):
        """Normalises the input features in the data using mean and standard deviation.

        Note: This assumes we are running with MET and btagging as the last two columns.

        This will work regardless of the type of 4vec representations we decide to use,
        but for tests without b-tagging and MET, we should have a work around for this
        in the future.
        """
        # squeeze the mean and std tensors to remove additional dimensions
        mean_squeezed = mean.squeeze()
        std_squeezed = std.squeeze()

        # Do not normalise the discrete labels like btagging, MET flag, etc! (add particle ID to this list)
        X[:, :-2] = (X[:, :-2] - mean_squeezed[:9]) / (std_squeezed[:9] + 1e-7)
        return X

    def normalize_data(self, train_dataset, val_dataset):
        """Normalises the data in the train and validation datasets using mean and standard deviation.

        We normalise the validation dataset using the mean and standard deviation from the training dataset.
        We can view the standardisation as part of the model building process, and as such we should include this
        when assessing the model.

        Parameters
        ----------
        train_dataset : list
            A list of tuples containing the training data and labels.
        val_dataset : list
            A list of tuples containing the validation data and labels.

        Returns
        -------
        tuple
            A tuple containing the normalised training and validation datasets.
        """
        logging.info(" Normalising the input data...")
        # normalise the training data
        X_train = [x[0] for x in train_dataset]
        X_train_stacked = torch.stack(X_train)
        mean_train = X_train_stacked[:, :-2].mean(dim=(0, 1), keepdim=True)
        std_train = X_train_stacked[:, :-2].std(dim=(0, 1), keepdim=True)

        train_dataset = [
            (self.normalize_features(x, mean_train, std_train).squeeze(), y)
            for x, y in train_dataset
        ]

        # normalise the validation data using the mean and std from the training data
        val_dataset = [
            (self.normalize_features(x, mean_train, std_train).squeeze(), y)
            for x, y in val_dataset
        ]
        return train_dataset, val_dataset

    def prepare_data(self) -> tuple:  # returns actual datasets (better for GNNs)
        """Prepares the data for training a neural network.

        Returns
        -------
        tuple
            A tuple containing:
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


# ================================================================================================
# FEED-FOWARD DATA PREPARATION


class FFDataPreparation(BaseDataPreparation):
    """A class for preparing data for feed-forward neural networks.

    Attributes
    ----------
    df_sig : pandas.DataFrame
        The signal dataset.
    df_bkg : pandas.DataFrame
        The background dataset.
    batch_size : int
        The batch size for training.
    train_ratio : float, optional
        The ratio of training data to total data.
    value_threshold : float, optional
        The threshold for removing small values in the input data.
    features_or_fvectors : list or numpy.ndarray
        A list of feature names or four-vectors.
    """

    def __init__(
        self,
        df_sig,
        df_bkg,
        batch_size,
        train_ratio,
        value_threshold,
        features_or_fvectors,
    ):
        super().__init__(
            df_sig,
            df_bkg,
            batch_size,
            train_ratio,
            value_threshold,
            features_or_fvectors,
        )


# ================================================================================================
# GRAPH DATA PREPARATION


class GraphDataPreparation(BaseDataPreparation):
    """A class for preparing data in graph format for GNN training.

    Attributes
    ----------
    df_sig : pandas.DataFrame
        A pandas dataframe containing the signal data.
    df_bkg : pandas.DataFrame
        A pandas dataframe containing the background data.
    batch_size : int
        The batch size for training.
    features : dict
        A dictionary containing the notrain_ratio : float
        The ratio of data to use for training.
    value_threshold : float
        The threshold for removing small values in the input data.
    """

    def __init__(
        self, df_sig, df_bkg, batch_size, features, train_ratio, value_threshold
    ):
        super().__init__(
            df_sig, df_bkg, batch_size, features, train_ratio, value_threshold
        )

    def _construct_edge_index(self, event, num_nodes):
        """Constructs edge indices for a fully connected graph in each event.

        Parameters
        ----------
        event : pandas.Series
            A pandas Series containing the jet data for a single event.
        num_nodes : int
            The number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            A tensor containing the edge indices.
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
        """Prepares the components necessary to construct a PyTorch Geometric Data object for each graph.

        Parameters
        ----------
        node_features : numpy.ndarray
            Node features for the current graph.
        edge_features : numpy.ndarray, optional
            Edge features for the current graph.
        global_features : numpy.ndarray, optional
            Global features for the current graph.

        Returns
        -------
        Data
            A PyTorch Geometric Data object representing the graph.
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
        """Converts the input dataframes into graph format.

        Returns
        -------
        list
            A list of PyTorch Geometric Data objects representing the graphs.
        """
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
        """Splits the list of Data objects into training and validation datasets.

        Parameters
        ----------
        data_list : list
            A list of torch_geometric.data.Data objects.

        Returns
        -------
        tuple
            A tuple containing:
            - train_dataset : (list) A list of Data objects for training.
            - val_dataset (list): A list of Data objects for validation.
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
        """Normalises the node features (`x`) in each graph in the dataset.

        Parameters
        ----------
        data_list : list of torch_geometric.data.Data
            The dataset containing graph data.

        Returns
        -------
        list of torch_geometric.data.Data
            The dataset with normalised node features.
        """
        logging.info("Now normalising the input features...")

        # Concatenate all node features to calculate the mean and std
        all_x = torch.cat([data.x for data in data_list], dim=0)
        mean = all_x.mean(dim=0, keepdim=True)
        std = all_x.std(dim=0, keepdim=True)

        # Normalise node features in each Data object
        for data in data_list:
            data.x = (data.x - mean) / (std + 1e-7)

        return data_list

    def prepare_GNN_data(self, config, preloaded_data=None):
        """Prepares the data for training a GNN model.

        Parameters
        ----------
        config : dict
            The configuration settings.
        preloaded_data : list, optional
            Pre-loaded graph data. If provided, skips the graph construction step.

        Returns
        -------
        tuple
            A tuple containing:
            - train_dataset (list): A list of Data objects for training.
            - val_dataset (list): A list of Data objects for validation.

        Raises
        ------
        ValueError
            If the path to save graph data is not found in the configuration.
        """
        if preloaded_data is not None:
            graph_data_list = preloaded_data
        else:
            graph_data_list = self.convert_to_graphs()

        # Normalise the data first, before splitting, to ensure consistent normalisation across datasets
        normalised_data_list = self.normalize_data(graph_data_list)

        # Split the normalised graph data into training and validation datasets
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
        """Verify if the edge indices in each graph in the data list are within bounds.

        Parameters
        ----------
        data_list : list
            A list of graphs.

        Returns
        -------
        bool
            True if all graphs have valid edge indices, False otherwise.
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


# ================================================================================================
# TRANSFORMERS WITH GCN CLASSIFIER DATA PREPARATION
class TransformerGCNDataPreparation(BaseDataPreparation):
    """A class for preparing data for Transformer with GCN training.

    This transformer is designed to work with the 4-vectors and
    additional edge attributes, for instance the separation in eta
    and phi between all objects in the event.

    Attributes
    ----------
    df_sig : pandas.DataFrame
        The signal dataset.
    df_bkg : pandas.DataFrame
        The background dataset.
    batch_size : int
        The batch size for training.
    train_ratio : float, optional
        The ratio of training data to total data.
    value_threshold : float, optional
        The threshold for removing small values in the input data.
    four_vectors : numpy.ndarray
        The four-vectors used as input features.
    """

    def __init__(
        self,
        df_sig,
        df_bkg,
        batch_size,
        train_ratio,
        value_threshold,
        features_or_fvectors,
    ):
        super().__init__(
            df_sig,
            df_bkg,
            batch_size,
            train_ratio,
            value_threshold,
            features_or_fvectors,
        )

    def convert_to_tensors(self):
        """Converts the input dataframes into PyTorch tensors and creates corresponding target tensors.

        Returns
        -------
        tuple
            A tuple containing:
            - X_sig: PyTorch tensor containing signal data
            - X_bkg: PyTorch tensor containing background data
            - y_sig: PyTorch tensor containing target values for signal data
            - y_bkg: PyTorch tensor containing target values for background data
        """
        logging.info("PreparationFactory :: Converting to Tensors...")
        if isinstance(self.features_or_fvectors, list):
            logging.info("Using the pre-defined features...")

            # Use the pre-defined features
            X_sig = torch.tensor(
                self.df_sig[self.features_or_fvectors].values, dtype=torch.float32
            )
            X_bkg = torch.tensor(
                self.df_bkg[self.features_or_fvectors].values, dtype=torch.float32
            )
        else:
            logging.info("PreparationFactory :: Using the constructed four-vectors...")
            # Use the constructed four-vectors
            X_sig = torch.tensor(self.df_sig, dtype=torch.float32)
            X_bkg = torch.tensor(self.df_bkg, dtype=torch.float32)

        logging.info(
            f"PreparationFactory :: Removing small values at {self.value_threshold} threshold value..."
        )
        # veto small values for training stability
        X_sig = self.mask_small_values(X_sig, self.value_threshold)
        X_bkg = self.mask_small_values(X_bkg, self.value_threshold)

        y_sig = torch.ones((X_sig.shape[0], 1), dtype=torch.float32)
        y_bkg = torch.zeros((X_bkg.shape[0], 1), dtype=torch.float32)

        # calc mean and std for normalisation
        X = torch.cat((X_sig, X_bkg), dim=0)
        mean = X.mean(dim=0, keepdim=True)
        std = X.std(dim=0, keepdim=True)

        return X_sig, X_bkg, y_sig, y_bkg, mean, std

    @staticmethod
    def check_for_nans_and_infs(tensor, tensor_name):
        if torch.isnan(tensor).any():
            logging.error(f"NaNs found in {tensor_name}")
        if torch.isinf(tensor).any():
            logging.error(f"Infs found in {tensor_name}")

    def normalize_node_features(self, X, mean, std):
        return (X - mean) / (std + 1e-7)

    def normalize_node_features(self, X, mean, std):
        # Do not normalise the discrete labels like btagging, MET flag, etc! (add particle ID to this list)
        X[:, :, :-2] = (X[:, :, :-2] - mean[:, :, :-2]) / (std[:, :, :-2] + 1e-7)
        self.check_for_nans_and_infs(X, "Normalized Node Features")
        return X

    def normalize_invariant_mass_min_max(self, edge_attr):
        """Normalizes the last value (invariant mass) in the edge attributes using min-max normalization.
        This is done to prevent non-physical negative values from arising.
        """
        invariant_mass = edge_attr[:, -1]
        min_val = invariant_mass.min()
        max_val = invariant_mass.max()

        # Handle the case where min_val and max_val are the same to avoid division by zero
        if min_val == max_val:
            logging.warning(
                f"Min and Max values are the same: {min_val}. Setting normalized mass to zero."
            )
            edge_attr[:, -1] = 0
        else:
            normalized_mass = (invariant_mass - min_val) / (max_val - min_val)
            edge_attr[:, -1] = normalized_mass

        self.check_for_nans_and_infs(edge_attr, "Normalized Edge Attributes")
        return edge_attr

    def prepare_transformer_gcn_data(
        self, signal_edges, signal_edge_attr, background_edges, background_edge_attr
    ):
        np.random.seed(42)
        torch.manual_seed(42)

        X_sig, X_bkg, y_sig, y_bkg, mean, std = self.convert_to_tensors()

        logging.info(
            f"PreparationFactory :: Normalising the node features using mean and std..."
        )
        # normalise the node features

        X_sig = self.normalize_node_features(X_sig, mean, std)
        X_bkg = self.normalize_node_features(X_bkg, mean, std)

        # convert to tensors
        signal_edges = [torch.tensor(edges, dtype=torch.long) for edges in signal_edges]
        signal_edge_attr = [
            torch.tensor(attr, dtype=torch.float32) for attr in signal_edge_attr
        ]
        background_edges = [
            torch.tensor(edges, dtype=torch.long) for edges in background_edges
        ]
        background_edge_attr = [
            torch.tensor(attr, dtype=torch.float32) for attr in background_edge_attr
        ]

        logging.info(
            f"PreparationFactory :: Splitting the data into training and validation datasets..."
        )

        # commented out for tests with dynamic edge scores
        # Normalise the invariant mass in the edge attributes
        # signal_edge_attr = [
        #     self.normalize_invariant_mass_min_max(attr) for attr in signal_edge_attr
        # ]
        # background_edge_attr = [
        #     self.normalize_invariant_mass_min_max(attr) for attr in background_edge_attr
        # ]

        # split data
        train_dataset_sig, val_dataset_sig = self.split_data(X_sig, y_sig)
        train_dataset_bkg, val_dataset_bkg = self.split_data(X_bkg, y_bkg)

        # split edges and edge attributes together
        train_size_sig = int(self.train_ratio * len(signal_edges))
        train_size_bkg = int(self.train_ratio * len(background_edges))

        train_edges_sig = signal_edges[:train_size_sig]
        val_edges_sig = signal_edges[train_size_sig:]
        train_edge_attr_sig = signal_edge_attr[:train_size_sig]
        val_edge_attr_sig = signal_edge_attr[train_size_sig:]

        train_edges_bkg = background_edges[:train_size_bkg]
        val_edges_bkg = background_edges[train_size_bkg:]
        train_edge_attr_bkg = background_edge_attr[:train_size_bkg]
        val_edge_attr_bkg = background_edge_attr[train_size_bkg:]

        # prepare graphs
        train_graphs_sig = self._prepare_graphs(
            train_dataset_sig, train_edges_sig, train_edge_attr_sig
        )
        val_graphs_sig = self._prepare_graphs(
            val_dataset_sig, val_edges_sig, val_edge_attr_sig
        )
        train_graphs_bkg = self._prepare_graphs(
            train_dataset_bkg, train_edges_bkg, train_edge_attr_bkg
        )
        val_graphs_bkg = self._prepare_graphs(
            val_dataset_bkg, val_edges_bkg, val_edge_attr_bkg
        )

        # Combine signal and background graphs
        train_graphs = train_graphs_sig + train_graphs_bkg
        val_graphs = val_graphs_sig + val_graphs_bkg

        # Find the maximum edge index across all tensors in the lists
        max_index_sig = max(
            torch.max(edges).item() for edges in signal_edges if edges.numel() > 0
        )
        max_index_bkg = max(
            torch.max(edges).item() for edges in background_edges if edges.numel() > 0
        )

        return train_graphs, val_graphs

    def _prepare_graphs(self, dataset, edges_list, edge_attr_list):
        graphs = []

        for (x, y), edges, edge_attr in tqdm(
            zip(dataset, edges_list, edge_attr_list), desc="Preparing Graphs"
        ):
            self.check_for_nans_and_infs(x, "Node Features")
            self.check_for_nans_and_infs(edges, "Edge Indices")
            self.check_for_nans_and_infs(edge_attr, "Edge Attributes")
            graph = geo_data.Data(
                x=x, edge_index=edges, edge_attr=edge_attr, y=y.unsqueeze(0)
            )
            graphs.append(graph)

        self._example_graph(num_printed=0, num_examples=2, graph=graph)

        return graphs

    def _example_graph(self, num_printed, num_examples, graph):
        "Inspect the first few graphs in the dataset."
        if num_printed < num_examples:
            logging.info(f"Example Graph {num_printed + 1}:")
            logging.info("Node Features (x):")
            logging.info(graph.x)
            logging.info("Edge Indices (edge_index):")
            logging.info(graph.edge_index)
            logging.info("Edge Attributes (edge_attr):")
            logging.info(graph.edge_attr)
            logging.info("Graph Label (y):")
            logging.info(graph.y)
            logging.info("---------------")
            num_printed += 1
