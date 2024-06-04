import pandas as pd
from typing import Tuple
import logging


class DataLoadingFactory:
    """A factory class for loading data based on the network type."""

    @staticmethod
    def load_data(network_type, config):
        """Load data based on the network type.

        Parameters
        ----------
        network_type : str
            The type of the network.
        config : dict
            The configuration dictionary.

        Returns
        -------
        tuple
            The loaded data.

        Raises
        ------
        ValueError
            If the network type is invalid.
        """
        if network_type == "FFNN":
            loader = FFNNDataLoader(
                config["data"]["signal_path"],
                config["data"]["background_path"],
                features=config["features"],
            )
            logging.info(
                f"DataLoadingFactory :: Loading data for {network_type} network"
            )

        elif network_type in ["GNN", "LENN"]:
            loader = DataLoader(
                config["data"]["signal_path"],
                config["data"]["background_path"],
                features=config["features"],
            )
            logging.info(
                f"DataLoadingFactory :: Loading data for {network_type} network"
            )

        elif network_type == "TransformerGCN":
            loader = TransformerGCNDataLoader(
                config["data"]["signal_path"],
                config["data"]["background_path"],
                features=config["features"],
            )
            logging.info(
                f"DataLoadingFactory :: Loading data for {network_type} network"
            )

        else:
            raise ValueError("Invalid network type")

        loaded_data = loader.load_data()

        # Move logging to after load_data()
        if network_type in ["GNN", "LENN"]:
            logging.info(
                f"DataLoadingFactory:: Loading data for {network_type} network"
            )
            logging.info(
                f"Preparing data for GNN/LENN with node_features_signal shape: {loader.node_features_sig.shape}, edge_features_signal shape: {loader.edge_features_sig.shape}"
            )

        return loaded_data


class DataLoader:
    """A class for loading data, with methods specific to graphs."""

    def __init__(self, signal_path, background_path, features):
        self.signal_path = signal_path
        self.background_path = background_path
        self.features = features

    def _read_dataframes(self):
        """Read the dataframes from the signal and background paths."""
        try:
            self.df_sig = pd.read_hdf(self.signal_path, key="df")
            self.df_bkg = pd.read_hdf(self.background_path, key="df")
        except:
            raise ValueError(
                "Error reading h5 files. Please, check your paths and try again."
            )

    def _check_columns(self):
        """Check if the required columns are present in the dataframes."""
        for col in (
            self.features["node_features"]
            + self.features["edge_features"]
            + self.features["global_features"]
        ):
            if col not in self.df_sig.columns or col not in self.df_bkg.columns:
                logging.error(f"Required column {col} not found in dataframes")
                raise ValueError(
                    "A variable you want to train with was not found in the dataframes!"
                )

    def _extract_features(self):
        """Extract the features from the dataframes."""
        self.node_features_sig = self.df_sig[self.features["node_features"]]
        self.edge_features_sig = self.df_sig[self.features["edge_features"]]
        self.global_features_sig = self.df_sig[self.features["global_features"]]

        self.node_features_bkg = self.df_bkg[self.features["node_features"]]
        self.edge_features_bkg = self.df_bkg[self.features["edge_features"]]
        self.global_features_bkg = self.df_bkg[self.features["global_features"]]

    def load_data(self) -> tuple:
        """Load the graph data.

        Returns
        -------
        tuple
            A tuple containing the loaded graph data.
        """

        self._read_dataframes()
        self._check_columns()
        self._extract_features()

        return (
            self.node_features_sig,
            self.edge_features_sig,
            self.global_features_sig,
            self.node_features_bkg,
            self.edge_features_bkg,
            self.global_features_bkg,
            self.df_sig,
            self.df_bkg,
        )


class FFNNDataLoader(DataLoader):
    """A data loader for a FFNN."""

    def load_data(self):
        self._read_dataframes()

        # Filter the dataframes based on the features
        self.df_sig = self.df_sig[self.features]
        self.df_bkg = self.df_bkg[self.features]

        return self.df_sig, self.df_bkg


class TransformerGCNDataLoader(DataLoader):
    """A data loader for a Transformer with GCN."""

    def load_data(self):
        self._read_dataframes()

        # Filter the dataframes based on the features
        self.df_sig = self.df_sig[self.features]
        self.df_bkg = self.df_bkg[self.features]

        return self.df_sig, self.df_bkg
