import pandas as pd
from typing import Tuple

class DataLoadingFactory:
    @staticmethod
    def load_data(network_type, config):
        if network_type == "FFNN":
            loader = FFNNDataLoader(config['data']['signal_path'], config['data']['background_path'], features=config['features'])
        elif network_type in ["GNN", "LENN"]:
            loader = DataLoader(config['data']['signal_path'], config['data']['background_path'], features=config['features'])
        else:
            raise ValueError("Invalid network type")
        return loader.load_data()

class DataLoader:
    def __init__(self, signal_path, background_path, features):
        self.signal_path = signal_path
        self.background_path = background_path
        self.features = features

    def _read_dataframes(self):
        try:
            self.df_sig = pd.read_hdf(self.signal_path, key="df")
            self.df_bkg = pd.read_hdf(self.background_path, key="df")
        except:
            raise ValueError("Error reading h5 files. Please, check your paths and try again.")

    def _check_columns(self):
        for col in self.features['node_features'] + self.features['edge_features'] + self.features['global_features']:
            if col not in self.df_sig.columns or col not in self.df_bkg.columns:
                print(f"Required column {col} not found in dataframes")
                raise ValueError("A variable you want to train with was not found in the dataframes!")

    def _extract_features(self):
        self.node_features_sig = self.df_sig[self.features['node_features']]
        self.edge_features_sig = self.df_sig[self.features['edge_features']]
        self.global_features_sig = self.df_sig[self.features['global_features']]

        self.node_features_bkg = self.df_bkg[self.features['node_features']]
        self.edge_features_bkg = self.df_bkg[self.features['edge_features']]
        self.global_features_bkg = self.df_bkg[self.features['global_features']]

    def load_data(self) -> tuple:
        self._read_dataframes()
        self._check_columns()
        self._extract_features()

        return self.node_features_sig, self.edge_features_sig, self.global_features_sig, self.node_features_bkg, self.edge_features_bkg, self.global_features_bkg,self.df_sig,self.df_bkg

class FFNNDataLoader(DataLoader):
    def load_data(self):
        self._read_dataframes()

        # Filter the dataframes based on the features
        self.df_sig = self.df_sig[self.features]
        self.df_bkg = self.df_bkg[self.features]

        return self.df_sig, self.df_bkg
