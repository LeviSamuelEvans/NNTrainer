import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import logging
import seaborn as sns


class DataPlotter:
    """
    A class for plotting signal and background distributions for a given set of features.

    Attributes:
    signal_path (str): The path to the signal data file.
    background_path (str): The path to the background data file.
    features (list): A list of feature names to plot.

    Methods:
    __init__(self, config_dict): Initializes the DataPlotter object with the given configuration dictionary.
    plot_feature(self, feature): Plots the signal and background distributions for a given feature.
    plot_all_features(self): Plots the signal and background distributions for all features in the features list.
    plot_correlation_matrix(self, data_type): Plots the linear correlation matrix for the input features.
    """

    def __init__(self, config_dict):
        """
        Initializes the DataPlotter object with the given configuration dictionary.

        Parameters:
        config_dict (dict): A dictionary containing the configuration parameters for the DataPlotter object.

        Returns:
        None
        """
        self.signal_path = config_dict['data']['signal_path']
        self.background_path = config_dict['data']['background_path']
        self.features = config_dict['features']
        self.df_sig = pd.read_hdf(self.signal_path, key="df")
        self.df_bkg = pd.read_hdf(self.background_path, key="df")

    def plot_feature(self, feature):
        """
        Plots the signal and background distributions for a given feature.

        Parameters:
        feature (str): The name of the feature to plot.

        Returns:
        None
        """
        plt.figure()
        logging.debug(self.df_sig.head()) # Print the first few rows of the signal DataFrame
        logging.debug(self.df_bkg.head()) # Print the first few rows of the background DataFrame
        # Set the number of bins
        num_bins = 50

        sig = self.df_sig[feature]
        bkg = self.df_bkg[feature]

        # Determine the bin edges based on the data range
        min_val = min(sig.min(), bkg.min())
        max_val = max(sig.max(), bkg.max())
        bin_edges = np.linspace(min_val, max_val, num_bins)

        # Plot histograms with normalisation (i.e shape differences only)

        plt.hist(sig, bins=bin_edges, alpha=0.5, label='Signal', density=True)
        plt.hist(bkg, bins=bin_edges, alpha=0.5, label='Background', density=True)

        # Plot the distributions
        plt.xlabel(feature)
        plt.ylabel("Probability Density")
        plt.legend(loc='upper right')
        hep.atlas.label(loc=0, label="Internal", lumi="140.0", com="13")
        plt.savefig(f"plots/Inputs/{feature}.png")
        logging.info(f"Plot of {feature} saved to plots/Inputs/{feature}.png")
        # Close the plot to free up memory
        plt.close()

    def plot_all_features(self):
        """
        Plots the signal and background distributions for all features in the features list.

        Parameters:
        None

        Returns:
        None
        """
        for feature in self.features:
            self.plot_feature(feature)

    def plot_correlation_matrix(self, data_type):
        """
        Plots the linear correlation matrix for the input features.

        Parameters:
        data_type (str): Specifies which data to use. Options are 'signal' or 'background'.
        """
        if data_type == 'signal':
            data = self.df_sig
            save_path = "plots/Inputs/Signal_CorrelationMatrix.png"
        elif data_type == 'background':
            data = self.df_bkg
            save_path = "plots/Inputs/Background_CorrelationMatrix.png"
        else:
            raise ValueError("data_type must be either 'signal' or 'background'.")

        # Compute the correlation matrix
        corr_matrix = data[self.features].corr()
        print(corr_matrix,"Correlation matrix:")

        # Plot the heatmap
        plt.figure(figsize=(10, 8))  # Adjust the size as needed
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,)
        plt.yticks(rotation=0, ha='right')
        plt.xticks(rotation=45, ha='right')
        # Add title and show plot
        #plt.title(f'{data_type.capitalize()} Linear Correlation Matrix')
        hep.atlas.label(loc=0, label="Internal")
        plt.tight_layout()
        plt.savefig(save_path)
        logging.info(f"Correlation matrix for {data_type} saved to {save_path}")
        plt.close()