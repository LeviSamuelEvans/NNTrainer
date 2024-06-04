import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import logging
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.metrics import roc_curve, auc
import re
import torch
from mpl_toolkits.mplot3d import Axes3D


class DataPlotter:
    """A class for plotting data distributions and model performance.

    Attributes
    ----------
    config_dict : dict
        A dictionary containing the configuration parameters for the DataPlotter object.
    signal_path : str
        The path to the signal data.
    background_path : str
        The path to the background data.
    features : list
        A list of features to plot.
    df_sig : pd.DataFrame
        The signal DataFrame.
    df_bkg : pd.DataFrame
        The background DataFrame.

    Methods
    -------
    plot_feature(feature)
        Plots the signal and background distributions for a given feature.
    plot_all_features()
        Plots the signal and background distributions for all features in the features list.
    plot_correlation_matrix(data_type)
        Plots the linear correlation matrix for the input features.
    plot_losses()
        Plots the training and validation losses for a given trainer object.
    plot_accuracy()
        Plots the training and validation accuracy over the number of epochs.
    plot_lr()
        Plots the learning rate over the number of epochs.
    plot_roc_curve()
        Plots the Receiver Operating Characteristic (ROC) curve for binary classification models.
    plot_confusion_matrix()
        Plots the confusion matrix for binary classification models.
    plot_pr_curve()
        Plots the Precision-Recall curve for binary classification models.
    """

    def __init__(self, config_dict, trainer=None, evaluator=None):
        """Initialises the DataPlotter object with the given configuration dictionary.

        Parameters
        ----------
        config_dict : dict
            A dictionary containing the configuration parameters for the DataPlotter object.
        trainer : Trainer
            An instance of the Trainer class containing the training and validation losses.
        evaluator : ModelEvaluator
            An instance of the ModelEvaluator class containing the evaluation metrics.

        Returns
        -------
        None
        """
        self.config_dict = config_dict
        self.trainer = trainer
        self.evaluator = evaluator
        self.setup_data_plotter()

    # =========================================================================
    # HELPER FUNCTIONS

    def _convert_to_gev(self, feature):
        """Helper function to convert features to GeV for plotting."""
        self.df_sig[feature] *= 0.001
        self.df_bkg[feature] *= 0.001

    def _remove_outliers(self, feature):
        """Removes outliers from the signal and background DataFrames for a given feature.

        Parameters
        ----------
        feature : str
            The name of the feature to remove outliers from.
        """
        # calculate z-scores for signal and background
        z_scores_sig = np.abs(
            (self.df_sig[feature] - self.df_sig[feature].mean())
            / self.df_sig[feature].std()
        )
        z_scores_bkg = np.abs(
            (self.df_bkg[feature] - self.df_bkg[feature].mean())
            / self.df_bkg[feature].std()
        )

        # now remove the outliers for plotting
        self.df_sig = self.df_sig[(z_scores_sig < 3)]
        self.df_bkg = self.df_bkg[(z_scores_bkg < 3)]

    def _remove_zeroes(self, feature):
        """Removes zeroes from the signal and background DataFrames for a given feature.

        Parameters
        ----------
        feature : str
            The name of the feature to remove zeroes from.
        """
        self.df_sig = self.df_sig[(self.df_sig[feature] != 0)]
        self.df_bkg = self.df_bkg[(self.df_bkg[feature] != 0)]

    def _fancy_titles(self, feature):
        """Helper function to convert feature names to fancy titles for inputs."""
        prefix_pt = ["jet_pt", "el_pt", "mu_pt"]
        prefix_phi = ["jet_phi", "el_phi", "mu_phi"]
        prefix_eta = ["jet_eta", "el_eta", "mu_eta"]
        prefix_e = ["jet_e", "el_e", "mu_e"]

        if feature == "nJets":
            return "Jet Multiplicity"
        elif feature == "HT_all":
            return r"$\it{H_{T}^{\text{all}}}$" + " [GeV]"
        elif any(re.match(prefix + ".*", feature) for prefix in prefix_pt):
            if "jet" in feature:
                return (
                    r"Jet $p_{T}$" + " " + feature.split("_")[2].capitalize() + " [GeV]"
                )
            if "el" in feature:
                return r"Electron $p_{T}$" + " " + " [GeV]"
            if "mu" in feature:
                return r"Muon $p_{T}$" + " " + " [GeV]"
        elif any(re.match(prefix + ".*", feature) for prefix in prefix_phi):
            if "jet" in feature:
                return r"Jet $\phi$" + " " + feature.split("_")[2].capitalize()
            if "el" in feature:
                return r"Electron $\phi$" + " " + feature.split("_")[2].capitalize()
            if "mu" in feature:
                return r"Muon $\phi$" + " " + feature.split("_")[2].capitalize()
        elif any(re.match(prefix + ".*", feature) for prefix in prefix_eta):
            if "jet" in feature:
                return r"Jet $\eta$" + " " + feature.split("_")[2].capitalize()
            if "el" in feature:
                return r"Electron $\eta$" + " " + feature.split("_")[2].capitalize()
            if "mu" in feature:
                return r"Muon $\eta$" + " " + feature.split("_")[2].capitalize()
        elif any(re.match(prefix + ".*", feature) for prefix in prefix_e):
            if "jet" in feature:
                return (
                    r"Jet Energy" + " " + feature.split("_")[2].capitalize() + " [GeV]"
                )
            if "el" in feature:
                return r"Electron Energy" + " " + " [GeV]"
            if "mu" in feature:
                return r"Muon Energy" + " " + " [GeV]"
        elif feature == "dRbb_avg_Sort4":
            return r"$\it{\Delta R_{bb}^{avg}}$"
        elif feature == "dRlepbb_MindR_70":
            return r"$\it{\Delta R_{lep,bb}^{min}}$"
        elif feature == "dEtabb_MaxdEta_70":
            return r"$\it{\Delta\eta_{max}^{70}}$"
        elif feature == "H0_all":
            return r"$\it{H_{0}^{all}}$"
        elif feature == "H1_all":
            return r"$\it{H_{1}^{all}}$"
        elif feature == "H2_jets":
            return r"$\it{H_{2}^{jets}}$"
        elif feature == "Centrality_all":
            return r"$\it{Centrality^{all}}$"
        elif feature == "met_met":
            return r"$\it{E_{T}^{]text{miss}}}$"
        else:
            return feature

    def setup_data_plotter(self):

        self.plot_save_path = self.config_dict.get("data", {}).get(
            "plot_save_path", "plots/"
        )

        plt.style.use(hep.style.ROOT)

        if self.trainer:
            self.train_losses = self.trainer.train_losses
            self.val_losses = self.trainer.val_losses
            self.train_accuracies = self.trainer.train_accuracies
            self.val_accuracies = self.trainer.val_accuracies
            self.learning_rates = self.trainer.learning_rates
            self.num_epochs = len(self.train_losses)

        if "data" in self.config_dict:
            self.signal_path = self.config_dict["data"]["signal_path"]
            self.background_path = self.config_dict["data"]["background_path"]
            self.features = self.config_dict["features"]
            self.df_sig = pd.read_hdf(self.signal_path, key="df")
            self.df_bkg = pd.read_hdf(self.background_path, key="df")

    @staticmethod
    def interpolate_models(state_dict, direction, alpha):
        """Interpolates between the model state and a direction in the parameter space."""
        interpolated_state_dict = {}
        for name, param in state_dict.items():
            if name in direction:
                interpolated_state_dict[name] = param + alpha * direction[name]
            else:
                interpolated_state_dict[name] = param
        return interpolated_state_dict

    @staticmethod
    def random_direction(model):
        """Generates a random direction in the parameter space of the model."""
        direction = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                direction[name] = torch.randn_like(param)
        return direction

    def compute_loss_landscape(
        self, model, criterion, data, target, alpha_range, num_points
    ):
        """Computes the loss landscape along a random direction."""
        losses = []
        direction = self.random_direction(model)
        for alpha in np.linspace(alpha_range[0], alpha_range[1], num_points):
            interpolated_state_dict = self.interpolate_models(
                model.state_dict(), direction, alpha
            )
            model.load_state_dict(interpolated_state_dict)
            model.eval()
            with torch.no_grad():
                output = model(data)
                loss = criterion(output, target)
                losses.append(loss.item())
        return losses

    # =========================================================================
    # PLOT ALL VALIDATION AND EVALUATION METRICS

    def plot_all(self):
        """Calls all plot methods.

        These methods include:
        - plot_losses: The training and validation losses.
        - plot_accuracy: The training and validation accuracy.
        - plot_lr: The learning rate over the number of epochs.
        - plot_roc_curve: The Receiver Operating Characteristic (ROC) curve.
        - plot_confusion_matrix: The confusion matrix.
        - plot_pr_curve: The Precision-Recall curve.
        - plot_score_distribution: The distribution of the model's decision scores.
        """
        self.plot_losses()
        self.plot_accuracy()
        self.plot_lr()
        self.plot_roc_curve()
        self.plot_confusion_matrix()
        self.plot_pr_curve()
        self.plot_score_distribution()

    # =========================================================================
    # PLOT DATA DISTRIBUTIONS

    def plot_feature(self, feature):
        """Plots the signal and background distributions for a given feature.

        Parameters
        ----------
        feature : str
            The name of the feature to plot.
        """
        # make the directory if it does not exist
        os.makedirs(f"{self.plot_save_path}Inputs/", exist_ok=True)

        plt.figure()
        logging.debug(self.df_sig.head())
        logging.debug(self.df_bkg.head())
        # Set the number of bins
        num_bins = 50

        if feature == "HT_all":
            self._convert_to_gev(feature)

        prefixes = ["jet_pt", "el_pt", "mu_pt", "jet_e", "el_e", "mu_e"]
        if any(re.match(prefix + "_*", feature) for prefix in prefixes):
            self._convert_to_gev(feature)
            self._remove_outliers(feature)
            self._remove_zeroes(feature)

        sig = self.df_sig[feature]
        bkg = self.df_bkg[feature]

        # Determine the bin edges based on the data range
        min_val = min(sig.min(), bkg.min())
        max_val = max(sig.max(), bkg.max())
        if feature == "HT_all":
            min_val = 200
            max_val = 2000
        bin_edges = np.linspace(min_val, max_val, num_bins)

        if feature == "nJets":
            num_bins = 8
            bin_edges = np.arange(5, 12, 1)

        # Plot histograms with normalisation (i.e shape differences only)

        plt.hist(sig, bins=bin_edges, alpha=0.5, label="Signal", density=True)
        plt.hist(bkg, bins=bin_edges, alpha=0.5, label="Background", density=True)

        # apply some fancy xlabels
        feature_xlabel = self._fancy_titles(feature)

        # Plot the distributions
        plt.xlabel(feature_xlabel)
        plt.ylabel("Probability Density")
        plt.legend(loc="upper right")
        hep.atlas.label(loc=0, label="Internal", lumi="140.0", com="13")
        plt.savefig(f"{self.plot_save_path}Inputs/{feature}.png")
        logging.info(
            f"DataPlotter :: Plot of {feature} saved to {self.plot_save_path}Inputs/{feature}.png"
        )
        # Close the plot to free up memory
        plt.close()

    def plot_all_features(self):
        """Plots the signal and background distributions for all features in the features list."""
        for feature in self.features:
            self.plot_feature(feature)

    # =========================================================================
    # PLOT CORRELATION MATRIX

    def plot_correlation_matrix(self, data_type):
        """Plots the linear correlation matrix for the input features.

        Parameters
        ----------
        data_type : str
            Specifies which data to use. Options are 'signal' or 'background'.
        """

        plt.style.use(hep.style.ATLAS)
        if data_type == "signal":
            data = self.df_sig
            save_path = f"{self.plot_save_path}Inputs/Signal_CorrelationMatrix.png"
        elif data_type == "background":
            data = self.df_bkg
            save_path = f"{self.plot_save_path}Inputs/Background_CorrelationMatrix.png"
        else:
            raise ValueError("data_type must be either 'signal' or 'background'.")

        # Compute the correlation matrix
        corr_matrix = data[self.features].corr()

        # Plot the heatmap
        plt.figure(figsize=(10, 8))  # Adjust the size as needed
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )
        plt.yticks(rotation=0, ha="right")
        plt.xticks(rotation=45, ha="right")
        # Add title and show plot
        # plt.title(f'{data_type.capitalize()} Linear Correlation Matrix')
        hep.atlas.label(loc=0, label="Internal")
        plt.tight_layout()
        plt.savefig(save_path)
        logging.info(
            f"DataPlotter :: Correlation matrix for {data_type} saved to {save_path}"
        )
        plt.close()

    # =========================================================================
    # PLOT LOSS CURVE

    def plot_losses(self):
        """Plots the training and validation losses stored in this object.

        The losses are expected to be stored in the `train_losses` and `val_losses`
        attributes of this object.
        """

        # make the directory if it does not exist
        os.makedirs(f"{self.plot_save_path}Validation/", exist_ok=True)

        # Determine the number of epochs based on the length of train_losses (e.g if Early Stopping is used, this will be less than num_epochs)
        actual_epochs = len(self.train_losses)
        self.num_epochs = actual_epochs

        plt.style.use(hep.style.ATLAS)
        plt.plot(np.arange(self.num_epochs), self.train_losses, label="Training loss")
        plt.plot(np.arange(self.num_epochs), self.val_losses, label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        hep.atlas.label(
            loc=0,
            label="Internal",
        )
        plt.tight_layout()
        plt.savefig(f"{self.plot_save_path}Validation/loss.png")
        logging.info(
            f"DataPlotter :: Loss plot saved to {self.plot_save_path}Validation/loss.png"
        )

    # =========================================================================
    # PLOT ACCURACY

    def plot_accuracy(self):
        """Plots the training and validation accuracy over the number of epochs.

        The accuracies are expected to be stored in the `train_accuracies` and `val_accuracies`
        attributes of this object.
        """
        plt.clf()  # Clear previous plot
        plt.style.use(hep.style.ATLAS)
        plt.plot(
            np.arange(self.num_epochs),
            self.train_accuracies,
            label="Training accuracy",
        )
        plt.plot(
            np.arange(self.num_epochs),
            self.val_accuracies,
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
        plt.savefig(f"{self.plot_save_path}Validation/accuracy.png")
        plt.savefig(f"{self.plot_save_path}Validation/accuracy.pdf")
        logging.info(
            f"DataPlotter :: Accuracy plot saved to {self.plot_save_path}Validation/accuracy.png"
        )

    # =========================================================================
    # PLOT LEARNING-RATE CURVE

    def plot_lr(self):
        """Plots the learning rate over the number of epochs.

        The learning rates are expected to be stored in the `learning_rates` attribute of this object.
        """
        plt.clf()  # Clear previous plot
        plt.style.use(hep.style.ATLAS)
        plt.plot(
            np.arange(self.num_epochs),
            self.learning_rates,
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
        plt.savefig(f"{self.plot_save_path}Validation/learning_rate.png")
        plt.savefig(f"{self.plot_save_path}Validation/learning_rate.pdf")
        logging.info(
            f"DataPlotter :: Learning rate plot saved to {self.plot_save_path}Validation/learning_rate.png"
        )

    # =========================================================================
    # PLOT ROC CURVE

    def plot_roc_curve(self):
        """Plots the Receiver Operating Characteristic (ROC) curve for binary classification models.

        The ROC curve is a graphical representation of the true positive rate (sensitivity)
        against the false positive rate (1 - specificity) for different thresholds.
        """

        # make the directory if it does not exist
        os.makedirs(f"{self.plot_save_path}Evaluation/", exist_ok=True)

        if self.evaluator is None or self.evaluator.roc_auc is None:
            logging.warning(
                "Evaluator or ROC AUC data not available. Cannot plot ROC curve."
            )
            return

        fpr = self.evaluator.fpr
        tpr = self.evaluator.tpr
        roc_auc = self.evaluator.roc_auc

        plt.style.use(hep.style.ROOT)
        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        hep.atlas.label(loc=0, label="Internal", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.plot_save_path}Evaluation/roc_curve.png")
        plt.savefig(f"{self.plot_save_path}Evaluation/roc_curve.pdf")
        logging.info(
            f"DataPlotter :: ROC curve plot produced and saved to '{self.plot_save_path}Evaluation/roc_curve.png'"
        )

    # =========================================================================
    # PLOT CONFUSION MATRIX

    def plot_confusion_matrix(self):
        """Plots the confusion matrix for binary classification models."""

        if (
            self.evaluator is None
            or self.evaluator.y_true is None
            or self.evaluator.y_pred is None
        ):
            logging.warning(
                "Evaluator or confusion matrix data not available. Cannot plot confusion matrix."
            )
            return

        y_true = self.evaluator.y_true
        y_pred = self.evaluator.y_pred

        predicted_labels = (y_pred > 0.5).astype(int)
        cm = confusion_matrix(y_true, predicted_labels)
        cm_percentage = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_percentage = cm_percentage * 100

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm_percentage,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=False,
            annot_kws={"size": 12},
            xticklabels=[0, 1],
            yticklabels=[0, 1],
        )
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        hep.atlas.label(loc=0, label="Internal", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.plot_save_path}Evaluation/confusion_matrix.png")
        plt.savefig(f"{self.plot_save_path}Evaluation/confusion_matrix.pdf")
        logging.info(
            f"DataPlotter :: Confusion matrix plot produced and saved as '{self.plot_save_path}Evaluation/confusion_matrix.png'"
        )

    # =========================================================================
    # PLOT PRECISION_RECALL CURVE

    def plot_pr_curve(self):
        """Plots the Precision-Recall curve for binary classification models."""

        if (
            self.evaluator is None
            or self.evaluator.precision is None
            or self.evaluator.recall is None
        ):
            logging.warning(
                "Evaluator or PR curve data not available. Cannot plot PR curve."
            )
            return

        precision = self.evaluator.precision
        recall = self.evaluator.recall
        average_precision = self.evaluator.average_precision

        plt.style.use(hep.style.ROOT)
        plt.figure()
        step = max(1, len(precision) // 1000)
        plt.plot(
            recall[::step],
            precision[::step],
            color="darkorange",
            lw=2,
            label=f"PR curve (area = {average_precision:.2f})",
        )
        plt.fill_between(
            recall[::step], precision[::step], alpha=0.2, color="darkorange"
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")
        hep.atlas.label(loc=0, label="Internal", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.plot_save_path}Evaluation/pr_curve.png")
        plt.savefig(f"{self.plot_save_path}Evaluation/pr_curve.pdf")
        logging.info(
            f"DataPlotter :: Precision-Recall curve plot produced and saved to '{self.plot_save_path}Evaluation/pr_curve.png"
        )

    # =========================================================================
    # PLOT SCORE DISTRIBUTION

    def plot_score_distribution(self):
        """
        Plots the distribution of the model's decision scores for each class.

        Args:
            scores (np.ndarray):
                The decision scores from the model.
            true_labels (np.ndarray):
                The true labels for the data.
        """
        plt.style.use(hep.style.ROOT)
        scores = self.evaluator.y_scores
        true_labels = self.evaluator.y_true

        signal_indices = np.where(true_labels == 1)[0]
        background_indices = np.where(true_labels == 0)[0]

        plt.figure()
        plt.hist(
            scores[signal_indices],
            bins=50,
            alpha=0.5,
            label="Signal",
            color="blue",
            density=True,
        )
        plt.hist(
            scores[background_indices],
            bins=50,
            alpha=0.5,
            label="Background",
            color="red",
            density=True,
        )
        plt.xlabel("Score")
        plt.ylabel("Normalised Entries")
        plt.legend(loc="best")
        plt.savefig(f"{self.plot_save_path}Evaluation/score_distribution.png")
        plt.savefig(f"{self.plot_save_path}Evaluation/score_distribution.pdf")
        logging.info(
            f"DataPlotter :: Score distribution plot saved to '{self.plot_save_path}Evaluation/score_distribution.png'"
        )

    # =========================================================================
    # PLOT LOSS FUNCTION

    def plot_loss_landscape(
        self,
        model,
        criterion,
        data,
        target,
        alpha_range=[-1.0, 1.0],
        beta_range=[-1.0, 1.0],
        num_points=100,
    ):
        """Plots the loss landscape for a given model and data."""

        model = self.evaluator.model
        criterion = self.evaluator.criterion
        data = self.evaluator.inputs
        target = self.evaluator.labels

        # get a grid of points in the parameter space
        alphas = np.linspace(alpha_range[0], alpha_range[1], num_points)
        betas = np.linspace(beta_range[0], beta_range[1], num_points)
        alphas, betas = np.meshgrid(alphas, betas)

        # compute values of the loss for each point in the grid
        losses = np.zeros_like(alphas)
        for i in range(num_points):
            for j in range(num_points):
                alpha = alphas[i, j]
                beta = betas[i, j]
                losses[i, j] = self.compute_loss_landscape(
                    model, criterion, data, target, [alpha, beta], 1
                )[0]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        loss_plot = ax.plot_surface(
            alphas, betas, losses, cmap="viridis", edgecolor="none", alpha=0.6
        )
        ax.set_xlabel("alpha", fontsize=12)
        ax.set_ylabel("beta", fontsize=12)
        ax.set_zlabel("Loss", fontsize=12)
        fig.colorbar(loss_plot, shrink=0.5, aspect=5)
        ax.view_init(elev=30, azim=30)
        plt.tight_layout()

        plt.tight_layout()
        plt.savefig(f"{self.plot_save_path}Evaluation/loss_landscape.png")
        logging.info(
            f"DataPlotter :: Loss landscape plot saved to '{self.plot_save_path}Evaluation/loss_landscape.png'"
        )
