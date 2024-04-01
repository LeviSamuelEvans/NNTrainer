import torch
import logging
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
)

"""
TODO:
- Add F1 score
- Add support for multi-class classification
- Add future support for regression tasks
"""


class ModelEvaluator:
    """Class to evaluate a trained model on the validation set.

    Attributes:
    -----------
        config:
            The configuration dictionary.
        model:
            The trained model.
        val_loader:
            DataLoader for the validation data.
        device:
            The device on which the model is evaluated.
        fpr:
            False positive rate.
        tpr:
            True positive rate.
        roc_auc:
            Area under the ROC curve.
        y_true:
            True labels.
        y_pred:
            Predicted labels.
        precision:
            Precision.
        recall:
            Recall.
        average_precision:
            Average precision score.
    """
    def __init__(self, config, model, val_loader=None):
        self.config = config
        self.val_loader = val_loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fpr = None
        self.tpr = None
        self.roc_auc = None
        self.y_true = None
        self.y_pred = None
        self.precision = None
        self.recall = None
        self.average_precision = None

        if config.get("evaluation", {}).get("use_saved_model", False):
            saved_model_path = config["evaluation"]["saved_model_path"]
            self.model = torch.load(saved_model_path, map_location=self.device)
        elif model is not None:
            self.model = model.to(self.device)
        else:
            raise ValueError(
                "Model not provided and 'use_saved_model' is False in config."
            )

    def evaluate_model(self):
        """Evaluate the model on the validation set.

        Returns:
        -----------
            accuracy:
                Accuracy of the model on the validation set.
            roc_auc:
                Area under the ROC curve.
            average_precision:
                Average precision score.
        """
        self.model.eval()
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        y_scores = []

        # get device of our model after training
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if self.model.__class__.__name__ == "TransformerClassifier2" or self.model.__class__.__name__ == "SetsTransformerClassifier":
                    outputs = self.model(inputs, inputs)  # pass inputs twice for x and x_coords
                else:
                    outputs = self.model(inputs)
                scores = torch.sigmoid(outputs).cpu().numpy()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                y_pred.extend(outputs.cpu().numpy())
                y_scores.extend(scores)
                y_true.extend(labels.cpu().numpy())

        y_pred = np.array(y_pred).squeeze()
        y_true = np.array(y_true).squeeze()

        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        average_precision = average_precision_score(y_true, y_scores)
        accuracy = self.calculate_accuracy(y_true, y_pred)
        logging.info(f"Accuracy on validation set: {accuracy:.2f}%")

        # ROC curve and AUC
        self.fpr, self.tpr, thresholds = roc_curve(y_true, y_pred)
        self.roc_auc = auc(self.fpr, self.tpr)
        logging.info(f"AUC: {self.roc_auc:.4f}")

        self.y_true = y_true
        self.y_pred = y_pred

        self.precision, self.recall, _ = precision_recall_curve(y_true, y_scores)
        self.average_precision = average_precision_score(y_true, y_scores)

        return accuracy, self.roc_auc, self.average_precision

    def calculate_accuracy(self, y_true, y_scores, threshold=0.5):
        y_pred = [1 if score > threshold else 0 for score in y_scores]
        correct = sum([1 for true, pred in zip(y_true, y_pred) if true == pred])
        accuracy = 100 * correct / len(y_true)
        return accuracy

    def get_roc_curve_data(self):
        """Get the data for the ROC curve."""
        return self.fpr, self.tpr, self.roc_auc

    def get_confusion_matrix_data(self):
        """Get the data for the confusion matrix."""
        return self.y_true, self.y_pred

    def get_pr_curve_data(self):
        """Get the data for the Precision-Recall curve."""
        return self.precision, self.recall, self.average_precision