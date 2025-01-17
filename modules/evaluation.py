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

    def __init__(self, config, model, val_loader=None, criterion=None):
        self.config = config
        self.val_loader = val_loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fpr = None
        self.tpr = None
        self.roc_auc = None
        self.y_true = None
        self.y_pred = None
        self.y_scores = None
        self.precision = None
        self.recall = None
        self.average_precision = None
        self.model = model
        self.criterion = criterion
        self.inputs = None
        self.labels = None
        self.score = None
        self.network_type = (None,)

        if config.get("evaluation", {}).get("use_saved_model", False):
            saved_model_path = config["evaluation"]["saved_model_path"]
            self.model = model.to(self.device)
            state_dict = torch.load(saved_model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
        elif model is not None:
            self.model = model.to(self.device)
        else:
            raise ValueError(
                "Model not provided and 'use_saved_model' is False in config."
            )
        self.network_type = config["Network_type"][0]

    def _compute_pos_weight(self, all_labels):
        """Compute the positive weight for imbalanced classes."""
        positive_examples = float((all_labels == 1).sum())
        negative_examples = float((all_labels == 0).sum())

        if positive_examples == 0 or negative_examples == 0:
            raise ValueError(
                "Dataset contains no positive or no negative examples, cannot compute pos_weight."
            )

        # Simplified computation logic with logging
        weight = (
            positive_examples / negative_examples
            if positive_examples > negative_examples
            else negative_examples / positive_examples
        )
        logging.debug(
            f"Computed pos_weight: {weight}, Positives: {positive_examples}, Negatives: {negative_examples}"
        )

        return torch.tensor([weight])

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
        total_loss = 0.0

        # get device of our model after training
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for batch_data in self.val_loader:
                if isinstance(batch_data, tuple):
                    inputs, labels = batch_data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                elif isinstance(batch_data, list) and all(
                    isinstance(x, torch.Tensor) for x in batch_data
                ):
                    batch_data = [x.to(device) for x in batch_data]
                    inputs, labels = batch_data
                else:
                    batch_data = batch_data.to(device)
                    inputs = batch_data.x
                    labels = batch_data.y
                    edge_index = batch_data.edge_index
                    edge_attr = batch_data.edge_attr
                    if hasattr(batch_data, "batch"):
                        batch = batch_data.batch

                # process the data based on the network type
                if self.network_type in ["GNN", "LENN", "TransformerGCN"]:
                    # for our graph-based models
                    outputs = (
                        self.model(inputs, edge_index, edge_attr, batch)
                        if hasattr(batch_data, "batch")
                        else self.model(inputs, edge_index)
                    )
                    scores = torch.sigmoid(outputs).cpu().numpy()
                elif self.network_type == "FFNN":
                    if self.model.__class__.__name__ in [
                        "TransformerClassifier2",
                        "SetsTransformerClassifier",
                        "TransformerClassifier5",
                    ]:
                        outputs = self.model(
                            inputs,
                            inputs,
                        )
                    elif self.model.__class__.__name__ in ["TransformerClassifier9"]:
                        outputs = self.model(
                            inputs, inputs, labels
                        )  # pass inputs twice for x and x_coords!
                    else:
                        outputs = self.model(inputs)

                    scores = torch.sigmoid(outputs).cpu().numpy()
                else:
                    raise ValueError(f"Unsupported network type: {self.network_type}")

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                y_pred.extend(outputs.cpu().numpy())
                y_scores.extend(scores)
                y_true.extend(labels.cpu().numpy())

        y_pred = np.array(y_pred).squeeze()
        y_true = np.array(y_true).squeeze()
        self.y_scores = np.array(y_scores).squeeze()

        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        average_precision = average_precision_score(y_true, y_scores)
        accuracy = self.calculate_accuracy(y_true, y_pred)
        logging.info(f"Accuracy on validation set: {accuracy:.2f}%")

        # ROC curve and AUC
        self.fpr, self.tpr, thresholds = roc_curve(y_true, y_pred)
        self.roc_auc = auc(self.fpr, self.tpr)
        logging.info(f"AUC: {self.roc_auc:.4f}")
        logging.info(f"Loss: {total_loss:.4f}")

        self.y_true = y_true
        self.y_pred = y_pred

        self.precision, self.recall, _ = precision_recall_curve(y_true, y_scores)
        self.average_precision = average_precision_score(y_true, y_scores)

        avg_loss = total_loss / len(self.val_loader)
        logging.info(f"Average loss on validation set: {avg_loss:.4f}")

        return (
            accuracy,
            self.roc_auc,
            self.average_precision,
            self.model,
            self.criterion,
            self.inputs,
            self.labels,
        )

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

    def get_loss_landscape_data(self):
        return self.model, self.criterion, self.inputs, self.labels

    def get_score_distribution_data(self):
        return self.y_pred, self.y_true
