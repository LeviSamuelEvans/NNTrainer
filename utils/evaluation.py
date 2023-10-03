import torch
import logging
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import seaborn as sns
from sklearn.metrics import confusion_matrix, average_precision_score

"""
TODO:
- Add/Fix precision-recall curve
- Add F1 score
- Normalise confusion matrix
- Add support for multi-class classification
- Add future support for regression tasks
"""


def evaluate_model(model, val_loader):
    """
    Evaluate the model on the validation set.

    Args:
        model: Trained model.
        val_loader: DataLoader for validation data.

    Returns:
        accuracy: Accuracy of the model on the validation set.
        roc_auc: Area under the ROC curve.
    """
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_pred.extend(outputs.numpy())
            y_true.extend(labels.numpy())

    accuracy = 100 * correct / total
    logging.info(f"Accuracy on validation set: {accuracy:.2f}%")

    y_pred = np.array(y_pred).squeeze()
    y_true = np.array(y_true).squeeze()

    # ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    logging.info(f"AUC: {roc_auc:.4f}")

    # Plotting
    plot_roc_curve(fpr, tpr, roc_auc)
    plot_confusion_matrix(y_true, y_pred)
    #plot_pr_curve(y_true, y_pred, average_precision_score(y_true, y_pred))

    return accuracy, roc_auc

def plot_roc_curve(fpr, tpr, roc_auc):
    """
    Plots the Receiver Operating Characteristic (ROC) curve for binary classification models.

    Parameters:
    fpr (array-like): False Positive Rate values.
    tpr (array-like): True Positive Rate values.
    roc_auc (float): Area under the ROC curve.

    Returns:
    None
    """
    plt.style.use(hep.style.ROOT)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    hep.atlas.label(loc=0, label="Internal",)
    plt.tight_layout()
    plt.savefig('plots/Evaluation/roc_curve.png')
    logging.info("ROC curve plot produced and saved to 'plots/Evaluation/roc_curve.png'")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Plots a confusion matrix for binary classification.

    Parameters:
    y_true (np.ndarray): True labels of the data.
    y_pred (np.ndarray): Predicted labels of the data.

    Returns:
    None
    """
    predicted_labels = (y_pred > 0.5).astype(int)
    cm = confusion_matrix(y_true, predicted_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    #plt.title('Confusion Matrix')
    hep.atlas.label(loc=0, label="Internal",)
    plt.tight_layout()
    plt.savefig('plots/Evaluation/confusion_matrix.png')
    logging.info("Confusion matrix plot produced and saved as 'plots/Evaluation/confusion_matrix.png'")

# NOT CURRENTLY WORING :( - NEED TO FIX
# def plot_pr_curve(precision, recall, average_precision):
#     """
#     Plots the Precision-Recall (PR) curve for binary classification models.

#     Parameters:
#     precision (array-like): Precision values.
#     recall (array-like): Recall values.
#     average_precision (float): Average precision score.

#     Returns:
#     None
#     """
#     #plt.style.use(hep.style.ROOT)
#     plt.figure()
#     plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (area = %0.2f)' % average_precision)
#     plt.fill_between(recall, precision, alpha=0.2, color='darkorange')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     print("Precision - Min:", min(precision), "Max:", max(precision), "Mean:", sum(precision)/len(precision))
#     print("Recall - Min:", min(recall), "Max:", max(recall), "Mean:", sum(recall)/len(recall))

#     #plt.title('Precision-Recall curve')
#     plt.legend(loc="lower left")
#     hep.atlas.label(loc=0, label="Internal",)
#     plt.tight_layout()
#     plt.savefig('plots/Evaluation/pr_curve.png')
#     logging.info("Precision-Recall curve plot produced and saved to 'plots/Evaluation/pr_curve.png'")

# Example usage:
# from sklearn.metrics import precision_recall_curve, average_precision_score
# y_true = [0, 1, 1, 0, 1]
# y_scores = [0.1, 0.4, 0.35, 0.8, 0.65]
# precision, recall, _ = precision_recall_curve(y_true, y_scores)
# average_precision = average_precision_score(y_true, y_scores)
# plot_pr_curve(precision, recall, average_precision)
