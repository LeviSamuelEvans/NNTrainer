import torch
import logging
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import seaborn as sns
from sklearn.metrics import confusion_matrix, average_precision_score, precision_recall_curve

"""
TODO:
- Add F1 score
- Add support for multi-class classification
- Add future support for regression tasks
"""

class ModelEvaluator:
    def __init__(self, config, model, val_loader=None):
        """
        Initializes the ModelEvaluator with the given model and validation DataLoader.

        Args:
            model: The trained PyTorch model to evaluate.
            val_loader: DataLoader for the validation dataset.
        """
        self.config = config
        self.val_loader = val_loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if config.get('evaluation', {}).get('use_saved_model', False):
            saved_model_path = config['evaluation']['saved_model_path']
            self.model = torch.load(saved_model_path, map_location=self.device)
        elif model is not None:
            self.model = model.to(self.device)
        else:
            raise ValueError("Model not provided and 'use_saved_model' is False in config.")
        

    def evaluate_model(self):
        """
        Evaluate the model on the validation set.

        Args:
            model: Trained model.
            val_loader: DataLoader for validation data.

        Returns:
            accuracy: Accuracy of the model on the validation set.
            roc_auc: Area under the ROC curve.
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
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        logging.info(f"AUC: {roc_auc:.4f}")

        # Plotting
        self.plot_roc_curve(fpr, tpr, roc_auc)
        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_pr_curve(precision, recall, average_precision)

        return accuracy, roc_auc, average_precision
    
    def calculate_accuracy(self, y_true, y_scores, threshold=0.5):
        y_pred = [1 if score > threshold else 0 for score in y_scores]
        correct = sum([1 for true, pred in zip(y_true, y_pred) if true == pred])
        accuracy = 100 * correct / len(y_true)
        return accuracy

    def plot_roc_curve(self, fpr, tpr, roc_auc):
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
        hep.atlas.label(loc=0, label="Internal",fontsize=12)
        plt.tight_layout()
        plt.savefig('/scratch4/levans/tth-network/plots/Evaluation/roc_curve.png')
        logging.info("ROC curve plot produced and saved to '/scratch4/levans/tth-network/plots/Evaluation/roc_curve.png'")


    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
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

        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_percentage = cm_percentage * 100

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', cbar=False, annot_kws={"size": 12})
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        #plt.title('Confusion Matrix')
        hep.atlas.label(loc=0, label="Internal", fontsize=12)
        plt.tight_layout()
        plt.savefig('/scratch4/levans/tth-network/plots/Evaluation/confusion_matrix.png')
        logging.info("Confusion matrix plot produced and saved as '/scratch4/levans/tth-network/plots/Evaluation/confusion_matrix.png'")


    def plot_pr_curve(self, precision, recall, average_precision):
        """
        Plots the Precision-Recall (PR) curve for binary classification models.

        Parameters:
        precision (array-like): Precision values.
        recall (array-like): Recall values.
        average_precision (float): Average precision score.

        Returns:
        None
        """
        plt.style.use(hep.style.ROOT)
        plt.figure()
        step = max(1, len(precision) // 1000) 
        plt.plot(recall[::step], precision[::step], color='darkorange', lw=2, label=f'PR curve (area = {average_precision:.2f})')
        plt.fill_between(recall[::step], precision[::step], alpha=0.2, color='darkorange')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        print("Precision - Min:", min(precision), "Max:", max(precision), "Mean:", sum(precision)/len(precision))
        print("Recall - Min:", min(recall), "Max:", max(recall), "Mean:", sum(recall)/len(recall))

        #plt.title('Precision-Recall curve')
        plt.legend(loc="lower left")
        hep.atlas.label(loc=0, label="Internal",fontsize=12)
        plt.tight_layout()
        plt.savefig('plots/Evaluation/pr_curve.png')
        logging.info("Precision-Recall curve plot produced and saved to '/scratch4/levans/tth-network/plots/Evaluation/pr_curve.png")