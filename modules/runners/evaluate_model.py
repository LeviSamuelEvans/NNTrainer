import logging
from modules.evaluation import ModelEvaluator


def evaluate_model(config, trainer, plotter, evaluator):
    """Evaluate the model."""
    accuracy, roc_auc, average_precision, model, criterion, inputs, labels = (
        evaluator.evaluate_model()
    )
    plotter.plot_all()

    if config["evaluation"].get("plot_loss_landscape", False):
        plotter.plot_loss_landscape(model, criterion, inputs, labels)

    logging.info(f"Evaluator :: Final Accuracy: {accuracy:.2f}%")
    logging.info(f"Evaluator :: Final AUC: {roc_auc:.4f}")
    logging.info(f"Evaluator :: Average Precision: {average_precision:.4f}")
