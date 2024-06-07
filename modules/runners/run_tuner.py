import logging
from modules.hyperparameter_tuning import tune_hyperparameters
import sys


def hyper_tuning(config, train_loader, val_loader, network_type):
    """Perform hyperparameter tuning."""
    if config["model"].get("tune_hyperparams", False):
        try:
            logging.info("Proceeding to hyperparameter tuning...")
            best_params, best_value = tune_hyperparameters(
                config, train_loader, val_loader, network_type
            )
            logging.info(f"Best hyperparameters: {best_params}")
            logging.info("Tuning :: Best value: %s", best_value)

            config["model"].update(best_params)
        except Exception as e:
            logging.error(f"Failed to tune hyperparameters: {e}")

        logging.info("Best configuration saved to txt file.")
        logging.info("Hyperparameter tuning completed. Exiting...")
        # exit after performing the tuning study
        sys.exit()
