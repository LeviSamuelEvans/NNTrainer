import optuna
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate
from models.importer import NetworkImporter
from modules.train import Trainer
from modules.evaluation import ModelEvaluator
from utils import TrainerArgs
import matplotlib.pyplot as plt
import os

def objective(trial, config, train_loader, val_loader, network_type):
    """
    Objective function for hyperparameter tuning.

    Parameters
    ----------
        trial : optuna.Trial
            Optuna trial object.
        config : dict
            Configuration dictionary.
        train_loader : torch.utils.data.DataLoader
            Training data loader.
        val_loader : torch.utils.data.DataLoader
            Validation data loader.
        network_type : str
            Type of network.

    Returns
    -------
        float:
            ROC AUC metric to optimize.
    """
    # parameters to optimise
    d_model = trial.suggest_categorical("d_model", [32, 64, 128, 256])
    nhead = trial.suggest_int("nhead", 1, 6)
    # Ensure nhead is a divisor of d_model
    if d_model % nhead != 0:
        # Adjust nhead to the closest divisor of d_model
        divisors = [i for i in range(1, 9) if d_model % i == 0]
        nhead = min(divisors, key=lambda x: abs(x - nhead))

    num_layers = trial.suggest_int("num_layers", 1, 6)
    dropout = trial.suggest_float("dropout", 0.01, 0.25)
    #learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)


    # configuration dictionary with the suggested hyperparameters
    config["model"]["d_model"] = d_model
    config["model"]["nhead"] = nhead
    config["model"]["num_encoder_layers"] = num_layers
    config["model"]["dropout"] = dropout

    # create the model based on the updated configuration
    network_importer = NetworkImporter("models")
    model = network_importer.create_model(config)

    # prepare the TrainerArgs object
    trainer_args = TrainerArgs(config, model, train_loader, val_loader, network_type)

    # Create the Trainer instance
    trainer = Trainer(**trainer_args.prepare_trainer_args())
    trainer.train_model()
    evaluator = ModelEvaluator(config, trainer.model, trainer.val_loader, trainer.criterion)
    accuracy, roc_auc, average_precision, _, _, _, _ = evaluator.evaluate_model()

    return roc_auc

def tune_hyperparameters(config, train_loader, val_loader, network_type):
    """
    Tune hyperparameters using Optuna.

    Optimisation algorithm used is TPE:
        TPE is a Bayesian optimisation algorithm that uses a tree-structured space partitioning
        approach to model the probability distribution of the objective function. It constructs
        two density estimators: one for the distribution of hyperparameters that yield high
        objective function values (good configurations) and another for the distribution of
        hyperparameters that yield low objective function values (bad configurations).
        TPE then uses these estimators to guide the search towards promising hyperparameter
        configurations.

    Returns:
    --------
        tuple:
            A tuple containing the best hyperparameters and the corresponding objective value.
    """

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, config, train_loader, val_loader, network_type), n_trials=30)

    best_params = study.best_params
    best_value = study.best_value

    print("Best hyperparameters:", best_params)
    print("Best value:", best_value)

    with open('tuning_results.txt', 'w') as f:
        f.write("Best hyperparameters:\n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nBest value: {best_value}\n")

    # Save the optimization history plot as a PNG file
    fig = plot_optimization_history(study)
    fig.write_image('optimisation_history.png')
    fig.write_image('optimisation_history.pdf')

    # Save the parallel coordinate plot as a PNG file
    fig = plot_parallel_coordinate(study)
    fig.write_image('parallel_coordinate.png')
    fig.write_image('parallel_coordinate.pdf')


    return best_params, best_value