"""
============================================
== ttH NN Trainer Framework by Levi Evans ==
============================================

This file is the main entry point of the program. It loads the data, prepares the data loaders, defines the model,
trains the model, and evaluates the model.

"""

# usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from modules.config_utils import print_config_summary, print_intro
from modules.cli import handleCommandLineArgs
from modules.logging import configure_logging
from models.importer import NetworkImporter
from modules import LoadingFactory, PreparationFactory, FeatureFactory, DataPlotter
from modules.train import Trainer
from modules.evaluation import ModelEvaluator


def main(config, config_path):
    """Load the data, prepare the data loaders, define the model, train the model, and optionally evaluate the model.

    Parameters
    ----------
    config : dict
        A dictionary containing the configuration parameters for the program.
    config_path : str
        The path to the configuration file.

    Returns
    -------
    None
    """

    # Print the program intro
    print_intro()

    # Load the configuration file
    config_dict = config

    # Print configuration summary
    print_config_summary(config_dict)

    # mode = config_dict.get('mode', 'training') //next

    ################################################
    ################################################

    # Get the network type from the config file
    network_type = config_dict["Network_type"][0]  # 'FFNN' or 'GNN' currently

    # Employ the appropriate data loader and data preparation classes
    loaded_data = LoadingFactory.load_data(network_type, config_dict)

    # Unpack the loaded data into signal and background DataFrames

    if network_type == "FFNN":
        signal_data, background_data = loaded_data
    elif network_type == "GNN" or network_type == "LENN":
        (
            node_features_signal,
            edge_features_signal,
            global_features_signal,
            labels_signal,
            node_features_background,
            edge_features_background,
            global_features_background,
            labels_background,
        ) = loaded_data

    # extract features using the FeatureFactory
    signal_fvectors, background_fvectors = FeatureFactory.extract_features(config_dict, signal_data, background_data)

    # prepare the data using the PreparationFactory
    train_loader, val_loader = PreparationFactory.prep_data(
        network_type,
        loaded_data,
        config_dict,
        signal_fvectors,
        background_fvectors,
    )

    ################################################
    ################################################

    # Plotting inputs
    if config_dict["data"]["plot_inputs"] == True:
        plotter = DataPlotter(config_dict)
        plotter.plot_all_features()
        plotter.plot_correlation_matrix("background")
        plotter.plot_correlation_matrix("signal")
    else:
        logging.info("Skipping plotting of inputs")

    logging.info("Starting the training process...")

    ################################################
    ################################################

    # Load the models from the models directory and create the model using the NetworkImporter
    network_importer = NetworkImporter("models")
    model = network_importer.create_model(config)

    # Debug before training
    logging.info(f"Checking if train_loader is iterable.")
    try:
        first_batch = next(iter(train_loader)) # if we want to add a debug print statement here we can use this
        logging.info(f"First batch successfully retrieved:")
    except Exception as e:
        logging.error(f"Failed to iterate over train_loader: {e}")

    logging.info(f"Model '{config['model']['name']}' loaded. Starting training.")

    ################################################
    ################################################

    # Train the model..
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config["training"]["num_epochs"],
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        patience=config["training"]["patience"],
        early_stopping=config["training"]["early_stopping"],
        use_scheduler=config["training"]["use_scheduler"],
        factor=config["training"]["factor"],
        criterion=config["training"]["criterion"],
        initialise_weights=config["training"]["initialise_weights"],
        balance_classes=config["training"]["balance_classes"],
        use_cosine_burnin=config["training"]["use_cosine_burnin"],
        lr_init=float(config["training"]["lr_init"]),
        lr_max=float(config["training"]["lr_max"]),
        lr_final=float(config["training"]["lr_final"]),
        burn_in=config["training"]["burn_in"],
        ramp_up=config["training"]["ramp_up"],
        plateau=config["training"]["plateau"],
        ramp_down=config["training"]["ramp_down"],
        network_type=(
            config_dict["Network_type"][0]
            if isinstance(config_dict["Network_type"], list)
            else config_dict["Network_type"]
        ),
    )

    logging.info(
        f"Training model '{config['model']['name']}' for {config['training']['num_epochs']} epochs."
    )

    trainer.train_model()

    ################################################
    ################################################

    # Now, evaluate the model..

    evaluator = ModelEvaluator(
        config=config,
        model=(
            None
            if config.get("evaluation", {}).get("use_saved_model", False)
            else trainer.model
        ),
        val_loader=trainer.val_loader,
    )

    accuracy, roc_auc, average_precision = evaluator.evaluate_model()

    # pass the trainer, evaluator to the DataPlotter
    plotter = DataPlotter(config_dict, trainer, evaluator)
    plotter.plot_losses()
    plotter.plot_accuracy()
    plotter.plot_lr()
    plotter.plot_roc_curve()
    plotter.plot_confusion_matrix()
    plotter.plot_pr_curve()

    # print the final accuracy and AUC score
    logging.info(f"Final Accuracy: {accuracy:.2f}%")
    logging.info(f"Final AUC: {roc_auc:.4f}")
    logging.info(f"Average Precision: {average_precision:.4f}")
    logging.info("Program finished successfully.")


if __name__ == "__main__":

    configure_logging()  # Set up logging

    network_importer = NetworkImporter("models")

    # load the models from the models directory
    all_networks = network_importer.load_networks_from_directory()

    # print the dictionary of models
    logging.debug(all_networks)

    config, config_path = handleCommandLineArgs()

    # call main
    main(config, config_path)
