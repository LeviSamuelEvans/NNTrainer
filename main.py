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
from utils.config_utils import print_config_summary, print_intro
from utils.cli import handleCommandLineArgs
from utils.logging import configure_logging
from models.importer import load_networks_from_directory
from utils import LoadingFactory, PreparationFactory, FeatureFactory, DataPlotter
from utils.train import Trainer, plot_losses, plot_accuracy, plot_lr
from utils.evaluation import ModelEvaluator


def main(config, config_path):
    """
    This function loads the data, prepares the data loaders, defines the model,
    trains the model, and optionally evaluates the model.

    Args:
        config (namedtuple): A named tuple containing the configuration parameters for the program.
        config_path (str): The path to the configuration file.

    Returns:
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
    node_features_signal, edge_features_signal, global_features_signal, labels_signal,
    node_features_background, edge_features_background, global_features_background, labels_background
    ) = loaded_data
    print(loaded_data)

    # fe_config = config_dict['feature_engineering']

    # feature_maker = FeatureFactory.make(max_particles=fe_config['max_particles'],
    #                                     n_leptons=fe_config['n_leptons'],
    #                                     extra_feats=fe_config.get('extra_feats'))

    # signal_fvectors = feature_maker.get_four_vectors(signal_data)
    # background_fvectors = feature_maker.get_four_vectors(background_data)

    ### NOW need to link into data preparation...

    train_loader, val_loader = PreparationFactory.prep_data(
        network_type,
        loaded_data,
        config_dict,
    )


    ################################################
    ################################################

    # Plotting inputs
    if config_dict["data"]["plot_inputs"] == 'True':
        plot_inputs = DataPlotter(config_dict)
        plot_inputs.plot_all_features()
        plot_inputs.plot_correlation_matrix("background")
        plot_inputs.plot_correlation_matrix("signal")
    else :
        logging.info("Skipping plotting of inputs")

    logging.info("Starting the training process...")

    ################################################
    ################################################

    # Load the models from the models directory
    all_networks = load_networks_from_directory("models/networks")
    input_dim = len(config["features"])  # Number of input features from the config file
    model_name = config["model"]["name"]

    if model_name in all_networks:
        model_class = all_networks[model_name]
        if model_name == "LorentzInteractionNetwork":
            print(f"About to instantiate class {model_class} defined in {model_class.__module__}")
            model = model_class(
            )
        elif model_name == "TransformerClassifier1":
            print(f"About to instantiate class {model_class} defined in {model_class.__module__}")
            model = model_class(input_dim, 128, 4, 4)
        else:
            model = model_class(input_dim)
    else:
        logging.error(
            f"Model '{model_name}' not found. Available models are: {list(all_networks.keys())}"
        )
        return

    # Debug before training
    logging.info(f"Checking if train_loader is iterable.")
    try:
        first_batch = next(iter(train_loader))
        logging.info(f"First batch successfully retrieved: {first_batch}")
    except Exception as e:
        logging.error(f"Failed to iterate over train_loader: {e}")



    logging.info(f"Model '{model_name}' loaded. Starting training.")

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
        burn_in = config["training"]["burn_in"],
        ramp_up = config["training"]["ramp_up"],
        plateau=config["training"]["plateau"],
        ramp_down=config["training"]["ramp_down"],
        network_type=config_dict["Network_type"][0] if isinstance(config_dict["Network_type"], list) else config_dict["Network_type"],
    )

    logging.info(
        f"Training model '{model_name}' for {config['training']['num_epochs']} epochs."
    )

    trainer.train_model()

    plot_losses(trainer)
    plot_accuracy(trainer)
    plot_lr(trainer)

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

    # print the final accuracy and AUC score
    logging.info(f"Final Accuracy: {accuracy:.2f}%")
    logging.info(f"Final AUC: {roc_auc:.4f}")
    logging.info(f"Average Precision: {average_precision:.4f}")
    logging.info("Program finished successfully.")


if __name__ == "__main__":
    configure_logging()  # Set up logging
    all_networks = load_networks_from_directory(
        "models"
    )                                              # Load the models from the models directory
    logging.debug(all_networks)                    # Print the dictionary of models
    config, config_path = handleCommandLineArgs()  # Handle command line arguments
    main(config, config_path)                      # Call the main function
