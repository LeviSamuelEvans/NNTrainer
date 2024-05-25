"""
===========================
== NN Trainer Framework  ==
============================

This file is the main entry point of the programme.

It loads the data, prepares the data loaders, defines the model,
trains the model, and evaluates the model.

"""

# usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys
from utils.config_utils import print_config_summary, print_intro
from modules.cli import handleCommandLineArgs
from modules.logging import configure_logging
from utils import TrainerArgs
from models.importer import NetworkImporter
from modules import LoadingFactory, PreparationFactory, FeatureFactory, DataPlotter
from modules import Augmenter
from modules.train import Trainer
from modules.evaluation import ModelEvaluator
from modules.hyperparameter_tuning import tune_hyperparameters
from insights.att_maps import AttentionMap
from insights.embed_maps import EmbeddingMaps


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

    #================================================================================================
    # DATA LOADING

    # Get the network type from the config file
    network_type = config_dict["Network_type"][0]  # 'FFNN' or 'GNN' currently

    # Employ the appropriate data loader and data preparation classes
    loaded_data = LoadingFactory.load_data(network_type, config_dict)

    # Unpack the loaded data into signal and background DataFrames

    if network_type == "FFNN" or network_type == "TransformerGCN":
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

    #================================================================================================
    # FEATURE EXTRACTION

    # extract features using the FeatureFactory
    signal_fvectors, background_fvectors, signal_edges, signal_edge_attr, background_edges, background_edge_attr = FeatureFactory.extract_features(
    config_dict, signal_data, background_data
    )

    use_four_vectors = config["preparation"].get("use_four_vectors", False)


    #================================================================================================
    # DATA AUGMENTATION

    try:
        if config["preparation"].get("augment_data", False) and use_four_vectors:
            logging.info("Augmenter :: Proceeding to augment the input data...")
            augmenter = Augmenter.standard_augmentation(use_four_vectors)

            if "augmentation_methods" not in config["preparation"]:
                augmenter.perform_all_augmentations(signal_fvectors, background_fvectors)
            else:
                if config["preparation"]["augmentation_methods"].get("phi-rotation", False):
                    augmenter.rotate_phi(signal_fvectors, background_fvectors)
                if config["preparation"]["augmentation_methods"].get("eta-reflection", False):
                    augmenter.reflect_eta(signal_fvectors, background_fvectors)
                if config["preparation"]["augmentation_methods"].get("translate_eta_phi", False):
                    augmenter.translate_eta_phi(signal_fvectors, background_fvectors)
                if config["preparation"]["augmentation_methods"].get("energy-variation", False):
                    augmenter.scale_energy_momentum(signal_fvectors, background_fvectors)
            logging.info("Augmenter :: Data augmentation complete.")
        else:
            logging.info("Augmenter :: Skipping data augmentation.")
    except Exception as e:
        logging.error(f"Failed to augment the input data: {e}")

    #================================================================================================
    # DATA PREPARATION

    # prepare the data using the PreparationFactory
    train_loader, val_loader = PreparationFactory.prep_data(
        network_type,
        loaded_data,
        config_dict,
        signal_fvectors,
        background_fvectors,
        signal_edges,
        signal_edge_attr,
        background_edges,
        background_edge_attr,
    )

    #================================================================================================
    # INPUT PLOTTING

    # Plotting inputs
    if config_dict["data"]["plot_inputs"] == True:
        plotter = DataPlotter(config_dict)
        plotter.plot_all_features()
        plotter.plot_correlation_matrix("background")
        plotter.plot_correlation_matrix("signal")
    else:
        logging.info("DataPlotter :: Skipping plotting of inputs")

    #================================================================================================

    network_importer = NetworkImporter("models")
    model = network_importer.create_model(config)

    #================================================================================================
    # INSIGHTS (WIP!!)

    def get_run_mode():
        run_mode = config_dict.get("run_mode", "train")
        return run_mode

    while True:
        run_mode = get_run_mode()
        if run_mode == "train":
            break
        elif run_mode == "attention_map":
            attention_map = AttentionMap(network_type, config_dict)
            loaded_model = attention_map.load_model()
            attention_weights = attention_map.get_attention_weights(val_loader, loaded_model)
            feature_names = config_dict["features"]

            # plot attention weights and entropy
            attention_map.plot_attention_weights(attention_weights, feature_names)
            attention_map.plot_attention_entropy(attention_weights, feature_names)
            sys.exit(0) # now, exit, as that's all we wanted to do :p

        elif run_mode == "evaluate":
            logging.info(f"Proceeding to evaluation without training...")
            logging.info(f"Using the model from {config_dict['evaluation']['saved_model_path']}")
            break

        elif run_mode == "embed_maps":
            logging.info("Proceeding to extract and visualise the positional embeddings...")
            embed_maps = EmbeddingMaps(model, network_type, config_dict)

            for batch in val_loader:
                print(f"Batch: {batch}")
                x, labels = batch
                break

            embed_maps.run(x, labels)
            sys.exit(0) # now, exit, as that's all we wanted to do :p

        else:
            logging.error(f"Invalid run mode: {run_mode}")
            sys.exit(0)


    #================================================================================================
    # MODEL TRAINING

    logging.info("Starting the training process...")

    # Load the models from the models directory and create the model using the NetworkImporter

    # Debug before training
    logging.info(f"Checking if train_loader is iterable.")
    try:
        first_batch = next(
            iter(train_loader)
        )  # if we want to add a debug print statement here we can use this
        logging.info(f"First batch successfully retrieved:")
    except Exception as e:
        logging.error(f"Failed to iterate over train_loader: {e}")

    logging.info(f"Model '{config['model']['name']}' loaded. Starting training.")


    #=============================================================================
    # Parameter optimising via optuna ---> configure study options and name study
    try:
        if config["model"].get("tune_hyperparams", False):
            logging.info("Proceeding to hyperparameter tuning...")
            best_params, best_value = tune_hyperparameters(config, train_loader, val_loader, network_type)
            print(f"Best hyperparameters: {best_params}")
            print(f"Best value: {best_value}")

            config["model"]["d_model"] = best_params["d_model"]
            config["model"]["nhead"] = best_params["nhead"]
            config["model"]["num_encoder_layers"] = best_params["num_layers"]
            config["model"]["dropout"] = best_params["dropout"]
        else:
            logging.info("Skipping hyperparameter tuning.")
    except Exception as e:
        logging.error(f"Failed to tune hyperparameters: {e}")
    #=============================================================================


    # prepare the arguments to pass to the Trainer class
    trainer_args = TrainerArgs(config, model, train_loader, val_loader, network_type)

    # create the Trainer class and proceed to train the model
    trainer = Trainer(**trainer_args.prepare_trainer_args())

    logging.info(
        f"Trainer :: Training model '{config['model']['name']}' for {config['training']['num_epochs']} epochs."
    )

    run_mode = get_run_mode()
    if run_mode != "evaluate":
        trainer.train_model()
        # save metadata to JSON for further use...
        trainer.save_metrics_to_json(f"{trainer.model_save_path}/metadata.json")

    #================================================================================================
    # MODEL EVALUATION

    evaluator = ModelEvaluator(
        config=config,
        model=(
            trainer.model
            if config.get("evaluation", {}).get("use_saved_model", False)
            else trainer.model
        ),
        val_loader=trainer.val_loader,
        criterion=trainer.criterion,
    )

    accuracy, roc_auc, average_precision, model, criterion, inputs, labels = evaluator.evaluate_model()

    # pass the trainer, evaluator to the DataPlotter
    plotter = DataPlotter(config_dict, trainer, evaluator)

    # plot all val and eval metrics
    plotter.plot_all()

    if config_dict["evaluation"].get("plot_loss_landscape", False):
        # takes some time to run
        plotter.plot_loss_landscape(model, criterion, inputs, labels)

    # print the final accuracy and AUC score
    logging.info(f"Evaluator :: Final Accuracy: {accuracy:.2f}%")
    logging.info(f"Evaluator :: Final AUC: {roc_auc:.4f}")
    logging.info(f"Evaluator :: Average Precision: {average_precision:.4f}")
    logging.info("Program finished successfully!")
    #================================================================================================


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
