#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys
from utils.config_utils import print_config_summary, print_intro
from utils.cli import handleCommandLineArgs
from models.importer import NetworkImporter
from modules.utils.logging import configure_logging
from modules.runners.load_data import load_data
from modules.runners.extract_features import extract_features
from modules.runners.augment_data import augment_data
from modules.runners.prepare_data import prepare_data
from modules.runners.plot_inputs import plot_inputs
from modules.runners.handle_insights import handle_insights
from modules.runners.train_model import train_model
from modules.runners.run_tuner import hyper_tuning
from modules.runners.evaluate_model import evaluate_model
from modules.evaluation import ModelEvaluator
from modules import DataPlotter


def main(config, config_path) -> None:
    """
    Main function to run the program.

    Parameter
    ---------
        config : dict
            Configuration dictionary containing program settings.
        config_path : str
            Path to the configuration file.
    """

    print_intro()
    print_config_summary(config)

    # ============================================================
    # Loading data
    network_type, loaded_data = load_data(config)
    signal_data, background_data = None, None
    if network_type in ["FFNN", "TransformerGCN"]:
        signal_data, background_data = loaded_data

    # ============================================================
    # creating/extracting features
    features = extract_features(config, network_type, signal_data, background_data)

    use_four_vectors = config["preparation"].get("use_four_vectors", False)

    # perform data augmentation
    augment_data(config, use_four_vectors, features)

    # ============================================================
    # data preparation
    train_loader, val_loader = prepare_data(
        network_type, loaded_data, config, *features
    )

    # ============================================================
    plotter = DataPlotter(config)
    plot_inputs(config, plotter)

    # ============================================================
    # create the model to train
    network_importer = NetworkImporter("models")
    model = network_importer.create_model(config)

    run_mode = config.get("run_mode", "train")

    # running model insights
    handle_insights(run_mode, config, network_type, val_loader, model)

    # ============================================================
    logging.info("Starting the training process...")
    try:
        # let's check we can retrieve the first batch okay...
        next(iter(train_loader))
        logging.info("First batch successfully retrieved.")
    except Exception as e:
        logging.error(f"Failed to iterate over train_loader: {e}")

    logging.info(f"Model '{config['model']['name']}' loaded. Starting training.")

    # ============================================================
    # hyperparameter tuning via Optuna
    hyper_tuning(config, train_loader, val_loader, network_type)

    # ============================================================
    # training the model
    trainer = train_model(config, model, train_loader, val_loader, run_mode)

    plotter = DataPlotter(config, trainer=trainer, evaluator=None)

    # ============================================================
    evaluator = ModelEvaluator(
        config=config,
        model=trainer.model,
        val_loader=val_loader,
        criterion=trainer.criterion,
    )
    # plotting metrics
    plotter.evaluator = evaluator

    # evaluate the model
    evaluate_model(config, trainer, plotter, evaluator)

    logging.info("Program finished successfully!")


if __name__ == "__main__":
    configure_logging()

    network_importer = NetworkImporter("models")
    all_networks = network_importer.load_networks_from_directory()
    logging.debug(all_networks)

    config, config_path = handleCommandLineArgs()
    main(config, config_path)
