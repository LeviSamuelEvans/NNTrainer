'''
============================================
== ttH NN Trainer Framework by Levi Evans ==
============================================

This file is the main entry point of the program. It loads the data, prepares the data loaders, defines the model,
trains the model, and evaluates the model.

'''

#usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from utils import LoadingFactory, PreparationFactory, FeatureFactory, DataPlotter
from utils.train import Trainer, plot_losses, plot_accuracy
from models.importer import load_networks_from_directory
from utils.cli import handleCommandLineArgs
from utils.logging import configure_logging
from utils.evaluation import evaluate_model #,plot_pr_curve
from utils.config_utils import print_config_summary, print_intro


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
    
    ################################################
    ################################################

    # Get the network type from the config file
    network_type = config_dict['Network_type'][0]  # 'FFNN' or 'GNN' currently
    

    # Employ the appropriate data loader and data preparation classes
    loaded_data = LoadingFactory.load_data(network_type, config_dict)
    
    # Unpack the loaded data into signal and background DataFrames
    signal_data, background_data = loaded_data
    
    fe_config = config_dict['feature_engineering']
    
    feature_maker = FeatureFactory.make(max_particles=fe_config['max_particles'],
                                        n_leptons=fe_config['n_leptons'],
                                        extra_feats=fe_config.get('extra_feats'))

    
    signal_fvectors = feature_maker.get_four_vectors(signal_data)
    background_fvectors = feature_maker.get_four_vectors(background_data)
    
    ### NOW need to link into data preparation...
    
    train_loader, val_loader = PreparationFactory.prep_data(network_type, loaded_data, config_dict)

    ################################################
    ################################################
    
    # Plotting inputs
    plot_inputs = DataPlotter(config_dict)
    plot_inputs.plot_all_features()
    plot_inputs.plot_correlation_matrix('background')
    plot_inputs.plot_correlation_matrix('signal')


    logging.info("Starting the training process...")

    ################################################
    ################################################
    
    # Load the models from the models directory
    all_networks = load_networks_from_directory('models/networks')
    input_dim = len(config['features']) # Number of input features from the config file
    model_name = config['model']['name']

    if model_name in all_networks:
        model_class = all_networks[model_name]
        if model_name == 'LorentzInteractionNetwork':
            model = model_class(input_dim, config['model']['hidden_dim'], config['model']['output_dim'])
        else:
            model = model_class(input_dim)
    else:
        logging.error(f"Model '{model_name}' not found. Available models are: {list(all_networks.keys())}")
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
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
    num_epochs=config['training']['num_epochs'],
    lr=config['training']['learning_rate'],
    weight_decay=config['training']['weight_decay'],
    patience=config['training']['patience'],
    early_stopping=config['training']['early_stopping'],
    use_scheduler=config['training']['use_scheduler'],
    factor = config['training']['factor'],
    criterion=config['training']['criterion'],
    initialise_weights=config['training']['initialise_weights'],
    balance_classes=config['training']['balance_classes'],
    network_type=config_dict['Network_type'],)

    logging.info (f"Training model '{model_name}' for {config['training']['num_epochs']} epochs.")
    trainer.train_model()

    plot_losses(trainer)  # pass the Trainer object to the plot_losses() function
    plot_accuracy(trainer) # pass the Trainer object to the plot_accuracy() function
    
    ################################################
    ################################################

    # Now, evaluate the model..
    accuracy, roc_auc = evaluate_model(model, val_loader)
    logging.info(f"Final Accuracy: {accuracy:.2f}%")
    logging.info(f"Final AUC: {roc_auc:.4f}")


if __name__ == '__main__':
    configure_logging()  # Set up logging
    all_networks = load_networks_from_directory('models') # Load the models from the models directory
    logging.debug(all_networks) # Print the dictionary of models
    config, config_path = handleCommandLineArgs() # Handle command line arguments
    main(config, config_path) # Call the main function