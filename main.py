'''
============================================
== ttH NN Trainer Framework by Levi Evans ==
============================================

This file is the main entry point of the program. It loads the data, prepares the data loaders, defines the model,
trains the model, and evaluates the model.

- Version 1.0
    - Initial version
    - Load and train a selection of models
    - steered via yaml configuration file
    - ability to easily select hyperparameters through config file
    - ability to easily select model through config file
    - ability to easily select input features through config file

- Version 1.1
    - encapsulated plotting into a class
    - added plotting of all input features (visualize the data)
    - added plotting of inputs correlation matrix for signal and background
    - added learning rate scheduler
    - added colour loging
    - added a LENN model to the model dictionary

TODO: (Y) = Yes, (P) = Partial, (N) = No
    - Add evaluation (Y) and saving (N)
    - add plotting of all input features (visualize the data) (Y)
    - add conversion to onnx (N)
    - add logging verbosity (P)
    - further develop the folowing:
        - model architecture (N)
        - input pre-processing (normalization, data-augmentation, etc.) (P)
        - scheduler and annealing (P)
        - early stopping (Y)
        - input features used (currently using all) -> feature selection vs. use all (Y)
        - compare models seamlessyly (same input features, same hyperparameters, etc.) (N)

FUTURE:
    - add attention mechanisms
    - add multiple scheduler options, along with annealing functions
    - add graph dataset producers and call on these with config options
    - add plotting of all input features (visualize the data)
    - add conversion to onnx
    - add logging verbosity
    - add logging to file
    - gpu support
'''

#usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from utils.load import load_data
from utils.prepare import DataPreparation
from utils.train import Trainer,plot_losses,plot_accuracy
from models.importer import load_models_from_directory
from utils.cli import handleCommandLineArgs
from utils.logging import configure_logging
from utils.evaluation import evaluate_model #,plot_pr_curve
from utils.config_utils import print_config_summary,print_intro
from utils.plotting import DataPlotter



def main(config, config_path):
    """
    This function is the main entry point of the program. It loads the data, prepares the data loaders, defines the model,
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
    config_dict = config._asdict() # Temporary workaround until we can use the new config class

    # Print configuration summary
    print_config_summary(config_dict)

    # Load the data from the HDF5 files
    df_sig, df_bkg = load_data(config_dict['data']['signal_path'], config_dict['data']['background_path'], features=config_dict['features'])
    # if confi not in df_sig.columns or config.features not in df_bkg.columns:
    #     raise ValueError('Variable variable_name is not available in both the signal and background dataframes.')

    # Prepare the data loaders and plot the input features
    data_prep = DataPreparation(df_sig, df_bkg, config_dict['training']['batch_size'], features=config_dict['features'])
    train_loader, val_loader = data_prep.prepare_data()
    plot_inputs = DataPlotter(config_dict)
    #background = df_bkg[config.features] # Define the background variable
    plot_inputs.plot_all_features()
    plot_inputs.plot_correlation_matrix('background')
    plot_inputs.plot_correlation_matrix('signal')

    # Load the models from the models directory
    all_models = load_models_from_directory('models')
    input_dim = len(config.features) # Number of input features from the config file
    model_name = config.model['name']  # Assuming you have an argument named 'model_name'

    if model_name in all_models:
        model_class = all_models[model_name]
        if model_name == 'LorentzInteractionNetwork':
            model = model_class(input_dim, config.model['hidden_dim'], config.model['output_dim'])
        else:
            model = model_class(input_dim)
    else:
        logging.error(f"Model '{model_name}' not found. Available models are: {list(all_models.keys())}")
        return

    # Train the model
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
    num_epochs=config.training['num_epochs'],
    lr=config.training['learning_rate'],
    weight_decay=config.training['weight_decay'],
    patience=config.training['patience'],
    early_stopping=config.training['early_stopping'],
    use_scheduler=config.training['use_scheduler'],
    factor = config.training['factor'],
    initialise_weights=config.training['initialise_weights'],
    balance_classes=config.training['balance_classes'],)

    trainer.train_model()

    plot_losses(trainer)  # pass the Trainer object to the plot_losses() function
    plot_accuracy(trainer) # pass the Trainer object to the plot_accuracy() function

    # Now, evaluate the model..
    accuracy, roc_auc = evaluate_model(model, val_loader)
    logging.info(f"Final Accuracy: {accuracy:.2f}%")
    logging.info(f"Final AUC: {roc_auc:.4f}")


if __name__ == '__main__':
    configure_logging()  # Set up logging
    all_models = load_models_from_directory('models') # Load the models from the models directory
    logging.debug(all_models) # Print the dictionary of models
    config, config_path = handleCommandLineArgs() # Handle command line arguments
    main(config, config_path) # Call the main function