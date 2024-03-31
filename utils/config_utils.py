import logging
import re

def print_intro():
    """Prints the introduction message for the ttH Neural Network Trainer Framework.
    """
    logging.info("=" * 36)
    logging.info("NN Trainer Framework")
    logging.info(get_version_from_file())
    logging.info("=" * 36)
    logging.info("")

def print_config_summary(config):
    """Prints a summary of the configuration settings.

    Parameters:
    -----------
    config : dict
        A dictionary containing the configuration settings.
    """
    log_with_separator("Configuration Summary")

    # = Model Configuration =
    logging.info("Model Configuration:")
    logging.info("- Model Name: {}".format(config["model"]["name"]))
    logging.info("- Input Dimension: {}".format(config["model"]["input_dim"]))
    if re.search(r"transformer", config["model"]["name"], re.IGNORECASE):
        logging.info("- d_model: {}".format(config["model"]["d_model"]))
        logging.info("- nhead: {}".format(config["model"]["nhead"]))
        logging.info("- num_encoder_layers: {}".format(config["model"]["num_encoder_layers"]))
        logging.info("- dropout: {}".format(config["model"]["dropout"]))

    # = Data Configuration =
    logging.info("Data Configuration:")
    logging.info("- Signal Path: {}".format(config["data"]["signal_path"]))
    logging.info("- Background Path: {}".format(config["data"]["background_path"]))

    # = Feature Configuration =
    logging.info("Feature Configuration:")
    feature_groups = {
        "Jet Features": [f for f in config["features"] if f.startswith("jet_")],
        "Electron Features": [f for f in config["features"] if f.startswith("el_")],
        "Muon Features": [f for f in config["features"] if f.startswith("mu_")],
        "Other Features": [f for f in config["features"] if not f.startswith(("jet_", "el_", "mu_"))],
    }

    for group, features in feature_groups.items():
        if features:
            logging.info(f"- {group}:")
            for feature in features:
                logging.info(f"  - {feature}")

    # = Training Configuration =
    logging.info("Training Configuration:")
    for key, value in config["training"].items():
        logging.info(f"- {key.capitalize()}: {value}")

    logging.info("=" * 30)

def log_with_separator(message):
    """Logs a message with a separator of equal signs.

    Parameters:
    -----------
    message : str
        The message to log.
    """
    logging.info(message)
    logging.info("=" * len(message))

def get_version_from_file(filename="version.txt"):
    """Reads the version number from a file.

    Parameters:
    -----------
    filename : str
        The name of the file containing the version number.

    Returns:
    -----------
    str
        The version number.
    """
    try:
        with open(filename, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        logging.warning(f"Version file '{filename}' not found. Returning 'Unknown'.")
        return "Unknown"