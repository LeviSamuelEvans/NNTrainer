import logging


def print_intro():
    """
    Prints the introduction message for the ttH Neural Network Trainer Framework.
    """
    logging.info("====================================")
    logging.info("ttH Neural Network Trainer Framework")
    logging.info(get_version_from_file())
    logging.info("====================================")
    logging.info("")


def print_config_summary(config):
    """
    Prints a summary of the configuration settings.

    Args:
        config (dict): A dictionary containing the configuration settings.

    Returns:
        None
    """
    log_with_separator("Configuration Summary")

    # Model Configuration
    logging.info("Model Configuration:")
    logging.info("- Model Name: {}".format(config["model"]["name"]))
    # logging.info("- Input Dimension: {}".format(config['model']['input_dim']))

    # Data Configuration
    logging.info("Data Configuration:")
    logging.info("- Signal Path: {}".format(config["data"]["signal_path"]))
    logging.info("- Background Path: {}".format(config["data"]["background_path"]))
    logging.info("- Features: {}".format(", ".join(config["features"])))

    # Training Configuration
    logging.info("Training Configuration:")
    for key, value in config["training"].items():
        logging.info(f"- {key.capitalize()}: {value}")
    logging.info("=" * 30)


# helper function to log a message with a separator
def log_with_separator(message):
    """
    Logs a message with a separator of equal signs.

    Args:
        message (str): The message to log.

    Returns:
        None
    """
    logging.info(message)
    logging.info("=" * len(message))


def get_version_from_file(filename="version.txt"):
    with open(filename, "r") as file:
        return file.read().strip()
