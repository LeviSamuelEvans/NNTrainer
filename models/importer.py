import os
import importlib

def load_networks_from_directory(directory):
    """
    Dynamically load all networks from a directory.

    Parameters:
    - directory (str): Name of the directory containing model modules.

    Returns:
    - dict: Dictionary containing model names as keys and model classes as values.
    """
    networks = {}
    # directory = NETWORK_DIR
    for filename in os.listdir(directory):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]  # Remove .py extension
            directory = directory.replace(
                "/", "."
            )  # Replace / with . in directory name
            module = importlib.import_module(f"{directory}.{module_name}")
            for attr_name in dir(module):
                attr_value = getattr(module, attr_name)
                if isinstance(attr_value, type):  # Check if it's a class
                    networks[attr_name] = attr_value
    return networks
