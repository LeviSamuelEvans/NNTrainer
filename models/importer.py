import os
import importlib

MODELS_DIR = 'models'

def load_models_from_directory(directory):
    """
    Dynamically load all models from a directory.

    Parameters:
    - directory (str): Name of the directory containing model modules.

    Returns:
    - dict: Dictionary containing model names as keys and model classes as values.
    """
    models = {}
    for filename in os.listdir(directory):
        if filename.endswith('.py') and not filename.startswith('__'):
            module_name = filename[:-3]  # Remove .py extension
            module = importlib.import_module(f"{directory}.{module_name}")
            for attr_name in dir(module):
                attr_value = getattr(module, attr_name)
                if isinstance(attr_value, type):  # Check if it's a class
                    models[attr_name] = attr_value
    return models