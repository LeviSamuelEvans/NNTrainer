import os
import importlib
import inspect

# https://www.tutorialspoint.com/python/os_walk.htm


def load_networks_from_directory(directory):
    """
    Dynamically load all networks from a directory and its subdirectories,
    making use of os.walk

    Parameters:
    - directory (str): Name of the directory containing model modules.

    Returns:
    - dict: Dictionary containing model names as keys and model classes as values.
    """
    networks = {}

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = filename[:-3]  # Remove .py extension
                module_path = root.replace(
                    "/", "."
                )  # Replace / with . in directory path
                module = importlib.import_module(f"{module_path}.{module_name}")

                for attr_name in dir(module):
                    attr_value = getattr(module, attr_name)
                    if (
                        inspect.isclass(attr_value)
                        and attr_value.__module__ == module.__name__
                    ):
                        networks[attr_name] = attr_value

    return networks
