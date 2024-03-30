import os
import importlib
import inspect
import logging
import re

# https://www.tutorialspoint.com/python/os_walk.htm


class NetworkImporter:
    def __init__(self, directory):
        self.directory = directory
        self.networks = {}

    def load_networks_from_directory(self):
        for root, dirs, files in os.walk(self.directory):
            for filename in files:
                if filename.endswith(".py") and not filename.startswith("__"):
                    module_name = filename[:-3]  # Remove .py extension
                    module_path = root.replace("/", ".")  # Replace / with . in directory path
                    module = importlib.import_module(f"{module_path}.{module_name}")
                    for attr_name in dir(module):
                        attr_value = getattr(module, attr_name)
                        if (
                            inspect.isclass(attr_value)
                            and attr_value.__module__ == module.__name__
                        ):
                            self.networks[attr_name] = attr_value
        return self.networks

    def create_model(self, config):
        self.load_networks_from_directory()
        all_networks = self.networks
        input_dim = 4 if config.get("preparation", {}).get("use_four_vectors", False) else len(config["features"])
        model_name = config["model"]["name"]
        if re.search(r"transformer", config["model"]["name"], re.IGNORECASE):
            d_model = config["model"]["d_model"]
            nhead = config["model"]["nhead"]
            num_layers = config["model"]["num_encoder_layers"]
            dropout = config["model"]["dropout"] # should maybe be in training params

        if model_name in all_networks:
            model_class = all_networks[model_name]
            if model_name == "LorentzInteractionNetwork":
                #print(
                #    f"About to instantiate class {model_class} defined in {model_class.__module__}"
                #)
                model = model_class()
            elif re.search(r"transformer", config["model"]["name"], re.IGNORECASE):
                #print(
                #    f"About to instantiate class {model_class} defined in {model_class.__module__}"
                #)
                model = model_class(input_dim, d_model, nhead, num_layers, dropout)
            else:
                model = model_class(input_dim)
            return model
        else:
            logging.error(
                f"Model '{model_name}' not found. Available models are: {list(all_networks.keys())}"
            )
            return None

    def __repr__(self):
        """Return a string representation of the NetworkImporter object."""
        return f"NetworkImporter(directory={self.directory})"
