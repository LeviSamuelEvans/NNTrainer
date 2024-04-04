import os
import importlib
import inspect
import logging
import re

# https://www.tutorialspoint.com/python/os_walk.htm


class NetworkImporter:
    "A class for importing networks."
    def __init__(self, directory):
        self.directory = directory
        self.networks = {}

    def load_networks_from_directory(self):
        """Load all networks from the specified directory.

        Returns
        -------

        dict
            A dictionary of all networks in the directory, with the class name as the
            key and the class as the value.

        """
        for root, dirs, files in os.walk(self.directory):
            for filename in files:
                if filename.endswith(".py") and not filename.startswith("__"):
                    # Remove .py extension
                    module_name = filename[:-3]
                    module_path = root.replace("/", ".")
                    # Replace / with . in directory path
                    module = importlib.import_module(f"{module_path}.{module_name}")
                    for attr_name in dir(module):
                        attr_value = getattr(module, attr_name)
                        if (
                            inspect.isclass(attr_value)
                            and attr_value.__module__ == module.__name__
                        ):
                            self.networks[attr_name] = attr_value
        return self.networks

    def calc_input_dim(self, config):
        """Calculate the input dimension based on the configuration.

        This helper function will calculate the input dimension based
        on the configuration to be passed to the model.

        Parameters
        ----------

        config : dict
            The configuration dictionary.

        Returns
        -------

        int
            The input dimension.

        """
        # TODO: add better logic here for extra_feats and graphs!
        # at the moment, we just assume that if we are using extra
        # features we also use more representations
        if config.get("preparation", {}).get("use_four_vectors", False) \
        and config.get("preparation", {}).get("use_representations", False):
            return 8
        elif config.get("preparation", {}).get("use_four_vectors", False):
            return 4
        else:
            return len(config["features"])

    def create_model(self, config):
        """ Create a model based on the configuration.

        Parameters
        ----------

        config : dict
            The configuration dictionary.
        """
        self.load_networks_from_directory()
        all_networks = self.networks
        input_dim = self.calc_input_dim(config)
        model_name = config["model"]["name"]
        if re.search(r"transformer", config["model"]["name"], re.IGNORECASE):
            d_model = config["model"]["d_model"]
            nhead = config["model"]["nhead"]
            num_layers = config["model"]["num_encoder_layers"]
            dropout = config["model"]["dropout"]

        if model_name in all_networks:
            model_class = all_networks[model_name]
            if model_name == "LorentzInteractionNetwork":
                model = model_class()
            elif re.search(r"transformer", config["model"]["name"], re.IGNORECASE):
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
