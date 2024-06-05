import logging
from modules import DataPlotter


def plot_inputs(config_dict, plotter):
    """Plot input features and correlation matrices."""
    if config_dict["data"]["plot_inputs"]:
        plotter.plot_all_features()
        plotter.plot_correlation_matrix("background")
        plotter.plot_correlation_matrix("signal")
    else:
        logging.info("DataPlotter :: Skipping plotting of inputs")
