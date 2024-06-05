import logging
from modules import LoadingFactory


def load_data(config_dict):
    """Load the data based on the network type."""
    network_type = config_dict["Network_type"][0]
    loaded_data = LoadingFactory.load_data(network_type, config_dict)

    if network_type in ["FFNN", "TransformerGCN"]:
        signal_data, background_data = loaded_data
    elif network_type in ["GNN", "LENN"]:
        (
            node_features_signal,
            edge_features_signal,
            global_features_signal,
            labels_signal,
            node_features_background,
            edge_features_background,
            global_features_background,
            labels_background,
        ) = loaded_data
    else:
        raise ValueError(f"Unsupported network type: {network_type}")

    return network_type, loaded_data
