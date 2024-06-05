from modules import FeatureFactory


def extract_features(config_dict, network_type, signal_data=None, background_data=None):
    """Extract features using the FeatureFactory."""
    if network_type in ["FFNN", "TransformerGCN"]:
        return FeatureFactory.extract_features(
            config_dict, signal_data, background_data
        )
    return None
