from modules import PreparationFactory


def prepare_data(
    network_type,
    loaded_data,
    config_dict,
    signal_fvectors,
    background_fvectors,
    signal_edges,
    signal_edge_attr,
    background_edges,
    background_edge_attr,
):
    """Prepare the data using the PreparationFactory."""
    return PreparationFactory.prep_data(
        network_type,
        loaded_data,
        config_dict,
        signal_fvectors,
        background_fvectors,
        signal_edges,
        signal_edge_attr,
        background_edges,
        background_edge_attr,
    )
