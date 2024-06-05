import sys
import logging
from modules.insights.att_maps import AttentionMap
from modules.insights.embed_maps import EmbeddingMaps


def handle_insights(run_mode, config_dict, network_type, val_loader, model):
    """Handle different insight modes."""
    if run_mode == "attention_map":
        attention_map = AttentionMap(network_type, config_dict)
        loaded_model = attention_map.load_model()
        attention_weights = attention_map.get_attention_weights(
            val_loader, loaded_model
        )
        feature_names = config_dict["features"]

        attention_map.plot_attention_weights(attention_weights, feature_names)
        attention_map.plot_attention_entropy(attention_weights, feature_names)
        sys.exit(0)

    elif run_mode == "embed_maps":
        logging.info("Proceeding to extract and visualise the positional embeddings...")
        embed_maps = EmbeddingMaps(model, network_type, config_dict)

        for batch in val_loader:
            print(f"Batch: {batch}")
            x, labels = batch
            break

        embed_maps.run(x, labels)
        sys.exit(0)
