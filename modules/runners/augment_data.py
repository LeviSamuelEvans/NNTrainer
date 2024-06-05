import logging
from modules import Augmenter

def augment_data(config, use_four_vectors, features):
    """Augment the data using the Augmenter."""
    try:
        signal_fvectors, background_fvectors, *_ = features
        if config["preparation"].get("augment_data", False) and use_four_vectors:
            logging.info("Augmenter :: Proceeding to augment the input data...")
            augmenter = Augmenter.standard_augmentation(use_four_vectors)
            augmentation_methods = config["preparation"].get("augmentation_methods", {})

            if not augmentation_methods:
                augmenter.perform_all_augmentations(signal_fvectors, background_fvectors)
            else:
                if augmentation_methods.get("phi-rotation", False):
                    augmenter.rotate_phi(signal_fvectors, background_fvectors)
                if augmentation_methods.get("eta-reflection", False):
                    augmenter.reflect_eta(signal_fvectors, background_fvectors)
                if augmentation_methods.get("translate_eta_phi", False):
                    augmenter.translate_eta_phi(signal_fvectors, background_fvectors)
                if augmentation_methods.get("energy-variation", False):
                    augmenter.scale_energy_momentum(signal_fvectors, background_fvectors)

            logging.info("Augmenter :: Data augmentation complete.")
        else:
            logging.info("Augmenter :: Skipping data augmentation.")
    except Exception as e:
        logging.error(f"Failed to augment the input data: {e}")
