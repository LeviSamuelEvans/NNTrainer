import logging
from utils import TrainerArgs
from modules.train import Trainer


def train_model(config, model, train_loader, val_loader, run_mode):
    """Train the model."""
    trainer_args = TrainerArgs(
        config, model, train_loader, val_loader, config["Network_type"][0]
    )
    trainer = Trainer(**trainer_args.prepare_trainer_args())

    if run_mode != "evaluate":
        trainer.train_model()
        trainer.save_metrics_to_json(f"{trainer.model_save_path}/metadata.json")

    return trainer
