import logging
import time
from prettytable import PrettyTable
from utils.config_utils import log_with_separator

def _print_model_summary(model):
    """Logs a summary of the model in a table format.
    """
    table = PrettyTable()
    table.field_names = ["Module", "Input Features", "Output Features", "Parameters", "Trainable"]
    table.align = "c"

    total_params = 0
    total_trainable_params = 0

    for name, module in model.named_children():
        if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            table.add_row([name, module.in_features, module.out_features, params, trainable_params])
            total_params += params
            total_trainable_params += trainable_params
        else:
            # handle other types of layers, like Dropout or BatchNorm
            params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            table.add_row([name, '-', '-', params, trainable_params])
            total_params += params
            total_trainable_params += trainable_params

    table.add_row(['Total', '-', '-', total_params, total_trainable_params])
    logging.info("Model Summary")
    logging.info("==============")
    logging.info(f"\n{table}")


def _print_cosine_scheduler_params(cosine_scheduler):
    """Logs the parameters of the Cosine Scheduler."""
    exclude_keys = ['optimizer', 'base_lrs']
    logging.info("Initialising Cosine Scheduler...")

    params_table = PrettyTable()
    params_table.field_names = ["Parameter", "Value"]
    params_table.align = "l"

    for key, value in cosine_scheduler.__dict__.items():
        if key not in exclude_keys:
            if not isinstance(value, (dict, list, tuple)):
                params_table.add_row([key, value])
            else:
                formatted_value = ', '.join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)
                params_table.add_row([key, formatted_value])

    # log the table
    logging.info(f"Initialised Cosine Scheduler with params:\n{params_table}")
