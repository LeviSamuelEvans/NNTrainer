import argparse
import yaml
from collections import namedtuple


def handleCommandLineArgs():
    # Create the ArgumentParser
    parser = argparse.ArgumentParser(description="Nerual Network for ttH/tt Classification", add_help=True)

    # Define the custom usage message
    parser.usage = "./main.py --config /path/to/config.yaml"

    # Add an argument for the YAML configuration file
    parser.add_argument("-c","--config", help="Path to the YAML configuration file.", required=True)

    # Parse the arguments
    args = parser.parse_args()

    # Load the YAML configuration
    with open(args.config, 'r') as file:
        config_dict = yaml.safe_load(file)

    # Convert the dictionary to a named tuple
    Config = namedtuple('Config', config_dict.keys())
    config = Config(**config_dict)

    return config, args.config


# to be included at a later date
# # Future uses
# parser = argparse.ArgumentParser(description="Neural Network for ttH/tt Classification")

# # Data paths
# parser.add_argument('--signal_path', type=str, default="signal.h5", help='Path to the signal dataset')
# parser.add_argument('--background_path', type=str, default="ttbarBackground.h5", help='Path to the background dataset')

# # Model Configuration
# parser.add_argument('--model_type', type=str, default="SimpleNN", choices=["SimpleNN", "ModifiedNN"], help='Type of model to use')
# parser.add_argument('--hidden_units', type=str, default="128,64,32", help='Number of hidden units in each layer')
# parser.add_argument('--dropout_prob', type=float, default=0.5, help='Dropout probability for models that use dropout')

# # Optimizer Configuration
# parser.add_argument('--optimizer', type=str, default="Adam", choices=["Adam", "SGD"], help='Optimizer to use')
# parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
# parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 penalty)')

# # Training Configuration
# parser.add_argument('--batch_size', type=int, default=250, help='Number of samples per batch')
# parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
# parser.add_argument('--lr_schedule', type=str, default=None, choices=[None, "step", "plateau"], help='Learning rate scheduling method')
# parser.add_argument('--lr_decay', type=float, default=0.9, help='Decay factor for learning rate')
# parser.add_argument('--early_stopping', action='store_true', help='Use early stopping')
# parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')

# # Data Configuration
# parser.add_argument('--validation_split', type=float, default=0.2, help='Fraction of data to use for validation')
# parser.add_argument('--shuffle', action='store_true', help='Shuffle the dataset before splitting')
# parser.add_argument('--random_seed', type=int, default=42, help='Seed for random number generators')

# # Evaluation and Saving
# parser.add_argument('--save_model', action='store_true', help='Save the trained model')
# parser.add_argument('--model_save_path', type=str, default="trained_model.pth", help='Path to save the trained model')
# parser.add_argument('--save_frequency', type=int, default=1, help='How often to save the model (every N epochs)')
# parser.add_argument('--log_interval', type=int, default=50, help='How often to print training logs (every N batches)')

# # Environment Configuration
# parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device to use for training')
# parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')

# # Regularization and Augmentation
# parser.add_argument('--augmentation', action='store_true', help='Use data augmentation')
# parser.add_argument('--augment_prob', type=float, default=0.5, help='Probability of applying augmentation to an input sample')
# parser.add_argument('--l1_reg', type=float, default=0.0, help='L1 regularization strength')

# # Miscellaneous
# parser.add_argument('--verbose', type=int, default=1, help='Control the verbosity of the training logs (0: silent, 1: progress bar, 2: one line per epoch)')

# args = parser.parse_args()