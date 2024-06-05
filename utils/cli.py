import argparse
import yaml
from collections import namedtuple


def handleCommandLineArgs() -> namedtuple:
    """This function handles the command line arguments passed to the program."""

    parser = argparse.ArgumentParser(
        description="Nerual Network for ttH/tt Classification", add_help=True
    )
    parser.usage = "./main.py --config /path/to/config.yaml"

    parser.add_argument(
        "-c", "--config", help="Path to the YAML configuration file.", required=True
    )

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config_dict = yaml.safe_load(file)

    return config_dict, args.config
