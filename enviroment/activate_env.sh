#!/bin/bash

# check if a path was provided
if [ $# -eq 0 ]; then
  echo "No path provided. Using the default home directory ($HOME/miniconda3)."
  echo "To specify a different path, run the script as: ./activate_env <path/to/miniconda3>"
  echo
fi

CONDA_INSTALL_DIR="${1:-$HOME/miniconda3}"
ENV_NAME="${2:-MLenv}"

# is conda even installed?
if [ -d "$CONDA_INSTALL_DIR" ]; then
  echo "Miniconda found at $CONDA_INSTALL_DIR"
else
  echo "Miniconda is not installed at $CONDA_INSTALL_DIR. Please run create_miniconda.sh first, or point to the correct installation."
  exit 1
fi

# initialise conda
source "$CONDA_INSTALL_DIR/etc/profile.d/conda.sh"
conda init

# does the enviroment exist?
if conda env list | grep -qw "$ENV_NAME"; then
  echo "Environment '$ENV_NAME' exists."
else
  echo "Environment '$ENV_NAME' does not exist, please create it first!"
  exit 1
fi

echo "Activating environment '$ENV_NAME'..."
conda activate "$ENV_NAME"
echo "Setup complete. Virtual environment '$ENV_NAME' is ready to use! Happy ML'ing! ;D"