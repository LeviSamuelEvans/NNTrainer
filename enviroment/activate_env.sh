#!/bin/bash

# directory where Miniconda is installed, default is $HOME/miniconda3
CONDA_INSTALL_DIR="${1:-$HOME/miniconda3}"

# name of environment, default is MLenv
ENV_NAME="${2:-MLenv}"

# does the Conda installation exist?
if [ -d "$CONDA_INSTALL_DIR" ]; then
  echo "Miniconda found at $CONDA_INSTALL_DIR"
else
  echo "Miniconda is not installed at $CONDA_INSTALL_DIR. Please run create_miniconda.sh first, or point to correct installation."
  exit 1
fi

# init conda
source "$CONDA_INSTALL_DIR/etc/profile.d/conda.sh"
conda init

# does the environment exist?
if conda env list | grep -qw "$ENV_NAME"; then
  echo "Environment '$ENV_NAME' exists."
else
  echo "Environment '$ENV_NAME' does not exist, please create it first!"
  exit 1
fi

# activate
echo "Activating environment '$ENV_NAME'..."
conda activate "$ENV_NAME"

echo "Setup complete. Virtual environment '$ENV_NAME' is ready to use! Happy ML'ing! ;D"