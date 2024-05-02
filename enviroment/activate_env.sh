#!/bin/bash

# directory to install Miniconda
CONDA_INSTALL_DIR="$HOME/miniconda3"

# does the Conda installation exist?
if [ -d "$CONDA_INSTALL_DIR" ]; then
  echo "Miniconda found at $CONDA_INSTALL_DIR"
else
  echo "Miniconda is not installed at $CONDA_INSTALL_DIR. Please run create_miniconda.sh first."
  exit 1
fi

# init conda
source "$CONDA_INSTALL_DIR/etc/profile.d/conda.sh"
conda init

# name of enviroment
ENV_NAME="MLenv"

# does the enviroment exist?
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
