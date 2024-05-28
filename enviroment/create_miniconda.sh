#!/bin/bash

# Usage: ./create_miniconda.sh [CONDA_INSTALL_DIR] [ENV_YAML] [ENV_NAME]

# directory to install Miniconda, default is $HOME/miniconda3
CONDA_INSTALL_DIR="${1:-$HOME/miniconda3}"

# path to our environment YAML file, default is $HOME/NNTrainer/tth-network/enviroment/enviroment.yml
ENV_YAML="${2:-$HOME/NNTrainer/tth-network/enviroment/enviroment.yml}"

# The name of our conda environment, default is MLenv
ENV_NAME="${3:-MLenv}"

# check first if Miniconda is already installed...
if [ ! -d "$CONDA_INSTALL_DIR" ]; then
  echo "Installing Miniconda..."
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p $CONDA_INSTALL_DIR
  rm -f miniconda.sh
else
  echo "Miniconda is already installed! :D"
fi

# init conda
source "$CONDA_INSTALL_DIR/etc/profile.d/conda.sh"
conda init

# Now, create or update the Conda environment
if [ -f "$ENV_YAML" ]; then
  echo "Creating or updating environment from $ENV_YAML..."
  conda env create -f "$ENV_YAML" || conda env update -f "$ENV_YAML"
else
  echo "Environment YAML file not found at $ENV_YAML"
fi

# activate
echo "Activating environment '$ENV_NAME'..."
conda activate $ENV_NAME

echo "Setup complete. Virtual environment '$ENV_NAME' is ready to use! Happy ML'ing! :D"

## == Some Useful Commands ==

# conda remove --name MLenv --all
# conda env list
# conda activate MLenv
# conda deactivate
# conda update conda

# SAVE TO THE YAML
# conda env export > environment.yml