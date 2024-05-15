#!/bin/bash
# https://batchdocs.web.cern.ch/gpu/index.html
# Request_GPUs = 1
# request_memory = 4 GB

# path to the environment we need
CONDA_INSTALL_DIR="$HOME/miniconda3"
export PATH="$CONDA_INSTALL_DIR/bin:$PATH"

# source the conda enviroment
source "$CONDA_INSTALL_DIR/etc/profile.d/conda.sh"

echo "Setting up the correct environment..."
# now, activate the environment
conda activate MLenv
echo "Environment activated! :D"

# go to the framework directory (assume no transfer is needed...)
cd /afs/cern.ch/user/l/leevans/NNTrainer/tth-network/

echo "Checking current directory and its contents:"
pwd
ls -l

# Debug: Check if PyTorch is installed
$HOME/miniconda3/envs/MLenv/bin/python -c "import torch; print(torch.__version__)"


# now, let's train the model!
$HOME/miniconda3/envs/MLenv/bin/python main.py -c /afs/cern.ch/user/l/leevans/NNTrainer/tth-network/configs/dev/config_GATv2.yaml

echo "Training completed"
