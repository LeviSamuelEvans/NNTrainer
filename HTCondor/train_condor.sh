#!/bin/bash
# https://batchdocs.web.cern.ch/gpu/index.html
# Request_GPUs = 1
# request_memory = 4 GB

# path to the environment we need
CONDA_INSTALL_DIR="$HOME/miniconda3"
export PATH="$CONDA_INSTALL_DIR/bin:$PATH"

# source the conda enviroment
source "$CONDA_INSTALL_DIR/etc/profile.d/conda.sh"

# now, activate the environment
conda activate MLenv

# Debug: Check if PyTorch is installed
python -c "import torch; print(torch.__version__)"

# Debug: Check the path of condor_gpu_discovery
which condor_gpu_discovery

# we need to set CUDA_VISIBLE_DEVICES to use the GPUs allocated by HTCondor
# ensure condor_gpu_discovery path is correct before using it
export CUDA_VISIBLE_DEVICES=$(condor_gpu_discovery)

# go to the framework directory (assume no transfer is needed...)
cd /home/levans/NNTrainer/tth-network

# now, let's train the model!
$HOME/miniconda3/envs/MLenv/bin/python main.py -c /home/levans/NNTrainer/tth-network/configs/dev/transformer5.yaml

echo "Training completed"
