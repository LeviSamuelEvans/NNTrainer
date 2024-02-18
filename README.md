# NN Trainer

This repository contains a small framework for training NN models in tasks such as classifcation and reconstruction.

Work in progress.

## Installation
Please clone this repository using

```
ssh://git@gitlab.cern.ch:7999/leevans/tth-network.git
```
Then, proceed to setup the correct enviroment needed to run the conversion and train the models by doing the following:
```
pip install virtualenv # if not already installed
virtualenv MyVirtualEnviroment
source MyVirtualEnviroment/bin/activate
pip install -r requirements.txt
```
Feel free to use conda for this, or any prefered method, e.g `python3 -m venv MLenv` (python's built-in virtual env)

To deactivate the enviroment simply do
```
deactivate
```
Note, if you are working on linappserv, please use the `linapp_requirements.txt` file. 
You can find an `NVIDIA GeForce GTX 970` available on `gpuserv1` for testing purposes.

For further development and enterprise level GPU systems, the UChicago facility offers resources for all ATLAS members. More information can be found [here](https://maniaclab.uchicago.edu/af-docs/)


## Conversion of Inputs
A script, `convert.py`, is available for converting ROOT ntuples into a more suitable format for ML tasks, e.g a pandas dataframe, which is then stored in the h5 file format. The feature variables to be extracted can be specified in a `.yaml` file.


## Input Inspection
A utility script for checking the contents of the created h5 files can be found in `Datasets/utils/` directory. It is called `h5reader.py`.

## Pre-configured networks
A set of pre-configured networks can be found in `models/networks`, where they are further described in the `catalogue.md` file.

## Basic Training
The data loading, preparation and training are handled by dedicated classes. It is configured via a `YAML` configuration file. Briefly, this contains the following:

- Model block - the model to load and use.
- Data block - the paths to signal and background samples.
- Network type - specifies type of data preparation method used.
- Preparation block - specifies dataset splits, data augmentation and feature makers (P)
- Features block - the feature variables to use in the training.
- Training block - specify hyperparameters
- Evaluation block

## Feature Validation
During the pre-processing stage before training, validation plots will be created of all feature inputs for the signal and background models. Alongside this, correlation matrices are also created. These will be found in `plots/Inputs`

## Model Validation
The trained model performance is validated with a validation set. Plots showing the accuracy and loss function are shown with respect to the training and validation performance, as a function of the epochs. These will found in `plots/Validation`

## Model Evaluation

The trained model is evaluated by examing the ROC curve and confusion matrix, which will be found in the `plots/Evaluation` after the completion of the training.

----

## Future plans

- Improved configuration files and less hard-coding
- Batch depolyment scripts
- ONNX conversion and Ntuple injection
- Website for full documentation
- Finish Feature Factory
- Validation Techniques (SHAP, LIME, Att.Maps, Data Adversarial Training)
- Extend network block for full user configuration beyond standard supplied networks
- Add scheduler options
- Improve .root conversion
- add CI pipeline

## On-going Projects

- GNNs for ttH(bb) Classification and Higgs reconstruction
- Optimised CP observables
- Decision metrics


## Useful Links

- Optuna https://optuna.org 
- Hyperopt http://hyperopt.github.io/hyperopt/
- SHAP https://shap.readthedocs.io/en/latest/
- Att. Maps https://www.mdpi.com/2076-3417/12/8/3846
- Layer-wise relevance propagation https://arxiv.org/abs/1803.08782
- [ATLAS GPU Training](https://indico.cern.ch/event/1331139/overview)
