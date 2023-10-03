# ttH NN Trainer

This repository contains the a small framework for training several NN models in tasks such as classifcation and reconstruction. It should serve as a basis for converting the initial ROOT ntuples into a more suitable format for ML tasks, e.g a pandas dataframe, which is then stored in the H5 file format. The training is steered via a YAML configuration file, in which multiple options exist for selecting models, model hyperparameters, training features etc.

This is still heavily a work in progress, with many more features to come in due time.

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

To deactivate the enviroment simply do
```
deactivate
```

## Conversion of Inputs

### Coversion script

### h5 Inspection


## Feature Plotting and Basic Training

## Model Validation

## Model Evaluation

## Graph Networks

## Lorentz Equivariant Networks

## Reconstruction Tasks

