- Version 0.4 dev


- Version 0.3
    - update scheduler and annealing functions options
    - added some attention models
    - add pre-processing factory for GNN and FNN models
    - added GPU support
    - added some GNN models
    - added pr curve in evaluation
    - added initial feature factory
        - four-vector inputs of objects
    - added NetworkImporter class
    - added a lorentz attention network

- Version 0.2
    - encapsulated plotting into a class
    - added plotting of all input features (visualise the data)
    - added plotting of inputs correlation matrix for signal and background
    - added learning rate scheduler
    - added colour logging
    - added a LENN model to the model dictionary

- Version 0.1
    - Initial version
    - Load and train a selection of models
    - steered via yaml configuration file
    - ability to easily select hyperparameters through config file
    - ability to easily select model through config file
    - ability to easily select input features through config file


FUTURE (notes):
    - add conversion to onnx
    - add logging verbosity
    - upgrade to multidimensional arrays
    - add network builder, configured via yaml
    - add data augmentation methods