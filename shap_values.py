import torch
import shap
import numpy as np
import csv
import yaml
import logging
import matplotlib.pyplot as plt

from utils.logging import configure_logging
from utils import LoadingFactory, PreparationFactory
from models.importer import load_networks_from_directory


"""
Hard coded inital look at SHAP values
Will neeed to implement this into the framework
Have different run modes with -- commdand line args
- load,prepare,augment -l, -p, -a, -t, -e, -s, -m and extras
- train
- evaluate
- shap
- maps etc
"""

configure_logging()

# Load configuration file
config_path = '/scratch4/levans/tth-network/configs/config.yaml' 
with open(config_path, 'r') as file:
    config_dict = yaml.safe_load(file)

logging.info("Starting SHAP explanation...")

# Load the model
model_name = config_dict["model"]["name"]
all_networks = load_networks_from_directory("models/networks")
model_class = all_networks.get(model_name)
if not model_class:
    raise ValueError(f"Model '{model_name}' not found. Available models are: {list(all_networks.keys())}")
input_dim = len(config_dict["features"])
model = model_class(input_dim)
model_path = '/scratch4/levans/tth-network/models/outputs/model_19_02_24_nn4.pt'
logging.info("Loading the model from: {}".format(model_path))
model = torch.load(model_path, map_location='cpu')
model.eval()  # Set the model to evaluation mode
logging.info("Model loaded successfully and set to evaluation mode.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load and prepare data using LoadingFactory and PreparationFactory
network_type = config_dict["Network_type"][0]
loaded_data = LoadingFactory.load_data(network_type, config_dict)
train_loader, val_loader = PreparationFactory.prep_data(network_type, loaded_data, config_dict)

# Use a subset of the training data as background for SHAP
background, _ = next(iter(train_loader))
background = background.to(device)
logging.info("Background set prepared for SHAP.")

# initialise the Deep SHAP explainer
# explainer = shap.DeepExplainer(model, background)
# logging.info("SHAP DeepExplainer initialised.")

explainer = shap.GradientExplainer(model, background)
logging.info("SHAP Gradient initialised.")

num_batches = 5
batch_count = 0

for test_examples, _ in val_loader:
    test_examples = test_examples.to(device)
    batch_count += 1
    if batch_count >= num_batches:
        break

logging.info("Test examples prepared for explanation.")

# Calculate SHAP values
shap_values = explainer.shap_values(test_examples)
logging.info("SHAP values calculated.")

explanation = explainer.shap_values(test_examples)

# Save SHAP values
shap_values_path = '/scratch4/levans/tth-network/models/outputs/shap_values.csv'
np.savetxt(shap_values_path, shap_values, delimiter=',')
logging.info("SHAP values saved to CSV file.")


test_examples_np = test_examples.cpu().numpy()
save_path = '/scratch4/levans/tth-network/models/outputs/shap_force_plot.png'
feature_names = config_dict['features']
logging.info("Generating SHAP summary plot...")
shap.summary_plot(shap_values, test_examples_np, feature_names=feature_names,show=False)
plt.savefig(save_path)
plt.close()
logging.info("SHAP summary plot saved.")
# Generating and saving scatter plots for each feature

# Plotting using SHAP's scatter plot, coloring by the overall SHAP values
shap.plots.scatter(shap_values, color=shap_values)
plt.savefig('/scratch4/levans/tth-network/models/outputs/shap_scatter_plot.png')



# shap.initjs()
# shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], test_examples.cpu().numpy()[0,:])

logging.info("SHAP explanation complete.")