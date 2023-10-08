#!/usr/bin/env python3

"""
============================
== ROOT to HDF5 Converter ==
============================

This python module provides functionality to convert datasets from ROOT files into
Pandas dataframes, which are then saved using the HDF5 storage format. This is
particularly useful for machine learning applications.

How to use:
python3 convert.py --directory <directory> --variables <variables.yaml>
Mounting:
sshfs -r levans@linappserv0.pp.rhul.ac.uk:/juicefs/data/levans/L2_ttHbb_Production_212238_v3/MLSamples/Datasets/1l/ ~/Desktop/PhD/MountPoint/
Dismounting:
sudo diskutil umount force /Users/levievans/Desktop/PhD/MountPoint/

path then equals -> /Users/levievans/Desktop/PhD/MountPoint/MLSamples/Datasets/1l/ttH for signal
or /Users/levievans/Desktop/PhD/MountPoint/MLSamples/Datasets/1l/ttbar for background
depends on the mount too -> /Users/levievans/Desktop/PhD/MountPoint/ttH/

Version 1.0:
           - Initial version.
           - Converts all ROOT files in a directory into a single HDF5 store.
           - The store contains a single dataframe with all the events.
           - Variables to be read from the ROOT tree are specified in a YAML file.
           - overwriteIsEnabled flag to overwrite existing store, or prevent overwriting.
           - handle jagged arrays

TODO:
        - Add support for multiple dataframes in the store.
        - Add support for reading multiple directories.
        - Add support for reading multiple YAML files (perhaps for different samples).
        - Add support for reading multiple variables from the YAML file.
        - Add support for reading variables from different branches.
        - Add support for reading variables from different trees.

"""

# Required modules:
import pandas as pd
import uproot                          # For reading ROOT files using Python and Numpy
import os                              # For OS-related operations like directory handling
import sys                             # For accessing Python interpreter attributes and functions
import glob                            # For searching specific file patterns using wildcards
import argparse                        # For parsing command-line arguments
import yaml                            # For processing yaml config files
from tqdm import tqdm                  # For displaying progress bars
from utils.dataTypes import DATA_TYPES # For converting data types
import numpy as np                     # For use in handling jagged arrays

class DataImporter(object):
    """
    Class to handle the import of data from ROOT files and save them as Pandas
    dataframes in HDF5 format.
    """

    def __init__(self, storeName, directory, variables, overwriteIsEnabled):
        """
        Constructor for the DataImporter class.

        Parameters:
        - storeName (str): Path for the HDF5 store to be created.
        - directory (str): Directory containing the ROOT files.
        - variables (list): Variables to be read from the ROOT tree.
        - overwriteIsEnabled (bool): Flag to overwrite existing store.
        """
        self.storeName = storeName
        self.directory = directory
        self.variables = variables
        self.overwriteIsEnabled = overwriteIsEnabled
        self.store = None

    def __enter__(self):
        """Prepare the HDF5 store for writing."""
        if self.overwriteIsEnabled and os.path.exists(self.storeName):
            os.remove(self.storeName)
            print(f"Overwriting enabled. Removed existing file: {self.storeName}")
        self.store = pd.HDFStore(self.storeName, min_itemsize=500)
        return self

    def __exit__(self, type, value, tb):
        """Ensure the HDF5 store is closed properly."""
        self.store.close()

    def getRootFilepaths(self):
        """
        Retrieve all the filepaths of ROOT files in the specified directory.

        Returns:
        - list: Sorted list of ROOT file paths.
        """
        absDirectory = os.path.realpath(os.path.expanduser(self.directory))
        listOfFiles = sorted(glob.glob(absDirectory + "/*.root"))
        return listOfFiles

    @staticmethod
    def flatten_jagged_array(array, var_name, max_length, pad_value=0):
        """Flatten a jagged array to a fixed length and return a DataFrame."""
        flattened_array = [np.pad(item, (0, max_length - len(item)), mode='constant', constant_values=pad_value) for item in array]
        column_names = [f"{var_name}_{i+1}" for i in range(max_length)]
        return pd.DataFrame(flattened_array, columns=column_names)

    def flatten_and_concat(self, df, column_name):
        """Flatten a jagged array column and concatenate it with the original DataFrame."""
        max_length = df[column_name].apply(len).max()
        flattened_df = self.flatten_jagged_array(df[column_name].tolist(), column_name, max_length)
        return pd.concat([df.drop(columns=[column_name]), flattened_df], axis=1)

    def getDataFrameFromRootfile(self, filepath):
        """
        Convert a ROOT file into a Pandas dataframe and save it to the HDF5 store.
        The final h5 file will have two keys:
        - 'df': A single dataframe with all the events.
        - 'IndividualFiles/<filename>': A dataframe for each ROOT file.

        Parameters:
        - filepath (str): Path to the ROOT file.
        """
        filename = os.path.basename(filepath)
        storeKey = f"IndividualFiles/{filename.replace('.', '_')}"
        if storeKey not in self.store.keys():
            print(f"Opening ROOT file {filepath}")
            print(f"Trying to extract the following variables: {self.variables}")
            tree = uproot.open(filepath)["nominal_Loose"]
            all_branches = tree.keys()
            #print(f"All branches in the tree: {all_branches}") #DEBUG

            #Check if your variables are in the tree
            for var in self.variables:
                if var in all_branches:
                    print(f"Variable {var} is present in the tree.")
                else:
                    print(f"Variable {var} is NOT present in the tree.")
            print(f"Converting tree from {filename} to DataFrame...")
            # Create a dictionary of arrays from the ROOT tree
            df_dict = {}
            for var in self.variables:
                df_dict[var] = tree[var].array(library="np")
            # Convert the dictionary to a DataFrame
            df = pd.DataFrame(df_dict)
            #print(type(df)) #DEBUG
            #print(df) #DEBUG

            # Handle jagged-array columns
            jagged_columns = ['jet_pt', 'jet_e', 'jet_eta', 'jet_phi', 'jet_tagWeightBin_DL1r_Continuous']  # Add anymore jagged array-type vars
            for column in jagged_columns:
                if column in df.columns:
                    print(f'Processing {column}...')
                    df = self.flatten_and_concat(df, column)

            #print(type(df)) # DEBUG
            #print(df) # DEBUG

            # Convert the data types of the columns
            for col, dtype in DATA_TYPES:
                if col in df.columns:
                    if dtype == 'float32':
                        df[col] = df[col].astype(str).astype(dtype, errors='ignore')
                        print()
                    else:
                        df[col] = df[col].astype(dtype, errors='ignore')

            print(f"Saving DataFrame to HDF5 store with key: {storeKey}...")
            df.to_hdf(self.store, key = "IndividualFiles/%s" % filepath.split("/")[-1].replace(".", "_"))
            print(f"Appending DataFrame to 'df' in the store...")
            self.store.append("df", df)
            print(f"Finished processing {filename}.")
        else:
            print(f"A file named {filename} already exists in the store. Ignored.")

    def processAllFiles(self):
        """
        Process all ROOT files in the specified directory and display a progress bar.
        """
        filepaths = self.getRootFilepaths()
        for filepath in tqdm(filepaths, desc="Processing files",unit="files",unit_scale=1, unit_divisor=60):
            self.getDataFrameFromRootfile(filepath)

def handleCommandLineArgs():
    """
    Parse and handle command-line arguments.

    Returns:
    - Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--directory", help="Directory with ROOT files.", required=True)
    parser.add_argument("-s","--storeName", help="Path for the HDF5 store.", default="store.h5")
    parser.add_argument("-v","--variables", help="YAML file containing variables to read from ROOT files.", required=True)
    parser.add_argument("-O","--overwrite", help="Overwrite existing store.", action="store_true")
    args = parser.parse_args()

    # Load variables from the YAML file
    with open(args.variables, 'r') as file:
        yaml_content = yaml.safe_load(file)
        if 'features' in yaml_content:
            args.variables = yaml_content['features']
            print("Variables to be used from the YAML file:", args.variables)
        else:
            print("The YAML file does not contain a 'features' key.")
            sys.exit(1)

    if args.overwrite:
        while True:
            confirmation = input("--overwrite specified. Proceed? (y/n): ").lower()
            if confirmation in ["y", "yes"]:
                break
            elif confirmation in ["n", "no"]:
                sys.exit(1)
            else:
                print("Invalid input. Answer 'yes' or 'no'.")

    return args

def main():
    """Main function to execute the data import process."""
    args = handleCommandLineArgs()
    with DataImporter(args.storeName, args.directory, args.variables, args.overwrite) as importer:
        importer.processAllFiles()

if __name__ == "__main__":
    main()