#!/usr/bin/env python3

"""
============================
== ROOT to HDF5 Converter ==
============================

This python module provides functionality to convert datasets from ROOT files into
Pandas dataframes, which are then saved using the HDF5 storage format. This is
particularly useful for machine learning applications.

How to use:
python3 convert.py -d <directory> -s <storeName.h5> -v <variables.yaml>
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
           - handle jagged root arrays

TODO:
        - Add support for multiple dataframes in the store.
        - Add support for reading multiple directories.
        - Add support for reading multiple YAML files (perhaps for different samples).
        - Add support for reading variables from different branches.

"""

# Required modules:
import pandas as pd
import uproot           # For reading ROOT files using Python and Numpy
import os               # For OS-related operations like directory handling
import sys              # For accessing Python interpreter attributes and functions
import glob             # For searching specific file patterns using wildcards
import argparse         # For parsing command-line arguments
import yaml             # For processing yaml config files
from tqdm import tqdm   # For displaying progress bars
import awkward as ak    # For manipulating jagged arrays
import numpy as np      # For numpy arrays


class DataImporter:
    """
    A class that imports data from ROOT files and stores it in an HDF5 file.

    Attributes:
        storeName (str): The name of the data store.
        directory (str): The directory where the data is stored.
        variables (list): A list of variables to be imported.
        overwriteIsEnabled (bool): Whether or not to overwrite existing data.
        store (pd.HDFStore): The HDF5 store where the data is stored.
        DATA_STRUCTURE (dict): The data structure to be used for importing the data.
    """

    def __init__(self, storeName, directory, variables, overwriteIsEnabled):
        """
        Initializes a DataImporter object with the specified parameters.

        Args:
            storeName (str): The name of the data store.
            directory (str): The directory where the data is stored.
            variables (list): A list of variables to be imported.
            overwriteIsEnabled (bool): Whether or not to overwrite existing data.
            generate_data_structure (bool): Whether or not to generate a data structure.
            DATA_STRUCTURE (dict): The data structure to be used for importing the data.
        """
        self.storeName = storeName
        self.directory = directory
        self.variables = variables
        self.overwriteIsEnabled = overwriteIsEnabled
        self.store = None
        self.DATA_STRUCTURE = DataImporter.generate_data_structure(self.variables)

    def __enter__(self):
        """
        Prepare the HDF5 store for writing.

        Returns:
            DataImporter: The DataImporter object.
        """
        if self.overwriteIsEnabled and os.path.exists(self.storeName):
            os.remove(self.storeName)
            print(f"Overwriting enabled. Removed existing file: {self.storeName}")
        self.store = pd.HDFStore(self.storeName)
        return self

    def __exit__(self, type, value, tb):
        """
        Ensure the HDF5 store is closed properly.

        Args:
            type: The type of the exception.
            value: The exception object.
            tb: The traceback object.
        """
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

    @classmethod
    def generate_data_structure(cls, variables, max_length=20):
        """
        Generate a data structure for importing the data.

        Args:
            variables (list): A list of variables to be imported.
            max_length (int): The maximum length of the jagged arrays.

        Returns:
            dict: The data structure to be used for importing the data.
        """
        data_structure = {}
        for var in variables:
            for i in range(1, max_length + 1):
                data_structure[f"{var}_{i}"] = []
        return data_structure


    def getDataFrameFromRootfile(self, filepath):
        """
        Convert a ROOT file to a DataFrame and store it in the HDF5 store.

        Args:
            filepath (str): The path to the ROOT file.
        """
        filename = os.path.basename(filepath)
        storeKey = f"IndividualFiles/{filename.replace('.', '_')}"
        if storeKey not in self.store.keys():
            print(f"Opening ROOT file {filepath}")
            print(f"Trying to extract the following variables: {self.variables}")
            tree = uproot.open(filepath)["nominal_Loose"]
            all_branches = tree.keys()
            #print(f"All branches in the tree: {all_branches}") # DEBUG

            # Check if the variables are present in the specifief tree
            for var in self.variables:
                if var in all_branches:
                    print(f"Variable {var} is present in the tree.")
                else:
                    print(f"Variable {var} is NOT present in the tree.")

            print(f"Converting tree from {filename} to DataFrame...")
            events = tree.arrays(self.variables, library="ak")

            # Flatten and pad the jagged arrays
            flattened_data = self.DATA_STRUCTURE.copy()
            for key in flattened_data.keys():
                if key in events.fields:
                    flattened_data[key] = ak.pad_none(events[key], 20, clip=True).to_list()
                else:
                    flattened_data[key] = [[None, None, None] for _ in range(len(events))]

            # Convert the data to a DataFrame
            df = pd.DataFrame(flattened_data)

            # Convert the None values to NaN
            df.replace({None: np.nan}, inplace=True)
            for i in range(1, 21):
                column_name = f"jet_pt_{i}"
                if column_name in df.columns:
                    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')


            print(f"Saving DataFrame to HDF5 store with key: {storeKey}...")
            df.to_hdf(self.store, key=storeKey)
            print(f"Appending DataFrame to 'df' in the store...")
            self.store.append("df", df)
            print(f"Finished processing {filename}.")
        else:
            print(f"A file named {filename} already exists in the store. Ignored.")


    def processAllFiles(self):
        """
        Process all ROOT files in the specified directory and store them in the HDF5 store.
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