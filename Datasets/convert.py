#!/usr/bin/env python3

"""
============================
== ROOT to HDF5 Converter ==
============================

Description:
------------
    This python module provides functionality to convert datasets from ROOT files into
    Pandas dataframes, which are then saved using the HDF5 storage format.

Usage:
------
    Run python3 convert.py -h to see the help message.

"""

# Required modules:
import pandas as pd
import uproot
import os
import sys
import glob
import argparse
import yaml
from tqdm import tqdm
from utils.dataTypes import DATA_TYPES
import numpy as np


class DataImporter(object):
    """
    Class to handle the import of data from ROOT files and save them as Pandas dataframes in HDF5 format.
    """

    def __init__(self, storeName, directory, variables, overwriteIsEnabled):
        """Constructor for the DataImporter class.

        Parameters:
        -----------
        storeName : str
            Path for the HDF5 store to be created.
        directory : str
            Directory containing the ROOT files.
        variables : list
            Variables to be read from the ROOT tree.
        overwriteIsEnabled : bool
            Flag to overwrite existing store.
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
        """Retrieve all the filepaths of ROOT files in the specified directory.

        Returns:
        --------
        list
            Sorted list of ROOT file paths.
        """
        absDirectory = os.path.realpath(os.path.expanduser(self.directory))
        listOfFiles = sorted(glob.glob(absDirectory + "/*.root"))
        return listOfFiles

    # ==============================================================================

    @staticmethod
    def flatten_jagged_array(array, var_name, fixed_length, pad_value=0):
        """Flatten a jagged array to a fixed length and return a DataFrame."""

        # Ensure the array is truncated or padded to the fixed length
        truncated_array = [
            item[:fixed_length] if len(item) > fixed_length else item for item in array
        ]
        flattened_array = [
            np.pad(
                item,
                (0, fixed_length - len(item)),
                mode="constant",
                constant_values=pad_value,
            ).astype(np.float32)
            for item in truncated_array
        ]
        column_names = [f"{var_name}_{i+1}" for i in range(fixed_length)]
        return pd.DataFrame(flattened_array, columns=column_names)

    def flatten_and_concat(self, df, column_name, fixed_length):
        """Flatten a jagged array column and concatenate it with the original DataFrame."""
        flattened_df = self.flatten_jagged_array(
            df[column_name].tolist(), column_name, fixed_length
        )
        return pd.concat([df.drop(columns=[column_name]), flattened_df], axis=1)

    # ==============================================================================

    def getDataFrameFromRootfile(
        self, filepath, fixed_jet_length, max_events=None, max_jets=12, order_jets=True
    ):
        """Convert a ROOT file into a Pandas dataframe and save it to the HDF5 store.

        The final h5 file will have two keys:
            'df': A single dataframe with all the events.
            'IndividualFiles/<filename>': A dataframe for each ROOT file.

        Parameters:
        -----------
        filepath : str
            Path to the ROOT file.
        fixed_jet_length : int
            The fixed length to which jagged arrays will be truncated or padded.
        max_events : int
            Maximum number of events to process per file.
        max_jets : int
            Maximum number of jets to process per event.
        """
        filename = os.path.basename(filepath)
        storeKey = f"IndividualFiles/{filename.replace('.', '_')}"
        if storeKey not in self.store.keys():
            print(f"INFO: Opening ROOT file {filepath}")
            print(f"INFO: Trying to extract the following variables: {self.variables}")
            tree = uproot.open(filepath)["nominal_Loose"]
            all_branches = tree.keys()

            # Check if your variables are in the tree
            for var in self.variables:
                if var in all_branches:
                    print(f"INFO: Variable {var} is present in the tree.")
                else:
                    print(
                        f"INFO: Variable {var} is NOT present in the tree. You might to check this..."
                    )
            print(f"INFO: Converting tree from {filename} to DataFrame...")
            print(f"INFO: Max events to process: {max_events}")
            print(f"INFO: Max jets in event to process: {max_jets}")

            # Create a dictionary of arrays from the ROOT tree
            df_dict = {}

            for var in self.variables:
                array = tree[var].array(library="np")
                if max_events is not None:
                    array = array[:max_events]
                df_dict[var] = array

            # Convert the dictionary to a DataFrame
            df = pd.DataFrame(df_dict)

            if order_jets:
                print("INFO: Ordering jets in each event by b-tagging score and pT...")
                self.order_jets(df, max_jets)
            else:
                print("INFO: Jets will not be ordered in each event!")

            # Log basic DataFrame information  DEBUGGING
            print(f"\nDataFrame Info for file {filename}:")
            print(f"Shape: {df.shape}")
            print("Columns and Data Types:")
            print(df.dtypes)

            # Handle jagged-array columns
            jagged_columns = [
                "jet_pt",
                "jet_e",
                "jet_eta",
                "jet_phi",
                "jet_tagWeightBin_DL1r_Continuous",
                "jet_eta_softmu_corr",
                "jet_pt_softmu_corr",
                "jet_phi_softmu_corr",
                "jet_e_softmu_corr",
                "jet_index_PCBT_ordered",
            ]  # Add anymore jagged array-type vars TODO:

            for column in jagged_columns:
                if column in df.columns:
                    print(f"Processing {column}...")
                    df = self.flatten_and_concat(df, column, fixed_jet_length)

            # Additional logging after handling jagged arrays DEBUG
            print(
                f"\nDataFrame Info after processing jagged arrays for file {filename}:"
            )
            print(f"Shape: {df.shape}")
            print("Columns and Data Types:")
            print(df.dtypes)

            # Log a sample of the DataFrame after processing jagged arrays DEBUGGING
            print("\nDataFrame Sample after processing jagged arrays:")
            print(df.head())  # Prints the first 5 rows of the DataFrame

            # convert columns to the correct data types
            for col, dtype in DATA_TYPES:
                if col in df.columns:
                    df[col] = df[col].astype(dtype, errors="ignore")

            # Cconversion of the object columns for lepton data
            object_columns = [
                "mu_pt",
                "mu_eta",
                "mu_phi",
                "mu_e",
                "el_pt",
                "el_eta",
                "el_phi",
                "el_e",
            ]
            for col in object_columns:
                if col in df.columns:
                    # get length of the longest list in the column
                    max_length = df[col].apply(len).max()

                    # create new columns for each element
                    for i in range(max_length):
                        new_col = f"{col}_{i}"
                        df[new_col] = df[col].apply(
                            lambda x: x[i] if len(x) > i else 0.0
                        )

                    # drop original object
                    df.drop(columns=[col], inplace=True)

            # Log final DataFrame structure before appending to HDF5 DEBUGGING
            print(
                f"INFO: \nFinal DataFrame structure before appending to HDF5 for file {filename}:"
            )

            print(f"Shape: {df.shape}")
            print("Columns and Data Types:")
            print(df.dtypes)

            # Log a final sample of the DataFrame DEBUGGING
            print("\nFinal DataFrame Sample before appending to HDF5:")
            print(df.head())

            print(f"Saving DataFrame to HDF5 store with key: {storeKey}...")
            df.to_hdf(
                self.store,
                key="IndividualFiles/%s" % filepath.split("/")[-1].replace(".", "_"),
            )
            print(f"INFO: Appending DataFrame to 'df' in the store...")
            self.store.append("df", df, min_itemsize=16)
            print(f"INFO: Finished processing {filename}.")
        else:
            print(
                f"INFO: A file named {filename} already exists in the store. Ignored."
            )

    # ==============================================================================

    def processAllFiles(self, max_events=None, max_jets=12, order_jets=True):
        """Process all ROOT files in the specified directory and display a progress bar."""
        fixed_jet_length = max_jets
        filepaths = self.getRootFilepaths()
        for filepath in tqdm(
            filepaths,
            desc="Processing files",
            unit="files",
            unit_scale=1,
            unit_divisor=60,
        ):
            self.getDataFrameFromRootfile(
                filepath, fixed_jet_length, max_events, max_jets
            )
    # ==============================================================================

    def order_jets(self, df, max_jets):
        """
        Order the jets in each event according to the b-tagging score and pT as tie-breaker.

        Parameters
        ----------
            df : pandas.DataFrame
                The DataFrame containing the jet data.
            max_jets : int
                The maximum number of jets to consider.

        Returns
        -------
            pandas.DataFrame:
                The DataFrame with the jets ordered based on the b-tagging
                score and pT.

        """
        jet_columns = [col for col in df.columns if col.startswith("jet_")]

        # create a structured array to store b-tagging scores and pT for each jet
        dtype = [("btag", float), ("pt", float)]
        jet_scores = np.empty((len(df), max_jets), dtype=dtype)

        jet_btag = df["jet_tagWeightBin_DL1r_Continuous"].tolist()
        jet_pt = df["jet_pt"].tolist()

        for i in range(max_jets):
            jet_scores[:, i]["btag"] = [
                btag[i] if len(btag) > i else 0.0 for btag in jet_btag
            ]
            jet_scores[:, i]["pt"] = [pt[i] if len(pt) > i else 0.0 for pt in jet_pt]

        # get indices that sort the structured array based on b-tagging score and pT
        sorted_indices = np.argsort(jet_scores, order=("btag", "pt"))[:, ::-1]

        # reorder jet columns based on the sorted indices
        for col in jet_columns:
            if col in ["jet_tagWeightBin_DL1r_Continuous", "jet_pt"]:
                jet_values = df[col].tolist()
                df[col] = [
                    [
                        jet_values[i][j] if len(jet_values[i]) > j else 0.0
                        for j in indices
                    ]
                    for i, indices in enumerate(sorted_indices)
                ]
            else:
                jet_values = df[col].tolist()
                df[col] = [
                    [
                        jet_values[i][j] if len(jet_values[i]) > j else 0.0
                        for j in indices
                    ]
                    for i, indices in enumerate(sorted_indices)
                ]

        return df
    # ==============================================================================

def handleCommandLineArgs():
    """Parse and handle command-line arguments.

    Returns:
    --------
    Namespace:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert ROOT files to HDF5 format.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-d", "--directory", help="Directory containing the ROOT files.", required=True
    )
    parser.add_argument(
        "-s",
        "--storeName",
        help="Path for the output HDF5 store (default: store.h5).",
        default="store.h5",
    )
    parser.add_argument(
        "-v",
        "--variables",
        help="YAML file containing the list of variables to read from ROOT files.",
        required=True,
    )
    parser.add_argument(
        "-O",
        "--overwrite",
        help="Overwrite existing store if it exists.",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--num-events",
        type=int,
        help="Maximum number of events to process per file (default: process all events).",
    )
    parser.add_argument(
        "--max-jets",
        type=int,
        default=12,
        help="Maximum number of jets to process per event (default: 12).",
    )
    parser.add_argument(
        "--order-jets",
        type=bool,
        default=True,
        help="Order jets in each event by b-tagging score and pT (default: True).\n"
        "This is useful when using --max-jets option to avoid losing important jets.",
    )
    args = parser.parse_args()

    # Load variables from the YAML file
    with open(args.variables, "r") as file:
        yaml_content = yaml.safe_load(file)
        if "features" in yaml_content:
            args.variables = yaml_content["features"]
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


# ==============================================================================
# MAIN METHOD TO EXECUTE THE DATA IMPORT PROCESS AND CONERT TO H5
def main():
    """Main function to execute the data import process."""
    args = handleCommandLineArgs()
    with DataImporter(
        args.storeName, args.directory, args.variables, args.overwrite
    ) as importer:
        importer.processAllFiles(
            max_events=args.num_events,
            max_jets=args.max_jets,
            order_jets=args.order_jets,
        )


if __name__ == "__main__":
    main()
