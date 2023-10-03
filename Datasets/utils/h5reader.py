import pandas as pd
import numpy as np

"""
A utility script to help inspect the signal and background
datasets and display the first few rows of the DataFrames.

"""

# Set paths to the signal and background datasets
signal ='../signal2.h5'
background = "../ttbarBackground2.h5"

# Read the signal and background datasets into Pandas dataframes
df_sig = pd.read_hdf(signal, key="df")
df_bkg = pd.read_hdf(background, key="df")

# Filter the dataframes based on the features
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)


def display_dataframes():
    """
    Displays the first few rows of the signal and background DataFrames, and their summary statistics.

    Parameters:
    None

    Returns:
    None
    """
    # Display the first few rows of the DataFrame
    print(df_sig.head())

    # Get summary statistics
    print("========================================")
    print('Summary Statistics for the signal sample')
    print(df_sig.describe())
    print("========================================")
    print()
    print("========================================")
    print('Summary Statistics for the bkg sample')
    print("========================================")
    print(df_bkg.describe())
    # Check for missing values
    print(df_sig.isnull().sum())


def print_h5_variables(store_path):
    """
    Prints all the variables in an h5 store.

    Parameters:
    store_path (str): The path to the h5 store.

    Returns:
    None
    """
    with pd.HDFStore(store_path) as store:
        print(store.keys())
        print(store.info())

        print("Column names for signal dataframe:")
        print(df_sig.columns)

        print("Column names for background dataframe:")
        print(df_bkg.columns)



display_dataframes()
print_h5_variables(signal)
