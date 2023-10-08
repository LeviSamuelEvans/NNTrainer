import pandas as pd
#import numpy as np

"""
A utility script to help inspect the signal and background
datasets and display the first few rows of the DataFrames.

"""

# Set paths to the signal and background datasets
signal ='../signal2.h5'
background = "../ttbarBackground2.h5"
test = "../test.h5"

# Read the signal and background datasets into Pandas dataframes
df_sig = pd.read_hdf(signal, key="df")
df_bkg = pd.read_hdf(background, key="df")
df_test: pd.DataFrame = pd.read_hdf(test, key="df")

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
    print("========================================")
    print('Summary Statistics for the test sample')
    print("========================================")
    print(df_test.describe())
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

        print("Column names for test dataframe:")
        print(df_test.columns)

def display_jetTagWeight_values():
    """
    Displays the first few values of the jetTagWeight column in the test sample.

    Parameters:
    None

    Returns:
    None
    """
    if 'jet_tagWeightBin_DL1r_Continuous_4' in df_test.columns:
        print("========================================")
        print('First few values of jetTagWeight in the test sample')
        print("========================================")
        print(df_test['jet_tagWeightBin_DL1r_Continuous_4'].head(20)) # Number of rows to display
    else:
        print("'jet_tagWeightBin_DL1r_Continuous_1' column not found in the test sample.")

def display_jet_pt():
    """
    Displays the first few values of the jetTagWeight column in the test sample.

    Parameters:
    None

    Returns:
    None
    """
    if 'jet_pt_1' in df_test.columns:
        print("========================================")
        print('First few values of jet pT in the test sample')
        print("========================================")
        print(df_test['jet_pt_1'].head(20)) # Number of rows to display
    else:
        print("'jet_pt_1' column not found in the test sample.")


display_dataframes()
display_jetTagWeight_values()
display_jet_pt()
print_h5_variables(signal)
print_h5_variables(test)