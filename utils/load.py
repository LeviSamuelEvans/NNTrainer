import pandas as pd

def load_data(signal_path, background_path, features):
    df_sig = pd.read_hdf(signal_path, key="df")
    df_bkg = pd.read_hdf(background_path, key="df")
    # Filter the dataframes based on the features
    df_sig = df_sig[features]
    df_bkg = df_bkg[features]

    return df_sig, df_bkg
