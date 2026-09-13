import sys
import os
sys.path.insert(1, os.getcwd())
from packages.rename.ml_source.data_processing.data_processing import rename

def run_data_processing(df):
    print('Running data processing on dataframe.')
    # Insert code here
    rename()

    return df