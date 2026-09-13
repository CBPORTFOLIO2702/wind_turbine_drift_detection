### DATA IMPORT MLOPS ###
from packages.rename.ml_source.data_import.data_import import import_data
import logging
import pandas as pd

def run_import_data(path, name):
    print(f'Importing {name} data from {path}')
    try:
        df = import_data(path)
    except:
        raise Exception('ERROR: There was an issue with data import. Please check file formatting.')
    print(f'Success! {name} dataframe has been imported!')
    print(f'{name} has {len(df)} rows and {len(df.columns)} columns.')
    return df
