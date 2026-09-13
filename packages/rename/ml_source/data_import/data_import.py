### DATA IMPORT ML SOURCE ###
import pandas as pd

def import_data(path):
    full_path = path
    df = pd.read_csv(full_path)
    return df