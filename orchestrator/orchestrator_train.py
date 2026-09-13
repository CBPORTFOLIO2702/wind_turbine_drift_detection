# add orchestration details here
import sys
import os
sys.path.insert(1, os.getcwd())
import pandas as pd
from packages.rename.ml_ops.data_import.data_import import run_import_data
from packages.rename.ml_ops.data_processing.data_processing import run_data_processing
from packages.rename.ml_ops.training.train import run_model_training

path = 'rename'
print(f'Importing data from {path}.')
df = run_import_data(path = path, name = 'mushrooms')
print('Successfully imported data! See head sample below:')
print(df.head())

print('Processing data for feature engineering.')
df = run_data_processing(df)
print('Successfully run feature engineering for dataset! See head sample below:')
print(df.head())

print('Running model training. This may take some time.')
model = run_model_training(df, 0.3, 0.5)
print('Successfully completed model training.')