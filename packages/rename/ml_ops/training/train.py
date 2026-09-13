import sys
import os
sys.path.insert(1, os.getcwd())
from packages.rename.ml_source.training.train import prepare_dataset, train_model, evaluate_model, get_model_feature_importance

def run_model_training(df, test_val_size, test_size):
    print('Splitting data into training, validation and testing sets.')
    try:
        X_train, X_val, y_train, y_val= prepare_dataset(df, 'class', test_val_size, test_size)
    except:
        raise Exception('There was an issue with the splitting of the data.')
    print('Done. See below for model input:')
    print(X_train.head())
    print('Training model.')
    try:
        model = train_model(X_train, y_train)
    except:
        raise Exception('There was an issue with model training.')
    print('Done')
    print('Collecting feature importances')
    importances = get_model_feature_importance(model, X_val=X_val)
    print('Done. See below for model feature importances:')
    print(importances)
    print('Validating model')
    try:
        evaluation = evaluate_model(model, X_val, y_val)
    except:
        raise Exception('There was an issue with validating the model.')
    print('Done')
    print(evaluation)

    return model