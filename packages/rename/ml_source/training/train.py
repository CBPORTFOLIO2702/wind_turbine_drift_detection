from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

def prepare_dataset(df: pd.DataFrame, target_var: str, val_test_size: float, test_size: float):
    X = df.drop(columns = target_var)
    y = df[target_var]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_test_size)

    return X_train, X_val, y_train, y_val

def train_model(X_train, y_train):
    # Change this at will!
    model = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth = 30)
    trained_model = model.fit(X_train, y_train)

    return trained_model

def evaluate_model(model, X_val, y_val):
    results = 'write evaluation method here!'
    print(results)
    return results

def get_model_feature_importance(model, X_val):
    importances = model.feature_importances_
    features = X_val.columns
    importance_df = pd.DataFrame({'feature': features, 'importances': importances}, index = range(len(importances)))
    importance_df['model_name'] = 'rename'
    importance_df.to_csv('results/feature_importance_rename.csv')

    return importance_df