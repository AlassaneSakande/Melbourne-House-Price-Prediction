"""
The load_data function :

    Call the correct and impute functions
    Specifying the trainng and testing data
    Scale the dataset then slicing it
"""
import datetime
import pandas as pd
from .correct_data import correct
from .impute_data import impute
from sklearn.model_selection import train_test_split
import numpy as np
"""
The load_data takes as arguments the target feature, random_state and size of
the test data to perform splitting
"""

def load_data(target, random_state, test_size):

    # importing the dataset
    dataset = pd.read_csv("/home/alassane/Documents/DATA/Melbourne/Melbourne_housing_FULL.csv")
    
    # Performing correction and imputation
    dataset = correct(dataset, new_date_col, year_col, cols_to_drop)
    dataset = impute(dataset, OHE_cols)

    # Splitting the data in training and testing
    X = dataset.drop([target], axis = 1)
    y = dataset[target]

    # Scale the data so the values will fit in an appropriate range
    X_scaled = (X - X.mean(axis = 0)) / X.std(axis=0)

    # Splitting randomly the data with according to the test size
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, random_state=random_state, test_size= test_size)
    
    # Returning all the splits
    return X_train, X_val, y_train, y_val