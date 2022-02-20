# The correct function
"""
After viewing some cols and their values we've found it's wiser
to make some changes to make the processing easier
"""
import pandas as pd
import datetime

"""
correct takes as params the path to the dataset, the col of dates in our dataset
as well as its the new datetime col transform and the set of some
cols we've found useless.
!! We've dropped categorical cols with a lot of unique values
"""

def correct(path, d_col, y_col, cols_drop):
	# importing the dataset
    dataset = pd.read_csv("/home/alassane/Documents/DATA/Melbourne/Melbourne_housing_FULL.csv")
	
	# Transforming the "Date" col initially of wrong type to Datetime type   
    dataset[d_col] = pd.to_datetime(dataset["Date"], infer_datetime_format = True)
    
    # Let's select only the year for the new "Datetime" type col
    dataset[y_col] = dataset[d_col].dt.year

    # Here we are dropping all unecessary cols
    dataset = dataset.drop(cols_drop , axis = 1)

    # Finally it's always better to see the output before moving on
    return dataset