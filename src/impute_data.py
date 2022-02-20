# Now , we will fix NaN values and do some Encoding to few categorical variables

"""
The impute function:

    Make an instance of the imputer
    Select numerical cols and apply One-Hot-Encoding to them
"""
import pandas as pd
from sklearn.impute import SimpleImputer

"""
The impute function takes as params the dataset returned by the correct
function and the set of cols on which we'll perform One-Hot-Encoding
because there is no order for these categorical cols with fewer unique values
"""
def impute(dataset, OHE_cols):

    # Making an instance of the imputer
    imputer = SimpleImputer()

    # Applying imputation for all numeric values in the dataset
    imputed_data = pd.DataFrame(imputer.fit_transform(dataset.select_dtypes(exclude = ["object"])))
    
    # Imputation removes columns, let's put them back
    imputed_data.columns = dataset.select_dtypes(exclude = ["object"]).columns
    dataset_imputed = imputed_data
    
    # One-Hot-Encoding for a set of cols
    dummies = pd.get_dummies(dataset, columns = OHE_cols)
    
    # select cols in our original dataset in order to ensure our
    # One-Hot-Encoding process doesn't involve them
    not_dummies_col = [col for col in imputed_data.columns]
    dummies = dummies.drop(not_dummies_col, axis = 1)
    
    # Concatenate our numerical data with One-Hot-encoding ones
    dataset = pd.concat([dataset_imputed, dummies.reindex(dataset.index)], axis = 1)
    
    # One again let's see the output
    return dataset