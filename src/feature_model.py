"""
These functions perform feature and model selection 
"""
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error
import pandas as pd
from lazypredict.Supervised import LazyRegressor


# Using mutual info to select features with high impact on the target feature
def make_mi(X, y):

    # Copy the dataset for not losing it
    X = X.copy()

    # Mutual info needs all the instances to be numerical type
    for col in X.select_dtypes(["object", "category"])
        X[col], _ = X[col].factorize()
        
    discrete = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    
    # Make an instance of mutal info
    mi = mutual_info_regression(X, y, discrete_features = discrete, random_state = 0)
    
    # Serires of the result
    mi = pd.Series(mi, name = "MI SCORES", index = X_train.columns)
    # Sorting the values for highest to lowest
    mi  = mi.sort_values(ascending = False)
    return mi

# LazyRegressor is a beautiful tool to have a list of regressors models
# which fit well our dataset
def Lazy(X, y):
  reg = LazyRegressor(ignore_warnings=False, custom_metric=None)

  # We've only choose 10,000 rows because lazyregressor is pretty memory consuming
  models, predictions = reg.fit(X.iloc[:10000, :], X[:10000, :],y[:10000], y[:10000])
  print(models)

# Have an idea of the perdomance of each model
def model_select(models, X_train, y_train, X_val, y_val):
    # We want the records in a dataframe format
    models_summary = pd.DataFrame([], columns = ["Model Name", "Score", "Mean_Absolute_Error"])
    for model in models:
        mo = model()
        mo.fit(X_train, y_train)
        mo_predicted = mo.predict(X_val)
        mo_score = mo.score(X_val, y_val)
        
        # We'll use mean_absolute_error as metric for prediction
        mea = mean_absolute_error(y_val, mo_predicted)
        models_summary = models_summary.append({
            "Model Name" : mo.__class__.__name__,
            "Score" : mo_score,
            "Mean_Absolute_Error" : mea
        }, ignore_index = True)
    return models_summary.sort_values("Score", ascending = False)
