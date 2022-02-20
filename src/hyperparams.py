# We'll define the functions and their repective parameters

# Libraries
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import optuna
from xgboost import XGBRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor
from lightgbm import LGBMRegressor
from .scoring import scored


rfr_model = RandomForestRegressor()
rfr_params = {
    "n_estimators" : [600,750,800,850],
    "max_features": [7, 9, 13],
    "min_samples_leaf": [3, 5, 7],
    "min_samples_split": [4, 6, 9],
    "random_state": [10, 31, 50, 100]
    
}

xgb_model = XGBRegressor()
xgb_params = {
    "n_estimators" : [110, 120, 130, 140],
    "learning_rate" : [0.05, 0.075, 0.1],
    "max_depth" : [7, 9, 13],
    "reg_lambda" : [0.3, 0.5, 0.7]
}

HGBR_model = HistGradientBoostingRegressor()
HGBR_params =  {
    'n_estimators': [150, 300, 450, 600],
    'max_depth': [7, 9, 11],
    'min_samples_split': [3, 7, 10],
    'min_samples_leaf': [5, 7, 9],
    'learning_rate': [0.01, 0.02, 0.4, 0.7],
    'max_features': [0.8, 0.7, 0.9],
    'loss': ['ls', 'lad', 'huber']
}

LGBMR_model = LGBMRegressor()
LGBMR_params = {
    "number_leaves" : [10, 31, 50, 100],
    "max_depth": .1,
    "learning_rate" : [0.01, 0.2, 0.5, 0.7],
    "n_estimators" : [50, 100, 150, 200]
}

HGBR_model = HistGradientBoostingRegressor()
HGBR_model = {loss='squared_error',
				learning_rate=0.1,
				max_iter=100,
				max_leaf_nodes=31,
				max_depth=None,
				min_samples_leaf=20,
				l2_regularization=0.0,
				max_bins=255,
				monotonic_cst=None,
				warm_start=False,
				early_stopping='auto',
				scoring='loss',
				validation_fraction=0.1,
				n_iter_no_change=10,
				tol=1e-07,
				verbose=0,
				random_state=None}


# We are about to tune our hyper-parameters

def tunning(model, params, X_t, y_t, X_v, y_v):
	# Our summary datadrame for records
    summary = pd.DataFrame([], columns = ["Model Name", "Score", "Mean_Absolute_Error"])

    # Applying GridSearh for hyperparams tunning
    model_reg = GridSearchCV(estimator = model, param_grid = params, cv=5, n_jobs = -1)
    model_reg.fit(X_t, y_t)
    model_score = model_reg.best_score_
    model_pred = model_reg.predict(X_v)
    mae = mean_absolute_error(y_v, model_pred)
    summary.append({
        "Model Name": model.__class__.__name__ + "tunned",
        "Score": model_score,
        "Mean_Absolute_Error": mae
    }, ignore_index = True)

    print("Best Score: ", model_reg.best_score_)
    print("Best params: ", model_reg.best_params_)
    print("mae: ", mae)
    return summary

# We'll also use optuna because GridSearch is time and memory consuming

# Optuna use case Especially for HistGradientBoostingRegressor
def objective_HGBR(trial):

	X_train, X_val, y_train, y_val = load_data(target_col, random_state, test_size)

	# Setting hyper-params
	HGBR_params = dict(loss = trial.suggest_categorical("loss", 'least_squares'),
	                    learning_rate = trial.suggest_float("learning_rate", 0.2, 0.4),
	                    max_iter= trial.suggest_int("max_iter", 100, 200, 300),
	                    max_leaf_nodes= trial.suggest_int("max_leaf_nodes", 15, 31, 60), 
	                    max_depth= trial.suggest_int("max_depth" , 7, 9, 15),
	                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 50, 80),
	                    l2_regularization= trial.suggest_float("l2_regularization", 0.3, 0.7),                                           scoring='loss',
	                    tol = trial.suggest_float("tol", 1e-05, 1e-02)
	                  )
	  
	HGBR_t = HistGradientBoostingRegressor(**HGBR_params)
	log_y = np.log(y_train)
	score = cross_val_score(HistGradientBoostingRegressor(), , X_train, y_train, cv = 10, scoring = "least_squares")
	score = -1 * score.mean()
	score = np.sqrt(score)
	return score


def objective_LGBMR(trial):
	LGBMR_params = dict(
	number_leaves = trial.suggest_int("number_leaves",15, 60, 80),
	max_depth = trial.suggest_int("max_depth", 7, 9, 14),
	learning_rate = trial.suggest_float("learning_rate", 0.01, 0.7),
	n_estimators = trial.suggest_int("n_estimators", 25, 45, 75),
	)
	LGBM_t = LGBMRegressor(**LGBMR_params)
	return scored(X_train, y_train, LGBM_t)
