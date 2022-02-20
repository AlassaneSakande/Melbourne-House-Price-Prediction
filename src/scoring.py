from sklearn.model_selection import cross_val_score

# Let's defining some scoring methods for HGBRegressor and LGBMRegressor 
def scored(X, y, model = HistGradientBoostingRegressor()):
  log_y = np.log(y)
  score = cross_val_score(model, X, log_y, cv=5, scoring = "neg_mean_squared_error")
  score = -1 * score.mean()
  score = np.sqrt(score)
  return score