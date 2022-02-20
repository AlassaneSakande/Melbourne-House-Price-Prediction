MELBOURNE-HOUSE-PRICE-PREDICTION

This is an ML regression project concerning the prediction of the prices of houses in Melbourne.

Set up

It has been developped on Google Colabs due to time and memory consumption.

You can get the required materials from the requirements.txt file

This repository is only on training purposes for any ML practionner who want to find out other ways to deal with this classic ML project.

Datasets

The dataset contains initially 34,857 instances with 15 columns, we've made spliting of:
training : 24,399
test : 10,458

Pre-training

We've finally worked with 43 columns after One-hot-encoding 

Metrics and Results

We've achieved :

Model							Score 		Mean_Absolute_Error

HistGradientBoostingRegressor 	0.57 		220792.68
LGBMRegressor 	                0.57 		222044.02
XGBRegressor                	0.55 		220782.39
GradientBoostingRegressor 	    0.54 		238479.73
RandomForestRegressor 	        0.52 		225812.64





