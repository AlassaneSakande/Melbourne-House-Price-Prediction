## MELBOURNE-HOUSE-PRICE-PREDICTION

<p align="center">
<img src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/python/python.png" alt="Python" height="40" style="vertical-align:top; margin:4px">
<img src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/jupyter-notebook/jupyter-notebook.png" alt="jupyter-notebook" height="40" style="vertical-align:top; margin:4px">
<img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252"/>
</p>

![melb](https://user-images.githubusercontent.com/84173235/177981587-3214c4a0-67a6-43d5-860f-5255b5ad96f9.jpeg)


This is a ML regression project concerning the prediction of the prices of houses in Melbourne.

# Set up

It has been developped on Google Colabs due to time and memory consumption.

You can get the required materials from the requirements.txt file

This repository is only on training purposes for any ML practionner who want to find out other ways to deal with this classic ML project.

# Datasets

We've used the Melboune_house_Price_FULL dataset from Kaggle.

The dataset contains initially 34,857 instances with 15 columns, we've made spliting of:

training : 24,399

test : 10,458

# Pre-training

We've finally worked with 43 columns after One-hot-encoding 

# Metrics and Results

We've achieved :

Model							Score 		Mean_Absolute_Error

HistGradientBoostingRegressor 	0.57 		220792.68

LGBMRegressor 	                0.57 		222044.02

XGBRegressor                	0.55 		220782.39

GradientBoostingRegressor 	    0.54 		238479.73

RandomForestRegressor 	        0.52 		225812.64





