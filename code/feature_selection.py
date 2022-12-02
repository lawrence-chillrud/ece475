import autograd.numpy as np
from autograd import grad, hessian
import random
import math
import matplotlib.pyplot as plt
import pandas
from matplotlib import ticker
import dcor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


num_features = 20
method = "forest"

# Colums that will be used for y
use_cols_y = 24

# import data
x_raw_data = pandas.io.parsers.read_csv("../data/generated/NSCLC_features.csv").values
x_data = x_raw_data[:,1:]

x = x_data.T

y_raw_data = pandas.io.parsers.read_csv("../data/generated/NSCLC_labels.csv").values
y_data = y_raw_data[:,use_cols_y]
y = y_data.T

#values that are not in x
y=np.delete(y,118-1)
y=np.delete(y,59-2)

inds = np.where((y=="Wildtype") | (y == "Mutant"))
x = x[:,inds[0]]
y = y[inds]

y[y=="Wildtype"]=-1
y[y=="Mutant"]=1

distance_correlations = []

if method == "forest":
	rf = RandomForestRegressor(n_estimators=100)
	rf.fit(x.T, y)
	distance_correlations=rf.feature_importances_
elif method == "distance":
	for i in range(x.shape[0]):
		distance_correlations +=[dcor.distance_correlation(x[i],y)]

top_features_indexes = np.argsort(distance_correlations)[-num_features:]
print(top_features_indexes[::-1])