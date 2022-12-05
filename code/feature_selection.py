import numpy as np
import pandas
import dcor
from sklearn.ensemble import RandomForestRegressor
import warnings

# Returns num_features number of the top features of x in predictin y, according to method
def get_features(x,y,num_features,method):
	correlations = []
	match method:
		case "distance_correlation":
			for i in range(x.shape[0]):
				correlations +=[dcor.distance_correlation(x[i],y)]
		case "random_forest":
			with warnings.catch_warnings():
				warnings.simplefilter('ignore')
				rf = RandomForestRegressor(n_estimators=100)
				rf.fit(x.T, y)
				correlations=rf.feature_importances_
		case _:
			raise Exception("Unknown method selected")

	top_features_indexes = np.argsort(correlations)[-num_features:]
	return top_features_indexes[::-1]

# Returns an array of [x,y] of the NSCLC extracted data
def  get_data():
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

	# Select only valid y entires and convert to numerical values
	inds = np.where((y=="Wildtype") | (y == "Mutant"))
	x = x[:,inds[0]]
	y = y[inds]
	y[y=="Wildtype"]=-1
	y[y=="Mutant"]=1

	return [x,y]

warnings.filterwarnings("ignore")
[x,y] = get_data()
get_features(x,y,20,"distance_correlation")

