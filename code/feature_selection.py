import pandas as pd
import dcor
from sklearn.ensemble import RandomForestRegressor
import warnings
import csv

# Returns num_features number of the top features of x in predictin y, according to method
def get_features(x,y,num_features,method):
	correlations = {}
	if method == "distance_correlation":
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			for i in range(len(x.columns)):
				correlations[x.columns[i]] = dcor.distance_correlation(x[x.columns[i]],y)
	elif method == "random_forest":
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			rf = RandomForestRegressor(n_estimators=100)
			rf.fit(x, y)
			fi=rf.feature_importances_
			for i in range(len(x.columns)):
				correlations[x.columns[i]] = fi[i]
	else:
		raise Exception("Unknown method selected")
	top_features_indexes = sorted(correlations.items(), key=lambda item: item[1],reverse=True)
	with open(method+'.csv', 'w') as f:
		f.write("feature,value\n")
		for v in top_features_indexes:
			f.write("%s,%s\n"%(v[0],v[1]))

	return top_features_indexes[0:num_features]
