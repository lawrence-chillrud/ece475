import matplotlib.pyplot as plt
import pandas as pd

methods = ['distance_correlation', 'lasso', 'xgboost', 'random_forest'] # , 'gbdt'



for method in methods:
	data = pd.read_csv("data/generated/feature_selection_output/"+method+"_features.csv")
	features = data['feature'].tolist()
	importance = data['importance'].tolist()

	if(method=='distance_correlation'):
		features = features[0:20]
		importance = importance[0:20]

	plt.rc('figure', figsize=(12, 9))
	fig, ax = plt.subplots()
	fig.subplots_adjust(left=0.45)
	plt.gca().invert_yaxis()

	# Create the bar chart
	ax.barh(features, importance)

	plt.title('Correlation of Features According to '+method)
	#plt.ylabel('Feature')
	plt.xlabel('Correlation')
	plt.savefig("results/"+method+"_features.png")
