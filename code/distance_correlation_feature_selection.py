import pandas as pd
import dcor
from sklearn.ensemble import RandomForestRegressor
import warnings
import csv
from utils import prep_data

[X_train, X_test, y_train, y_test] = prep_data()

correlations = {}
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for i in range(len(X_train.columns)):
        correlations[X_train.columns[i]] = dcor.distance_correlation(X_train[X_train.columns[i]],y_train)
top_features_indexes = sorted(correlations.items(), key=lambda item: item[1],reverse=True)
with open('data/generated/feature_selection_output/distance_correlation_features.csv', 'w') as f:
    f.write("feature,importance\n")
    for v in top_features_indexes:
        f.write("%s,%s\n"%(v[0],v[1]))

