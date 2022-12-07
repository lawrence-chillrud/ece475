from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import make_scorer, get_scorer_names, confusion_matrix, f1_score, balanced_accuracy_score, matthews_corrcoef, plot_confusion_matrix
from sklearn.metrics import make_scorer, confusion_matrix, f1_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef, plot_confusion_matrix, classification_report, precision_score, recall_score
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import cross_val_score
import joblib
import pandas as pd
from utils import prep_data
import numpy as np

methods = ['distance_correlation', 'lasso', 'xgboost', 'random_forest'] # , 'gbdt'

metrics_df = pd.DataFrame()
train_metrics_df = pd.DataFrame()
for m in methods:
	[X_train, X_test, y_train, y_test] = prep_data()
	data = pd.read_csv("data/generated/feature_selection_output/"+m+"_features.csv")
	if m!="distance_correlation":
		use_cols_x = data['feature'].tolist()
	else:
		use_cols_x = (data['feature'].tolist())[0:1000]

	# Columns that will be used for x, y
	X_train = X_train[use_cols_x]      
	X_test = X_test[use_cols_x]

	scoring = {'AUC': make_scorer(roc_auc_score), 'F1': make_scorer(f1_score), 'Balanced Acc': make_scorer(balanced_accuracy_score), 'MCC': make_scorer(matthews_corrcoef), 'Precision': make_scorer(precision_score), 'Recall': make_scorer(recall_score)}
	cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=3, random_state=1)

	grid={"C":[0.001,0.01,0.1,1,10], "solver":["saga"], "fit_intercept":[True,False],"class_weight":["balanced"], "l1_ratio":[0.5],"penalty":["l1","l2","none","elasticnet"]}# l1 lasso l2 ridge
	logreg=LogisticRegression()
	logreg_cv=GridSearchCV(logreg,grid, scoring=scoring,refit='AUC', cv=cv)
	logreg_cv.fit(X_train,y_train)


	# 6e. Save results
	results = logreg_cv.cv_results_
	keys = [k for k, v in results.items() if k.startswith('split')]
	for x in keys:
		del results[x]

	results_df = pd.DataFrame().from_dict(results)
	print("\nBEST PARAMS:\n")
	print(logreg_cv.best_params_)
	joblib.dump(logreg_cv.best_estimator_, f"data/generated/models/SVM_{m}.pkl")
	results_df.to_csv(f"data/generated/models/SVM_{m}_results_df_final.csv", index=False)

	train_results = pd.read_csv(f"data/generated/models/SVM_{m}_results_df_final.csv")

	# 10. Performance on test
	print("\nPERFORMANCE ON THE TEST SET:\n")
	y_test_hat = logreg_cv.best_estimator_.predict(X_test)
	metrics = pd.DataFrame({
		'AUC': roc_auc_score(y_test, y_test_hat), 
		'F1': f1_score(y_test, y_test_hat), 
		'Balanced_Acc': balanced_accuracy_score(y_test, y_test_hat), 
		'MCC': matthews_corrcoef(y_test, y_test_hat), 
		'Precision': precision_score(y_test, y_test_hat),
		'Recall': recall_score(y_test, y_test_hat),
		'feature_selector': m
	}, index = [0])
	metrics_df = pd.concat([metrics_df, metrics])
	print("\nConfusion matrix:\n", confusion_matrix(y_test, y_test_hat))
	print("\nAUC: ", roc_auc_score(y_test, y_test_hat))
	print("\nF1 score: ", f1_score(y_test, y_test_hat))
	print("\nBalanced acc: ", balanced_accuracy_score(y_test, y_test_hat))
	print("\nMCC: ", matthews_corrcoef(y_test, y_test_hat))
	tr = pd.DataFrame(train_results.sort_values('mean_test_AUC', ascending=False).loc[0,:]).T
	train_metrics_df = pd.concat([train_metrics_df, tr])
	train_metrics_df = train_metrics_df.filter(train_metrics_df.columns[train_metrics_df.columns.str.contains('mean')])
	train_metrics_df['classifier'] = 'logistic'
	metrics_df['classifier'] = 'logistic'

train_metrics_df.to_csv('data/generated/stats/logistic_regression_train.csv', index=False)
metrics_df.to_csv('data/generated/stats/logistic_regression_test.csv', index=False)
