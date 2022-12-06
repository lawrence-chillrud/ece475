from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import make_scorer, get_scorer_names, confusion_matrix, f1_score, balanced_accuracy_score, matthews_corrcoef, plot_confusion_matrix
from sklearn.metrics import make_scorer, confusion_matrix, f1_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef, plot_confusion_matrix
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import cross_val_score
import pandas as pd
from feature_selection import get_features
from utils import prep_data
import numpy as np

[X_train, X_test, y_train, y_test] = prep_data()

# Columns that will be used for x, y
use_cols_x = []
features = get_features(X_train,y_train,20,"distance_correlation")
print(features)
exit()
for f in features:
	use_cols_x += [f[0]]

X_train = X_train[use_cols_x]
X_test = X_test[use_cols_x]

scoring = {'AUC': make_scorer(roc_auc_score), 'F1': make_scorer(f1_score), 'Balanced Acc': make_scorer(balanced_accuracy_score), 'MCC': make_scorer(matthews_corrcoef)}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

grid={"C":[0.001,0.01,0.1,1,10], "solver":["saga"], "fit_intercept":[True,False],"class_weight":["balanced"], "l1_ratio":[0.5],"penalty":["l1","l2","none","elasticnet"]}# l1 lasso l2 ridge
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid, scoring=scoring,refit='AUC', cv=cv)
logreg_cv.fit(X_train,y_train)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)

# create a LogisticRegression object
y_pred = logreg_cv.predict(X_test)

# evaluate the model performance
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nBalanced acc: ", balanced_accuracy_score(y_test, y_pred))
print("\nF1 score: ", f1_score(y_test, y_pred))
print("\nMCC: ", matthews_corrcoef(y_test, y_pred))
