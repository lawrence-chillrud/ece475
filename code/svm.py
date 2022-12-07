# File: svm.py
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Date: 11/30/2022

# %%0. Package importants
from sklearn.metrics import make_scorer, confusion_matrix, f1_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef, plot_confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.svm import SVC
from utils import prep_data
import numpy as np
import pandas as pd
import joblib
import os

X_train_val, X_test, y_train_val, y_test = prep_data(labels=[1, 0])
methods = ['distance_correlation', 'lasso', 'xgboost', 'random_forest'] # , 'gbdt'

scoring = {'AUC': make_scorer(roc_auc_score), 'F1': make_scorer(f1_score), 'Balanced Acc': make_scorer(balanced_accuracy_score), 'MCC': make_scorer(matthews_corrcoef)}
cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=3, random_state=1)

print("SVM TRAINING AND TESTING:\n")

# %%1. Loop through all feature selectors
for m in methods:
    print(f"\nFeature selection technique: {m}\n")
    if m != 'all':
        features_df = pd.read_csv(f"data/generated/feature_selection_output/{m}_features.csv")
        features = features_df['feature'].tolist()
    else:
        features = X_train_val.columns
    
    X_tv = X_train_val[features]
    X_t = X_test[features]
    
    if f"SVM_{m}.pkl" not in os.listdir('data/generated/models/'):
        estimator = SVC()

        # 6b. Define grid of hyperparams we want to search through
        params = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            #'degree': [2, 3, 4, 5, 10],
            'gamma': ['scale', 'auto']
        }

        # 6c. Initialize gridsearch
        gs = GridSearchCV(
            estimator=estimator,
            param_grid=params,
            scoring=scoring,
            refit='AUC',
            n_jobs=-1,
            cv=cv,
            return_train_score=True,
            verbose=10
        )

        # 6d. Run gridsearch
        gs.fit(X_tv, y_train_val)

        # 6e. Save results
        results = gs.cv_results_
        keys = [k for k, v in results.items() if k.startswith('split')]
        for x in keys:
            del results[x]

        results_df = pd.DataFrame().from_dict(results)
        print("\nBEST PARAMS:\n")
        print(gs.best_params_)
        joblib.dump(gs.best_estimator_, f"data/generated/models/SVM_{m}.pkl")
        results_df.to_csv(f"data/generated/models/SVM_{m}_results_df_final.csv", index=False)

    model = joblib.load(f"data/generated/models/SVM_{m}.pkl")

    # 10. Performance on test
    print("\nPERFORMANCE ON THE TEST SET:\n")
    y_test_hat = model.predict(X_t)
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_test_hat))
    print("\nAUC: ", roc_auc_score(y_test, y_test_hat))
    print("\nF1 score: ", f1_score(y_test, y_test_hat))
    print("\nBalanced acc: ", balanced_accuracy_score(y_test, y_test_hat))
    print("\nMCC: ", matthews_corrcoef(y_test, y_test_hat))
    print(classification_report(y_test, y_test_hat))
# %%
