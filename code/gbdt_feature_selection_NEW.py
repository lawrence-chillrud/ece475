# %%Import libraries
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import warnings
from utils import prep_data
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import make_scorer, confusion_matrix, f1_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef, plot_confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier


# 1. Set up directories, read in data
warnings.filterwarnings('ignore')
X_train_val, X_test, y_train_val, y_test = prep_data(labels=[1, -1])

# 2. Training and validation splits
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=0, stratify=y_train_val)

print("TRAIN: \n", y_train.value_counts())
print("VAL: \n", y_val.value_counts())
print("TEST: \n", y_test.value_counts())

train_class_counts = y_train.value_counts().values
inverse_weight = train_class_counts[0]/train_class_counts[1]

# 3. Identify metrics we care about (for either gridsearch or cross val later)
scoring = {'AUC': make_scorer(roc_auc_score), 'F1': make_scorer(f1_score), 'Balanced Acc': make_scorer(balanced_accuracy_score), 'MCC': make_scorer(matthews_corrcoef)}

# 4. Set up k-fold cross-val scheme (for either gridsearch or cross val later)
cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=3, random_state=1)

# 5. Decide if gridsearch needed or not
run_grid_search = False

# 6. Set up and run gridsearch of xgboost hyperparameters if need be
if run_grid_search:
    # 6a. Initialize GBDT
    estimator = GradientBoostingClassifier()

    # 6b. Define grid of hyperparams we want to search through
    params = {
        'loss': ["log_loss"],
        'learning_rate': [0.001, 0.01, 0.1],
        'n_estimators': [100, 200],
        'criterion': ["squared_error"],
        'max_depth': [3, 4, 5],
        'verbose': [1]
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
    gs.fit(X_train_val, y_train_val)

    # 6e. Save results
    results = gs.cv_results_
    keys = [k for k, v in results.items() if k.startswith('split')]
    for x in keys:
        del results[x]

    results_df = pd.DataFrame().from_dict(results)
    print("\nBEST PARAMS:\n")
    print(gs.best_params_)
    joblib.dump(gs.best_estimator_, 'data/generated/models/gbdt_final.pkl')
    results_df.to_csv('data/generated/models/gbdt_results_df_final.csv', index=False)

# 7. Load best model resulting from gridsearch
model = joblib.load('data/generated/models/gbdt_final.pkl')

# %%
y_val_hat = model.predict(X_val)
print("\nConfusion matrix:\n", confusion_matrix(y_val, y_val_hat))
print("\nBalanced acc: ", balanced_accuracy_score(y_val, y_val_hat))
print("\nF1 score: ", f1_score(y_val, y_val_hat))
print("\nMCC: ", matthews_corrcoef(y_val, y_val_hat))

# %%
y_train_hat = model.predict(X_train)
print("\nConfusion matrix:\n", confusion_matrix(y_train, y_train_hat))
print("\nBalanced acc: ", balanced_accuracy_score(y_train, y_train_hat))
print("\nF1 score: ", f1_score(y_train, y_train_hat))
print("\nMCC: ", matthews_corrcoef(y_train, y_train_hat))

# %%
# xgb.plot_importance(model, max_num_features=30)

# %%
thresholds = np.sort(model.feature_importances_)
thresholds = thresholds[np.where(thresholds != 0)]
thresholds = np.insert(thresholds, 0, 0, axis=0)
thresholds = thresholds[-30:]
metrics_df = pd.DataFrame()
X_train_val = X_train_val.reset_index(drop=True)
y_train_val = y_train_val.reset_index(drop=True)
cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=1, random_state=1)

for train_index, val_index in cv.split(X_train_val, y_train_val):
    X_train, X_val = X_train_val.iloc[train_index, :], X_train_val.iloc[val_index, :]
    y_train, y_val = y_train_val[train_index], y_train_val[val_index]
    for thresh in thresholds:
        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)

        # train model
        selection_model = GradientBoostingClassifier(**model.get_params())
        selection_model.fit(select_X_train, y_train)

        # eval model
        select_X_val = selection.transform(X_val)
        y_pred = selection_model.predict(select_X_val)

        metrics = pd.DataFrame({
            'AUC': roc_auc_score(y_val, y_pred),
            'F1': f1_score(y_val, y_pred),
            'Balanced_Acc': balanced_accuracy_score(y_val, y_pred),
            'MCC': matthews_corrcoef(y_val, y_pred),
            'thresh': thresh,
            'val_idx': np.array2string(np.array(val_index))
        }, index = [0])
        metrics_df = pd.concat([metrics_df, metrics])
    print('Done fold')

# %% Print summary of best run's features, save for downstream use
metrics_summary = metrics_df.groupby(['thresh'], as_index=False).mean().sort_values('AUC', ascending=False)
best_thresh = metrics_summary['thresh'].iloc[0]
print("\nBest stats:\n", metrics_summary.iloc[0,:])
selection = SelectFromModel(model, threshold=best_thresh, prefit=True)
best_feature_idx = selection.get_support()
top_features = X_train_val.columns[best_feature_idx]
top_importances = model.feature_importances_[best_feature_idx]
top_df = pd.DataFrame({'feature': top_features, 'importance': top_importances}).sort_values('importance', ascending=False)
top_df.to_csv('data/generated/feature_selection_output/gbdt_features.csv', index=False)
# %%
