# %%0. Package imports
import numpy as np
import pandas as pd
from utils import prep_data
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import make_scorer, confusion_matrix, f1_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef, plot_confusion_matrix
from sklearn.feature_selection import SelectFromModel
import joblib
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import warnings

# %%1. Set up directories, read in data
warnings.filterwarnings('ignore')
X_train_val, X_test, y_train_val, y_test = prep_data(labels=[1, 0])
#X.hist(grid=False, figsize=[20,10], column=X.columns[1:100])    
#plt.tight_layout()

# %%2. training and validation splits
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=0, stratify=y_train_val)

print("TRAIN: \n", y_train.value_counts())
print("VAL: \n", y_val.value_counts())
print("TEST: \n", y_test.value_counts())

train_class_counts = y_train.value_counts().values
inverse_weight = train_class_counts[0]/train_class_counts[1]

# %%3. train xgboost classifier
estimator = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    scale_pos_weight=inverse_weight,
    use_label_encoder=False
)

params = {
    'max_depth': [2, 3],
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.1, 0.01],
    'colsample_bytree': [0.5, 0.75, 1.0],
    'subsample': [0.6, 0.8, 1.0],
    'alpha': [0, 1],
    'min_child_weight': [1, 5]
}

cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=2, random_state=1)

scoring = {'AUC': make_scorer(roc_auc_score), 'F1': make_scorer(f1_score), 'Balanced Acc': make_scorer(balanced_accuracy_score), 'MCC': make_scorer(matthews_corrcoef)}

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

gs.fit(X_train_val, y_train_val)
results = gs.cv_results_
keys = [k for k, v in results.items() if k.startswith('split')]
for x in keys:
    del results[x]

results_df = pd.DataFrame().from_dict(results)
print(gs.best_params_)
joblib.dump(gs.best_estimator_, 'data/generated/models/xgboost.pkl')
results_df.to_csv('data/generated/models/xgboost_results_df.csv', index=False)
exit()

# %%
model = gs.best_estimator_.fit(X_train, y_train)

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
plot_importance(model, max_num_features=30)

# %%
thresholds = np.sort(model.feature_importances_)
thresholds = thresholds[np.where(thresholds != 0)]
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = XGBClassifier(**model.get_params())
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_val = selection.transform(X_val)
    y_pred = selection_model.predict(select_X_val)
    f1 = f1_score(y_val, y_pred)
    print("Thresh=%.3f, n=%d, F1-score: %.3f" % (thresh, select_X_train.shape[1], f1))
    print("Confusion mat:\n", confusion_matrix(y_val, y_pred))
# %%
