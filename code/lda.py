import numpy as np
import pandas as pd
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from utils import prep_data

X_train_val, X_test, y_train_val, y_test = prep_data(labels=[1, -1])
features_df = pd.read_csv('data/generated/feature_selection_output/gbdt_features.csv')
features = feature_df['feature'].tolist()
X_tv = X_train_val[features]
X_t = X_test[features]

if 'lda.pkl' not in os.listdir('data/generated/models/'):
    model = LDA()

    params = {
                solver = ['svd', 'lsqr', 'eigen'],
                n_components = list(range(1, len(features)+1))
    }

    gs = GridSearchCV(
                        estimator=model,
                        param_grid=params,
                        scoring=scoring,
                        refit='AUC',
                        n_jobs=-1,
                        cv=cv,
                        return_train_score=True,
                        verbose=10
    )

    gs.fit(X_tv, y_train_val)

    results = gs.cv_results_
    keys = [k for k, v in results.items() if k.startswith('split')]
    for x in keys:
        del results[x]

    results_df = pd.DataFrame().from_dict(results)
    print("\nBEST PARAMS:\n")
    print(gs.best_params_)
    joblib.dump(gs.best_estimator_, 'data/generated/models/lda.pkl')
    results_df.to_csv('data/generated/models/lda_results_df_final.csv', index=False)



# TODO: Decide on model evaluation methods
# def evaluate_model(X, y, model):
#     # Define model evaluation method:
#     cv = RepeatedStratifiedKFold()
#     # Evaluate model:
#     scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
#     # Present result:
#     print(f'Mean accuracy is: {mean(scores)}.')
