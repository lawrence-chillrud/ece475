import pandas as pd
import sklearn.feature_selection as fs
from utils import prep_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error

def get_data():
    # Get prepared data from utils.py:
    X_train_val, X_test, y_train_val, y_test = prep_data(labels=[1, 0])
    # Split training and validation data:
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=0.25,
                                                      random_state=0,
                                                      stratify=y_train_val)
    return X_train, X_val, X_test, y_train, y_val, y_test


def select_features(X_train, y_train, X_test, num_features):
    '''
    Selects the best num_features features based on a gradient-boosted decision
    tree (GBDT) model. "Gradient-boosted" practically refers to the fact that we
    will have an additional tree that can predict the errors (differences
    between the predictions and ground-truth data) made by the initial tree.

    Inputs:
    - X_train
    - y_train
    - X_test
    - num_features: how many features we want to keep

    Output:
    - new_X_train

    '''

    # Create and fit a GBDT object:
    gbc = GradientBoostingClassifier(learning_rate=0.1,
                                     n_estimators=200,
                                     max_depth=3)
    gbc.fit(X_train, y_train)

    # TODO: Optimize GBDT model, time permitting.

    features = X_train.columns
    feature_importances = gbc.feature_importances_
    features_with_importances = dict(zip(features, feature_importances))
    sorted_features_with_importances = sorted(features_with_importances.items(),
                                           key=lambda x:x[1],
                                           reverse=True)
    best_features = dict(sorted_features_with_importances).keys()[:num_features]

    print(f'''The most important {num_features} features, sorted by importance,
          are: {best_features}''')

    new_X_train = X_train[best_features]

    return new_X_train
