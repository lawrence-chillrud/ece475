# File: 3_visualize_features.py
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Date: 11/25/2022
# Description: visualizes raw features extracted from 2_extract_features.py,
# so we can decide how best to scale the raw features for working w/them.
# Note: It's assumed this script is run from somewhere within the ece475/ dir.

# %%0. Package imports
import numpy as np
from utils import prep_data
import matplotlib.pyplot as plt

# %%1. Plot demographics data
X, y = prep_data(scale='None', include_demographics=True, split=False)
demographics = X.filter(['Age at Histological Diagnosis', 'Weight (lbs)', 'Gender', 'Ethnicity', 'Smoking status', 'Pack Years'])
dems = demographics.columns
demographics['EGFR mutation status'] = y
demographics.replace(np.nan, -88).hist(grid=False, figsize=[20,10])    
plt.tight_layout()
plt.show()

# %%2. Plot smattering of raw features
features = X.drop(dems, axis=1)
features.hist(grid=False, figsize=[20,10], column=features.columns[51:100])
plt.tight_layout()
plt.show()

# %%3. Get the train/val & testing splits by class
X_train_val, X_test, y_train_val, y_test = prep_data()
train_val_classes = y_train_val.value_counts()
test_classes = y_test.value_counts()
print('\nTRAINING/VALIDATION DATA CLASS SPLIT:\n', train_val_classes)
print('\nTESTING DATA CLASS SPLIT:\n', test_classes)
# %%
