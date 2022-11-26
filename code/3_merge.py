# File: 3_merge.py
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Date: 11/25/2022
# Description: merges the NSCLC features and labels .csv files.
# Note: It's assumed this script is run from somewhere within the ece475/ dir.

# %%0. Package imports
import pandas as pd
from utils import set_wd

# 1. Set up directories
set_wd()
data_dir = 'data/generated/'

# 2. Read in data
dff = pd.read_csv(data_dir + 'NSCLC_features.csv')
dfl = pd.read_csv(data_dir + 'NSCLC_labels.csv')

# 3. Merge data by Case ID
df = pd.merge(dff, dfl, on = 'Case ID')

# 4. Print summary mutation stats
print(df['EGFR mutation status'].value_counts(), '\n')
print(df['KRAS mutation status'].value_counts(), '\n')
print(df['ALK translocation status'].value_counts(), '\n')
print(df['Histopathological Grade'].value_counts(), '\n')