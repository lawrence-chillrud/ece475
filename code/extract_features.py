# File: extract_features.py
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Date: 11/21/2022
# Description: From the .nii.gz files in the data/generated folder, this script
# extracts radiomic features of interest from CT scans & their segmentations
# using the pyradiomics package

# %%0. Package imports 
import os
import pandas as pd
from radiomics import featureextractor
from datetime import datetime
from utils import print_prog_bar, set_wd

# 1. Set working directory to ece475/ if that isn't it already
print("Running extract_features.py...")
set_wd()

# 2. Read relevant .csv files
data_dir = 'data/generated/'
df = pd.read_csv(data_dir + 'NSCLC_labels.csv')
