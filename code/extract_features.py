# File: extract_features.py
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Date: 11/21/2022
# Description: From the .nii.gz files in the data/generated folder, this script
# extracts radiomic features of interest from CT scans & their segmentations
# using the pyradiomics package. NOTE: Cases R01-059 and R01-119 are weird...

# %%0. Package imports 
from glob import glob
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

# 3. Extract features from each study in cohort
extractor = featureextractor.RadiomicsFeatureExtractor()
weird_studies = ['R01-059', 'R01-119'] # has 10, 3 .nii.gz files respectively
studies = df['Case ID']
n = len(studies)
print(f"Extracting features from {n} studies...")
print_prog_bar(0, n, prefix='Progress:', suffix='processed study 0/%d, ETA: %s' % (n, 'N/A'), length=25)
t0 = datetime.now()
for i, study in enumerate(studies):
    if study not in weird_studies:
        files = glob(data_dir + 'NIfTI/' + study + '/*.nii.gz')
        result = extractor.execute(files[0], files[1])
        eta = (n - i - 1)*(datetime.now() - t0)/(i + 1)
    print_prog_bar(i + 1, n, prefix='Progress:', suffix='processed study %d/%d, ETA: %s' % (i + 1, n, str(eta)), length=25)
# %%
