# File: extract_features.py
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Date: 11/21/2022
# Description: From the .nii.gz files in the data/generated folder, this script
# extracts radiomic features of interest from CT scans & their segmentations
# using the pyradiomics package. NOTE: Cases R01-059 and R01-119 are weird...

# %%0. Package imports 
import os
from glob import glob
import numpy as np
import pandas as pd
from radiomics import featureextractor
import radiomics
import logging
from datetime import datetime
from utils import print_prog_bar, set_wd, str_dict

# 1. Set working directory to ece475/ if that isn't it already
print("Running extract_features.py...")
set_wd()

# 2. Read relevant .csv files
data_dir = 'data/generated/'
df = pd.read_csv(data_dir + 'NSCLC_labels.csv')

# %%3. Set up MY logging file
log_path = data_dir + 'feature_logs/NSCLC_features_README.txt'
output_path = data_dir + 'NSCLC_features.csv'
print(f"Logging output to {log_path}")
log_file = open(log_path, 'w')
log_file.write(("-"*80) + "\n\nREADME / log file for extract_features.py run from " + datetime.now().strftime('%m-%d-%Y %H:%M:%S')) 
log_file.write("\nSee " + output_path + "for resulting features.\n\n" + ("-"*80) + "\n\n")

# %%4. Set up radiomics' logging file
radiomics.setVerbosity(level=60)
rad_log_path = data_dir + 'feature_logs/radiomics_log.txt'
handler = logging.FileHandler(rad_log_path, 'w')
formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
handler.setFormatter(formatter)
radiomics.logger.addHandler(handler)

# %%5. Setting up pyradiomics feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.enableAllImageTypes()
log_file.write("Pyradiomics extractor used:\n\n")
log_file.write("Settings:\n" + str_dict(extractor.settings) + "\n")
log_file.write("Enabled features:\n" + str_dict(extractor.enabledFeatures) + "\n")
log_file.write("Enabled image types (filters):\n" + str_dict(extractor.enabledImagetypes) + "\n")
log_file.write(("-"*80) + "\n\n")

# 6. Get studies for feature extraction
studies = df['Case ID'].values
normal_studies = [len(glob(data_dir + 'NIfTI/' + s + '/*.nii.gz')) == 2 for s in studies]
weird_studies = [not s for s in normal_studies]
log_file.write(f"Dropped {studies[weird_studies]} from dataset, unexpected num of .nii.gz files.\n\n")
studies = studies[normal_studies]
n = len(studies)

# 7. Init record keeping
log_file.write(f"Status report for each of the {n} studies in the dataset:\n\n")
print(f"Extracting features from {n} studies...")
print_prog_bar(0, n, prefix='Progress:', suffix='processed study 0/%d, ETA: %s' % (n, 'N/A'), length=25)
t0 = datetime.now()

# 8. Extract features!
for i, study in enumerate(studies):

    paths = glob(data_dir + 'NIfTI/' + study + '/*.nii.gz')
    sizes = [os.path.getsize(p) for p in paths]
    ct_path = paths[np.argmax(sizes)]
    seg_path = paths[np.argmin(sizes)]

    try:
        result = extractor.execute(ct_path, seg_path)
        result_trimmed = result.copy()
        for k in result.keys():
            if not isinstance(result[k], np.ndarray):
                del result_trimmed[k]
            elif result[k].size > 1:
                del result_trimmed[k]
        
        features_row = pd.concat([pd.DataFrame({'Case ID': [study]}), pd.Series(result_trimmed).to_frame().T], axis=1)
        if i == 0:
            features_row.to_csv(output_path, index=False)
        else:
            features_row.to_csv(output_path, mode='a', index=False, header=False)
        
        log_file.write(f"{study} status: Success!\n")
    except Exception as e:
        error_message = str(type(e)) + ' ' + str(e)
        log_file.write(f"{study} status: Error, {error_message}\n")
        
    eta = (n - i - 1)*(datetime.now() - t0)/(i + 1)
    print_prog_bar(i + 1, n, prefix='Progress:', suffix='processed study %d/%d, ETA: %s' % (i + 1, n, str(eta)), length=25)

log_file.write("\n" + (80*"-") + "\n\nOutput saved to " + output_path + "\nEnd of README / log file, " + datetime.now().strftime('%m-%d-%Y %H:%M:%S') + "\n\n" + (80*"-"))
log_file.close()