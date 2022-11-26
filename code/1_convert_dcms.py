# File: 1_convert_dcms.py
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Date: 11/21/2022
# Description: this file converts both the CT and SEG .dcm (DICOM) files 
# downloaded from TCIA's Radiogenics dataset to .nii.gz (NIfTI) format.
# Note: It's assumed this script is run from somewhere within the ece475/ dir.
# It's also assumed the .dcm data is formatted as downloaded from TCIA.

# %%0. Package imports
import os
import pandas as pd
from datetime import datetime
from utils import print_prog_bar, set_wd

# 1. Set working directory to ece475/ if that isn't it already
print("Running convert_dcms.py...")
set_wd()

# 2. Read relevant .csv files
tcia_dir = 'data/TCIA/manifest/'
md = pd.read_csv(tcia_dir + 'metadata.csv')
df = pd.read_csv(tcia_dir + 'NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv')

# 3. Get list of directories with SEG files
segs_md = md.loc[md['Modality'] == 'SEG', ['Subject ID', 'Study Description', 'Manufacturer', 'File Location']]
segs_fs = segs_md['File Location']

# 4. Convert all .dcms with a valid segmentation...
log_file = 'data/generated/NIfTI/convert_dcms_log.txt'
os.system(f"echo '' > {log_file}")
n = len(segs_fs)
print(f"Logging output to {log_file}")
print(f"Converting {n} studies to NIfTI format...")
print_prog_bar(0, n, prefix='Progress:', suffix='converted study 0/%d, ETA: %s' % (n, 'N/A'), length=25)
t0 = datetime.now()
for i, f in enumerate(segs_fs):
    f_list = f.split('/')[1:-1]
    in_path = tcia_dir + '/'.join(f_list)
    in_path = in_path.replace(' ', '\\ ')
    out_dir = 'data/generated/NIfTI/' + f_list[1]
    os.makedirs(out_dir, exist_ok=True)
    log_cmd = f"echo '\nConverting {f_list[1]}' >> {log_file}"
    convert_cmd = f"dcm2niix -z y -9 -d -v 0 -o {out_dir} -f %d {in_path} >> {log_file}"
    os.system(log_cmd)
    os.system(convert_cmd)
    eta = (n - i - 1)*(datetime.now() - t0)/(i + 1)
    print_prog_bar(i + 1, n, prefix='Progress:', suffix='converted study %d/%d, ETA: %s' % (i + 1, n, str(eta)), length=25)

# 5. Save relevant labels and metadata
merged_df = pd.merge(df, segs_md, left_on='Case ID', right_on='Subject ID')
merged_df = merged_df.drop(columns=['Subject ID', 'File Location'])
merged_df.to_csv('data/generated/NSCLC_labels.csv', index=False)