# File: fix_dcm_header.py
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Date: 11/22/2022
# Description: This file contains a sample workflow for fixing some corrupted value
# in a DICOM file header

# %%
import os
from pydicom import dcmread

study = 'R01-008'
f = f"data/TCIA/manifest/NSCLC Radiogenomics/{study}/04-24-1991-NA-CT CHEST WO CONTRAST-33112/1000.000000-3D Slicer segmentation result-47965/1-1.dcm"
dcm = dcmread(f)
dcm.NumberOfFrames = 119

# %%
out_dir = 'data/generated/fixed_DICOMS/'
suffix = '_SEG'
f_fixed = f"{out_dir}{study}{suffix}.dcm"
dcm.save_as(f_fixed, write_like_original=False)

# %%
convert_cmd = f"dcm2niix -z y -9 -v 2 -o {out_dir} -f %d {f_fixed}"
os.system(convert_cmd)
