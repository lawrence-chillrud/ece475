# ece475
Directory for EE475 final project.

**Radiogenomic machine learning (ML) for EGFR mutation status prediction from Computed Tomography (CT) scans of patients with Non-Small Cell Lung Cancer (NSCLC).**

Table of contents:
1. [Requirements](#1-requirements)
2. [The data](#2-the-data)
3. [Downloading the data](#3-downloading-the-data)
4. [Converting the data](#4-converting-the-data)
5. [Feature extraction](#5-feature-extraction)
6. [References](#references)

## 1. Requirements

We also provide the `environment.yml` YAML file for easy cloning of the exact conda virtual environment used to run all project scripts. With [conda](https://conda.io/projects/conda/en/latest/user-guide/install/download.html) installed, simply run the following line in `ece475` directory:

```conda env create --file environment.yml```

Then activate the newly created `rgml` environment with:

```conda activate rgml```

You should now be able to run all scripts in the `code` folder without issue.

## 2. The data

The dataset we are using is The Cancer Imaging Archive's (TCIA)[^1] publicly available [NSCLC Radiogenomics dataset](https://wiki.cancerimagingarchive.net/display/Public/NSCLC+Radiogenomics), made from a cohort of 211 NSCLC patients. A brief description from the dataset's TCIA page is reproduced below:

> "The dataset comprises Computed Tomography (CT) images, ... and segmentation maps of tumors in the CT scans. Imaging data are also paired with results of gene mutation analyses... from samples of surgically excised tumor tissue, and clinical data, including survival outcomes. This dataset was created to facilitate the discovery of the underlying relationship between tumor molecular and medical image features, as well as the development and evaluation of prognostic medical image biomarkers."

We refer interested readers to Bakr et al.[^2] for further details regarding the dataset.

## 3. Downloading the data
1. Download [the .tcia manifest file](https://wiki.cancerimagingarchive.net/download/attachments/28672347/NSCLC_Radiogenomics-6-1-21%20Version%204.tcia?version=1&modificationDate=1622561925765&api=v2) (88 KB) that will be needed to later retrieve the images and their segmentations from the NCBI Data Retriever (see steps 3 and 4).
2. Download [the .csv file](https://wiki.cancerimagingarchive.net/download/attachments/28672347/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv?version=1&modificationDate=1531967714295&api=v2) (67 KB) providing the clinical data obtained for each patient (including each patient's EGFR mutation status).
3. Follow [these instructions](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images#DownloadingTCIAImages-InstallingtheNBIADataRetriever) to install the NCBI Data Retriever application.
4. Now open the `.tcia` manifest file downloaded in step 1 (this will open the NCBI app installed in step 3). Follow the app's prompts to download all the images and their segmentations (97.6 GB total). They will be downloaded in `.dcm` DICOM format. Detailed instructions can be found [here](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images#DownloadingTCIAImages-OpeningtheManifestFileandDownloadingtheData).

## 4. Converting the data
In order to make the data smaller and easier to work with, we must convert all downloaded CT scans and their accompanying segmentations from their raw DICOM `.dcm` format to a compressed NIfTI `.nii.gz` format. Interested readers can study [this useful DICOM and NIfTI Primer post](https://github.com/DataCurationNetwork/data-primers/blob/master/Neuroimaging%20DICOM%20and%20NIfTI%20Data%20Curation%20Primer/neuroimaging-dicom-and-nifti-data-curation-primer.md) for the differences between the two file types.

There are many tools for converting `.dcm` -> `.nii.gz`; the software we choose to employ is `dcm2niix`[^3]. Documentation for `dcm2niix` can be found [here](https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage).

<!---
1. Install `dcm2niix` by following [these installation instructions](https://github.com/rordenlab/dcm2niix#Install). _Note: for faster compression, also download the soft (ie optional) dependency [pigz](https://github.com/madler/pigz). This can be done in a conda environment with:_ 

    ```conda install -c conda-forge pigz```

2. 
-->

We convert all CT scans and segmentations to `.nii.gz` files by running the `convert_dcms.py` script, which utilizes `dcm2niix` under the hood.

## 5. Feature extraction

Now that we have downloaded the data and converted it into the appropriate format, we can extract features from the CT scans and their segmentations using the `pyradiomics` python package[^4]. Documentation for `pyradiomics` is extensive and can be found [here](https://pyradiomics.readthedocs.io/en/latest/index.html#).

<!---
1. Install `pyradiomics` with the following _(Note that installation via conda had unresolved bugs at the time of writing)_:

    ```pip install pyradiomics```
-->

We extract radiomic features from the `.nii.gz` files by running the `extract_features.py` script. This should output the file `NSCLC_features.csv`, which can be used for downstream ML tasks.

## References

[^1]: Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. [The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository](https://doi.org/10.1007/s10278-013-9622-7). _Journal of Digital Imaging_, Volume 26, Number 6, December, 2013, pp 1045-1057.  

[^2]: Bakr S, Gevaert O, Echegaray S, Ayers K, Zhou M, Shafiq M, Zheng H, Benson JA, Zhang W, Leung ANC, Kadoch M, Hoang CD, Shrager J, Quon A, Rubin DL, Plevritis SK, Napel S. [A radiogenomic dataset of non-small cell lung cancer.](https://pubmed.ncbi.nlm.nih.gov/30325352/) _Sci Data_. 2018 Oct 16;5:180202.

[^3]: Li X, Morgan PS, Ashburner J, Smith J, Rorden C. [The first step for neuroimaging data analysis: DICOM to NIfTI conversion.](https://pubmed.ncbi.nlm.nih.gov/26945974/) _J Neurosci Methods_. 2016;264:47-56.

[^4]: Van Griethuysen JJ, Fedorov A, Parmar C, Hosny A, Aucoin N, Narayan V, Beets-Tan RG, Fillion-Robin JC, Pieper S and Aerts HJ. [Computational radiomics system to decode the radiographic phenotype.](https://aacrjournals.org/cancerres/article/77/21/e104/662617) _Cancer research_, 2017, 77(21), pp.e104-e107.