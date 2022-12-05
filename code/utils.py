# File: utils.py
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Date: 11/21/2022
# Description: Contains some helper fuctions.
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def print_prog_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    if iteration == total: 
        print()

def set_wd():
    cwd = os.getcwd().split('/')
    try:
        while cwd.pop() != 'ece475':
            os.chdir('..')
    except IndexError: 
        print("Error: This script must be run from somewhere inside the ece475/ dir.")
        exit()
    print("Set working directory to:", os.getcwd())

def str_dict(d):
    s = ''
    for k in d.keys():
        s += '\t' + str(k) + ': ' + str(d[k]) + '\n'
    return s

def prep_data(data_dir='data/generated/', outcome_of_interest='EGFR mutation status', scale='Normalize'):
    if scale not in ['None', 'Normalize', 'Standardize']:
        raise Exception("Unknown scale method selected. Must be one of {'None', 'Normalize', 'Standardize'}")
    set_wd()
    dff = pd.read_csv(data_dir + 'NSCLC_features.csv')
    dfl = pd.read_csv(data_dir + 'NSCLC_labels.csv').filter(['Case ID', 'Age at Histological Diagnosis', 'Weight (lbs)', 'Gender', 'Ethnicity', 'Smoking status', 'Pack Years', outcome_of_interest])
    dfl = dfl[dfl[outcome_of_interest] != 'Unknown']
    dfl['Weight (lbs)'] = dfl['Weight (lbs)'].replace('Not Collected', float('nan'))
    dfl['Pack Years'] = dfl['Pack Years'].replace('Not Collected', float('nan'))
    dfl['Pack Years'] = dfl['Pack Years'].mask(dfl['Smoking status'] == 'Nonsmoker', 0)
    dfl['Gender'] = dfl['Gender'].replace(['Male', 'Female'], [0, 1])
    dfl['Ethnicity'] = dfl['Ethnicity'].replace(['Caucasian', 'Native Hawaiian/Pacific Islander', 'African-American', 'Asian', 'Hispanic/Latino'], [0, 1, 2, 3, 4])
    dfl[outcome_of_interest] = dfl[outcome_of_interest].replace(['Mutant', 'Wildtype'], [1, -1])
    dfl['Smoking status'] = dfl['Smoking status'].replace(['Nonsmoker', 'Former', 'Current'], [0, 1, 2])
    df = pd.merge(dfl, dff, on='Case ID')
    X = df.drop(['Case ID', outcome_of_interest], axis=1)
    y = df[outcome_of_interest]
    if scale == 'Normalize':
        X = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)
    elif scale == 'Standardize':
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    
    return X, y