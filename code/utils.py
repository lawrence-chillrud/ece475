# File: utils.py
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Date: 11/21/2022
# Description: Contains some helper fuctions.

import os

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