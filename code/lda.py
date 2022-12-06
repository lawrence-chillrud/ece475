import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import RepeatedStratifiedKFold


def create_model(X, y):
    '''
    Linear Discriminant Analysis (LDA) involves reducing dimensionality by
    projecting a dataset onto a lower-dimensional space by finding the area that
    maximizes the separation between classes. It maintains the information that
    discriminates between classes.

    It involves calculating the d-dimensional mean vectors for each class, then
    calculating the within-class scatter matrix, then calculating the between-class
    scatter matrix, then calculating the eigenvectors and eigenvalues for the
    scatter matrix, sorting those eigenvectors by eigenvalues and choosing the k
    eigenvectors that have the biggest eigenvalues, and finally reducing the
    dimensionality accordingly.

    Inputs:
    - X
    - y

    Output:
    - model
    '''

    model = LDA()
    model.fit(X, y)

    return model

# TODO: Decide on model evaluation methods
# def evaluate_model(X, y, model):
#     # Define model evaluation method:
#     cv = RepeatedStratifiedKFold()
#     # Evaluate model:
#     scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
#     # Present result:
#     print(f'Mean accuracy is: {mean(scores)}.')
