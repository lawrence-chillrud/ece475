import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas

# Colums that will be used for x, y
use_cols_x = (660, 634, 231, 637, 785, 1408, 961, 105, 1132, 853, 610, 1209, 58, 576,
 1254, 752, 1643, 1160, 85, 783)
use_cols_y = 24

# import data
x_raw_data = pandas.io.parsers.read_csv("../data/generated/NSCLC_features.csv").values
x_data = x_raw_data[:,1:]
x_data = x_data[:,use_cols_x]
x = x_data.T

y_raw_data = pandas.io.parsers.read_csv("../data/generated/NSCLC_labels.csv").values
y_data = y_raw_data[:,use_cols_y]
y = y_data.T

#values that are not in x
y=np.delete(y,118-1)
y=np.delete(y,59-2)

inds = np.where((y=="Wildtype") | (y == "Mutant"))
x = x[:,inds[0]]
y = y[inds]

y[y=="Wildtype"]=-1
y[y=="Mutant"]=1

x = x.T
y = y.astype('int')
print(x)
print(y)

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# create a LogisticRegression object
model = LogisticRegression()

# train the model using the training data
model.fit(X_train, y_train)

# make predictions on the test data
y_pred = model.predict(X_test)

# evaluate the model performance
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % acc)
