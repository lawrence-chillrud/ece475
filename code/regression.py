import autograd.numpy as np
from autograd import grad, hessian
import random
import math
import matplotlib.pyplot as plt
import pandas
from matplotlib import ticker


# Colums that will be used for x
use_cols_x = range(1,6)
use_cols_y = 24

epsilon = 10 **(-2)
alpha = 5
itera = 1000
sampleSize = 200
lam = 10**(-5)
C=10
# import data
x_raw_data = pandas.io.parsers.read_csv("../data/generated/NSCLC_features.csv").values
x_data = x_raw_data[:,use_cols_x]
x = x_data.T
y_raw_data = pandas.io.parsers.read_csv("../data/generated/NSCLC_labels.csv").values
y_data = y_raw_data[:,use_cols_y]
y = y_data.T

#values that are not in x
y=np.delete(y,118-1)
y=np.delete(y,59-2)
print(y)

inds = np.where((y=="Wildtype") | (y == "Mutant"))
print(x.shape)
print(y.shape)
x = x[:,inds]
y = y[inds]

exit()
x = x.values
y = y.values

x = x.T
y = np.array([int(v) for v in y])[np.newaxis,:]


print("input shape = " , x.shape)
print("output shape = ", y.shape)

# sample indices
num_sample = 50000
inds = np.random.permutation(y.shape[1])[:num_sample]
x = x[:,inds]
y = y[:,inds]

size = np.size(y)

xmean = np.nanmean(x,axis = 1)[:,np.newaxis]
xstd = np.nanstd(x,axis = 1)[:,np.newaxis]   

idx = np.argwhere(xstd < 0.1)
if len(idx) > 0:
    idx = [v[0] for v in idx]
    offset = np.zeros((xstd.shape))
    offset[idx] = 1
    xstd += offset

undefineds = np.argwhere(np.isnan(x) == True)
for i in undefineds:
    x[i[0],i[1]] = xmean[i[0]]

for i in range(len(x)):
	for j in range(size):
		x[i][j] = (x[i][j]-xmean[i])/xstd[i]	

w = 0.1*np.random.randn(784+1,C)
def model(x,w):
	a = w[0] + np.dot(x.T,w[1:])
	return a.T

def multiclass_perceptron(w,samp):
	all_evals = model(x[:,samp], w)
	a = np.max( all_evals,axis = 0)
	b = all_evals [y[:,samp].astype(int).flatten(),np.arange(np.size(y[:,samp]))]
	cost = np.sum(a - b)
	cost = cost + lam*np.linalg.norm(w[1:,:],'fro')**2
	return cost / float(np.size(y))

gradient = grad(multiclass_perceptron)

gvals = [[],[]]

for k in range(itera):
	all_evals = model(x,w)
	y_predicted = np.argmax( all_evals,axis = 0)
	accracy = np.zeros(2)
	inds = np.random.permutation(y.shape[1])[:sampleSize]
	grad_eval = gradient(w,inds)
	w -= alpha*grad_eval
	for j in range(size):
		if(y[0,j] == y_predicted[j]):
			accracy[0]+=1
		else:
			accracy[1]+=1
	print(accracy)
	gvals[0] += [k]
	inds = np.random.permutation(y.shape[1])[:50000]
	gvals[1] += [multiclass_perceptron(w,inds)]
	#gvals[0] += [k]
	#gvals[1] += [accracy[1]]
#print((accracy[0]+accracy[3])/np.sum(accracy))

fig,ax=plt.subplots(1,1)
ax.set_title('MNIST Raw Pixels')
ax.set_xlabel('Steps')
ax.set_ylabel('g(w)')
plt.plot(gvals[0], gvals[1], "r-")
plt.show()
