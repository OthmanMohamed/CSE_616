import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from matplotlib import pyplot as plt


def forward(samples, w1, w2):
    z1_cache = []
    z1_a_cache = []
    z2_cache = []
    for x in samples:
        z1 = np.dot(x, w1)
        z1_cache.append(z1)
        z1_activation = np.maximum(0, z1)
        z1_a_cache.append(z1_activation)
        z2 = np.dot(z1_activation, w2)
        z2_cache.append(z2)
    return np.array(z1_cache), np.array(z1_a_cache), np.array(z2_cache)

def mean_squared_error(pred, truth):
    truth = np.reshape(truth, (len(truth),1))
    error = pred-truth
    squared_error = error ** 2
    mean_sq_err = (1 / len(squared_error)) * np.sum(squared_error)
    return mean_sq_err

def grad(X, labels, z1_cache, z1_a_cache, z2_cache, w1, w2):
    dw1 = np.zeros((12,30))
    dw2 = np.zeros((30,1))
    for i in range(len(X)):
        y = labels[i]
        x = X[i]
        z1 = z1_cache[i]
        z1_activation = z1_a_cache[i]
        z2 = z2_cache[i]
        
        dz2 = 2*(z2 - y)
        z1_activation = np.reshape(z1_activation, (z1_activation.shape[0], 1))
        dw2 += dz2 * z1_activation
        dw2 = np.reshape(dw2, (len(dw2),1))
        dz1_activation = dz2 * w2
        dz1 = np.reshape(np.greater(z1_activation, 0), (30,1)) * dz1_activation
        dw1 += np.transpose(dz1 * x)
    return dw1, dw2

def back_prop(w1, w2, dw1, dw2, lr):
    w1 = w1 - lr*dw1
    w2 = w2 - lr*dw2
    return w1, w2

wine_ds = pd.read_csv("C:/Users/Othman/Downloads/winequality-red.csv", sep=";")
wine_ds.insert(11, 'bias', 1)
wine_ds.head(n=20)

wine_ds = wine_ds.sample(frac=1).reset_index(drop=True)
train_ds = wine_ds.iloc[0:800]
test_ds = wine_ds.iloc[800:]

mean = train_ds.mean(axis=0)
std = train_ds.std(axis=0)
print("MEAN : ", mean)
print("STD : ", std)

for column in train_ds:
    if column != 'bias':
        train_ds[column] = (train_ds[column] - mean[column]) / std[column]
print("NEW STD : ", train_ds.std(axis=0))
print("NEW MEAN : ", train_ds.mean(axis=0))
train_ds_np = train_ds.to_numpy()
train_ds_np.shape
features = train_ds_np[:, :12]
lbls = train_ds_np[:, 12]

for column in test_ds:
    if column != 'bias':
        test_ds[column] = (test_ds[column] - mean[column]) / std[column]
test_ds_np = test_ds.to_numpy()
features_test = test_ds_np[:, :12]
lbls_test = test_ds_np[:, 12]

w1 = np.random.rand(12,30)
w2 = np.random.rand(30,1)

err_1e5 = []
lr = 1e-5
w1_1e5 = w1
w2_1e5 = w2

for i in range(1000):
    f = features
    z1_cache, z1_a_cache, z2_cache = forward(f, w1_1e5, w2_1e5)
    error = mean_squared_error(z2_cache, np.reshape(lbls, (800,1)))
    print("I : ", i, "\t ERROR = ", error)
    err_1e5.append(error)
    dw1, dw2 = grad(f, lbls, z1_cache, z1_a_cache, z2_cache, w1_1e5, w2_1e5)
    w1_1e5, w2_1e5 = back_prop(w1_1e5, w2_1e5, dw1, dw2, lr)

plt.xlabel('iteration')
plt.ylabel('LOSS')
plt.plot(err_1e5[15:])
plt.savefig('err_1e5.png')

_,_, z2_cache = forward(features_test, w1_1e5, w2_1e5)
error_test_1e5 = mean_squared_error(z2_cache, np.reshape(lbls_test, (799,1)))
print("TEST ERROR : ", error_test_1e5)
