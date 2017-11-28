#title              :kParameter.py
#description        :Creates dictionary from data and tests rank based on k
#author             :Brandon Campbell
#date               :11/28/2017
#usage              :python kParameter.py
#notes              :Make sure data is in the shape (num_feat x sample)
#python_version     :Anaconda 2.7
#===========================================================
from sklearn import preprocessing
import spams
import nibabel
import numpy as np
import time
##########################################################
# Load Data, Standardize, and Set parameters
##########################################################

data = np.load('../Data/fmri_data.npy')
scaler = preprocessing.StandardScaler()
data = np.transpose(scaler.fit_transform(data))
print('Data shape:', np.shape(data))

param = { 'lambda1' : .15,
          'numThreads' : 1,
          'batchsize' : 400,
          'iter' : 100 }
##########################################################
# Create dictionaries and determine ranks
##########################################################
k = range(1000,0, -50)
print(k)

f = open('outputs.txt', 'w')
for i in range(len(k)):
    D = spams.trainDL(data,K=k[i], **param) # Train Dictionary
    Q, R = np.linalg.qr(D) # Run QR factorization on Dictionary

    diag_R = np.diagonal(R)
    d_i = np.linalg.norm(diag_R) # Magnitude of diagonal of R matrix
    r_i = d_i / (d_i + 1)
    r_p = np.max(diag_R) # Max value in diag_R
    tau = (k[i] - 1)*r_p / (np.sum(diag_R) - r_p)

    f.write("K: %d\n" % k[i])
    f.write("-----------------\n")
    f.write("d_i: %f\n" % d_i)
    f.write("r_i: %f\n" % r_i)
    f.write("tau: %f\n" % tau)
    f.write("\n")
    f.write("\n")
f.close()
