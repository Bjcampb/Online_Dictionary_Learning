import spams
import numpy as np
from sklearn.preprocessing import StandardScaler

#-----------------------------------------------
# Load Data and Scale
#-----------------------------------------------
data = np.load('../Data/fmri_data.npy')
data = data.astype(float)

scaler = StandardScaler()
data = np.transpose(scaler.fit_transform(data))

#-----------------------------------------------
# Create dictionary
#-----------------------------------------------
param = {'K' : 300,
	 'lambda1' : 1.0,
	 'numThreads' : 4,
	 'batchsize' : 400,
	 'modeParam' : 0,
	 'gamma1' : 0.3,
	 'modeD' : 1,
	 'iter' : 1000 }

D = spams.trainDL(data, **param)
alpha = spams.omp(data, D, L=10)
alpha = alpha.todense()

np.save('../Data/dictionary_300_atoms', D)
np.save('../Data/alpha_300_atoms', alpha)
