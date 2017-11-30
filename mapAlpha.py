import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import copy
from numpy import ma

#-------------------------------------------
# Load anatomy and alpha matrix
#-------------------------------------------
anatomy = nib.load('../Data/anatomy.nii').get_data()
mask = np.load('../Data/fmri_mask.npy')
x,y,z = mask.nonzero()

alpha = np.transpose(np.load('../Data/alpha_300_atoms.npy'))

#----------------------------------------------
# Map alpha values back to anatomy
#-----------------------------------------------
alpha_map = np.zeros((np.shape(mask)[0],
			np.shape(mask)[1],
			np.shape(mask)[2],
			np.shape(alpha)[1]))

count = 0
for i in range(len(x)):
	for j in range(np.shape(alpha)[1]):
		alpha_map[x[i], y[i], z[i], j] = alpha[count, j]
	count = count + 1
np.save('alpha_map_300_atoms.npy', alpha_map)

#----------------------------------------------
# Plot
#----------------------------------------------
if False:
	my_cmap = copy.copy(plt.cm.get_cmap('gnuplot'))
	my_cmap.set_bad(alpha=0.0)
	masked_map = ma.masked_where(alpha_map == 0, alpha_map)
	for i in range(np.shape(alpha_map)[3]):
		plt.figure(figsize=(10,10))
		for j in range(np.shape(alpha_map)[2]):
			plt.subplot(5,4,j+1)
			plt.imshow(anatomy[35:95,40:100,j], cmap=plt.cm.gray)
			plt.imshow(masked_map[35:95,40:100,j,i], alpha=0.7, cmap=my_cmap,
				vmin=np.min(alpha_map),vmax=np.max(alpha_map))
			plt.tick_params(axis='both',which='both',bottom='off',top='off',
				labelbottom='off',right='off',left='off',
				labelleft='off')
			plt.colorbar()
		plt.suptitle("Alpha map for Atom %d" % (i+1), fontsize=14)
		plt.savefig('../Results/300Atoms/Atom_%d.jpg' % (i+1))
		plt.close()






