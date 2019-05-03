import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale
import numpy as np
import pandas as pd

dataseries=pd.read_csv('training00_E1.txt', delimiter=',')
#dataseries=pd.read_csv('psf000mm_E0.txt', delimiter=',')
#dataseries=pd.read_csv('testRaman.csv', delimiter=',')
print(len(dataseries))
#print(dataseries.sindata[0])
#print(dataseries.sindata[65522])

#f=open('training00_E00.csv')
#lines=f.readlines()
#print(len(lines))
#
numOfDet = int(dataseries.sindata[0])
numOfAng = int(dataseries.sindata[1])
tmp= int(dataseries.sindata[2])
#
supList=[]
for i in range(numOfDet):
    start=3+(numOfAng*i)
    end=start+numOfAng
    subList=dataseries.sindata[start:end]
    #supList.append(np.array([subList]))
    supList.append(subList)

print("===== SupList =====")
sinogram=np.array(supList)
sinogram=sinogram.reshape(numOfDet,numOfAng)
print(sinogram.dtype)
print(sinogram.shape)
print(sinogram)

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.set_title("Original")

ax1.imshow(sinogram, cmap=plt.cm.Greys_r,
           #extent=(0, 180, 0, 180), aspect='auto')
           extent=(0, 360, 0, sinogram.shape[1]), aspect='auto')


#inverse radon transform
from skimage.transform import iradon
#theta=range(360)
theta = np.linspace(0., numOfAng/2., numOfAng, endpoint=False)
#theta = np.linspace(0., 180., 380, endpoint=False)

reconstruction_fbp = iradon(sinogram, theta=theta) #, circle=True)
#plots
ax2 = fig.add_subplot(2,2,2)

#imkwargs = dict(vmin=-0.2, vmax=0.2)
ax2.set_title("Reconstruction\nFiltered back projection")
ax2.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)


from skimage.transform import iradon_sart
reconstruction_sart = iradon_sart(sinogram, theta=theta)
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(reconstruction_sart, cmap=plt.cm.Greys_r)
for i in range(10):
    print("Iteration : ")
    print(i+1)
    reconstruction_sart = iradon_sart(sinogram, theta=theta,
                                   image=reconstruction_sart)
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(reconstruction_sart, cmap=plt.cm.Greys_r)

plt.show()

