import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale
import numpy as np
import pandas as pd

dataseries=pd.read_csv('training09_E0.txt', delimiter=',')
#dataseries=pd.read_csv('sample00_E0.txt', delimiter=',')
print(len(dataseries))
numOfDet = int(dataseries.sindata[0])
numOfAng = int(dataseries.sindata[1])
center= dataseries.sindata[2]
print(center)
head1supList=[]
head2supList=[]
for i in range(numOfAng):
    start=3+(numOfDet*i)
    end=start+numOfDet
    head1List=dataseries.sindata[start:end:1]
    head2List=dataseries.sindata[start+1:end:2]
    head1supList.append([head1List])
    head2supList.append([head2List])

npDataArrayHead1=np.array(head1supList)
npDataArrayHead2=np.array(head2supList)

#print(npDataArray.shape)
#print(npDataArray[0].shape)
#print(npDataArray[0])

import tomopy
ang = tomopy.angles(180)
sim1=npDataArrayHead1
sim2=npDataArrayHead2
#sim1 = tomopy.minus_log(sim1)


rec1 = tomopy.recon(sim1,ang,center=center,  algorithm='bart',num_iter=200)
#rec1 = tomopy.recon(sim1,ang,num_gridx=200,num_gridy=200,center=center,  algorithm='bart',num_iter=100)

#rec1 = tomopy.circ_mask(rec1, axis=0, ratio=0.60)

#rec1 = tomopy.recon(sim1, ang, center=center,algorithm='sirt',num_iter=50)#,num_iter=10)
#rec1 = tomopy.recon(sim1, ang, algorithm=tomopy.astra, options={'method':'SART', 'num_iter':10*180, 'proj_type':'linear','extra_options':{'MinConstraint':0}})


rec2 = tomopy.recon(sim2, ang, algorithm='bart')

#import matplotlib.pyplot as plt
#plt.imshow(sim1[:, 0, :], cmap='Greys_r')
#plt.show()

print(rec1.shape)
import pylab

#pylab.imshow(rec1[0], cmap='gray')
#pylab.show()

plt.imshow(rec1[0, :,:], cmap='Greys_r')
plt.show()
