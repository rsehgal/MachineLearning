import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import iradon
from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale
import pandas as pd
from skimage.filters import try_all_threshold
from skimage.filters import threshold_minimum
from skimage.filters import threshold_mean

def readSimulatedSinogram(filename,theta):
    print("======== Input image Dim =========")
    image = imread(filename, as_gray=True)
    print(image.shape)
    sinogram = radon(image, theta=theta, circle=True)
    sinogram=np.array(sinogram)
    print("======= Simulated Data Dim =========")
    print(sinogram.shape)
    return sinogram

def readIAEAData(filename):
    print("========= IAEA Data ========")
    dataseries=pd.read_csv(filename, delimiter=',')
    numOfDet = int(dataseries.sindata[0])
    numOfAng = int(dataseries.sindata[1])
    center= int(dataseries.sindata[2])

    supList=[]
    for i in range(numOfAng):
	start=3+(numOfDet*i)
	end=start+numOfDet
	subList=dataseries.sindata[start:end]
	supList.append(subList)
	
    sinogram=np.array(supList)
    print(sinogram.shape)
    return np.transpose(sinogram)



def splitTheta(dtheta,numOfDivisions):
    theta = np.linspace(0., dtheta, numOfDivisions, endpoint=False)
    return theta

def reconstructImage(sinogram,theta,filter='ramp',inter='linear'):
    reconstructed_fbp = iradon(sinogram, theta=theta, circle=False,filter=filter,interpolation=inter)
    return reconstructed_fbp 

def plotSinogramAndReconstrutedImg(sinogram,reconsImage):
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    ax1.set_title("Radon transform\n(Sinogram)")
    ax1.set_xlabel("Projection angle (deg)")
    ax1.set_ylabel("Projection position (pixels)")
    ax1.imshow(sinogram, cmap=plt.cm.Greys_r,extent=(0, 360, 0, 360), aspect='auto')

    ax2.set_title("Reconstruction\nFiltered back projection")
    ax2.imshow(reconsImage, cmap=plt.cm.Greys_r)

    plt.show()

def thresholdComparison(img):
    fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
    plt.show()

'''
def thresholdMinimum(img):
    reconsImage=threshold_minimum(img)
    thresh = threshold_mean(img)
    binary = reconsImage > thresh
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    ax.imshow(binary, cmap=plt.cm.gray)
    plt.show()
'''    
