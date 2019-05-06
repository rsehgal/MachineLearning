from readCTData import *
import matplotlib.pyplot as plt
import skimage

theta=splitTheta(360,360)
#sinogram=readSimulatedSinogram('raman.png',theta)
#recons=reconstructImage(sinogram,theta)
#plotSinogramAndReconstrutedImg(sinogram,recons)

#sinogram=readIAEAData("training00_E0.txt")
sinogram=readIAEAData("sample00_E0.txt")
theta=splitTheta(360,360)
recons=reconstructImage(sinogram,theta,filter='hamming',inter='cubic')
#plotSinogramAndReconstrutedImg(sinogram,recons)

#Plotting all
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)


recons=reconstructImage(sinogram,theta,filter='ramp',inter='cubic')
ax1.set_title("ramp")
ax1.imshow(recons, cmap=plt.cm.Greys_r)

recons=reconstructImage(sinogram,theta,filter='hann',inter='cubic')
ax2.set_title("hann")
ax2.imshow(recons, cmap=plt.cm.Greys_r)

recons=reconstructImage(sinogram,theta,filter='hamming',inter='cubic')
ax3.set_title("hamming")
ax3.imshow(recons, cmap=plt.cm.Greys_r)

recons=reconstructImage(sinogram,theta,filter='cosine',inter='cubic')
ax4.set_title("cosine")
ax4.imshow(recons, cmap=plt.cm.Greys_r)

plt.show()

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.dpi'] = 200
recons=reconstructImage(sinogram,theta,filter='hann',inter='cubic')
sobel = skimage.filters.sobel(recons)
plt.imshow(sobel)
plt.show()

#blurrin is of no use
#blurred = skimage.filters.gaussian(sobel, sigma=2.0)
#plt.imshow(blurred)
#plt.show()
#print("-------------------------------------")
#thresholdMinimum(recons)
