import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale

#image = imread(data_dir + "/phantom.png", as_gray=True)
image = imread("raman.png", as_gray=True)
image = rescale(image, scale=0.4, mode='reflect', multichannel=False)

#fig, (ax1, ax2,ax3,ax4) = plt.subplots(2, 2, figsize=(8, 4.5))
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

ax1.set_title("Original")
ax1.imshow(image, cmap=plt.cm.Greys_r)

theta = np.linspace(0., 180., max(image.shape), endpoint=False)
print(len(theta))
print(theta.dtype)
print(theta)
sinogram = radon(image, theta=theta, circle=True)
ax2.set_title("Radon transform\n(Sinogram)")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

fig.tight_layout()


from skimage.transform import iradon

reconstruction_fbp = iradon(sinogram, theta=theta, circle=True)
#error = reconstruction_fbp - image
#print('FBP rms reconstruction error: %.3g' % np.sqrt(np.mean(error**2)))

imkwargs = dict(vmin=-0.2, vmax=0.2)
ax3.set_title("Reconstruction\nFiltered back projection")
ax3.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
#plt.show()


plt.show()
