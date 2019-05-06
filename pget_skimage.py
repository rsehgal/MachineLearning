from readCTData import *
import matplotlib.pyplot as plt

theta=splitTheta(360,360)
sinogram=readSimulatedSinogram('raman.png',theta)
recons=reconstructImage(sinogram,theta)
plotSinogramAndReconstrutedImg(sinogram,recons)

sinogram=readIAEAData("training00_E0.txt")
theta=splitTheta(360,360)
recons=reconstructImage(sinogram,theta)
plotSinogramAndReconstrutedImg(sinogram,recons)
