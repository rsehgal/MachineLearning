import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from utilities import *
import sys

def GetColMap(data):
    col=[]
    for val in data['mat']:
        if(val==1.0):
            col.append('red')
        if(val==2.0):
            col.append('green')
        if(val==3.0):
            col.append('blue')
        if(val==4.0):
            col.append('magenta')
    return col
    

colNames=['x','y','z','mat']

rawFileName=sys.argv[1]
filteredFileName=sys.argv[2]

#Reading Raw data
data=GetData(rawFileName,colNames) #pd.read_csv(filename,delimiter=' ',names=colNames)
print(data.head())
x,y,z=data['x'],data['y'],data['z']
#Plotting
col=GetColMap(data)
plt.subplot(221)
plt.scatter(x,y,s=0.5,c=col)


#Reading filtered data
data=GetData(filteredFileName,colNames) #pd.read_csv(filename,delimiter=' ',names=colNames)
x,y,z=data['x'],data['y'],data['z']
#Plotting
col=GetColMap(data)
plt.subplot(222)
plt.scatter(x,y,s=0.5,c=col)

'''
#cmap = ListedColormap(["darkorange", "gold", "lawngreen", "lightseagreen"])
col=[]
for val in data['mat']:
    if(val==1.0):
        col.append('red')
    if(val==2.0):
        col.append('green')
    if(val==3.0):
        col.append('blue')
    if(val==4.0):
        col.append('magenta')
'''    
'''
plt.subplot(222)
plt.scatter(y,z,s=1,c='b')
plt.subplot(223)
plt.scatter(x,z,s=1,c='r')
'''
plt.show()
