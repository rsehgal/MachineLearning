import numpy as np
from root_numpy import root2array, tree2array
from ROOT import TFile, TTree, TList
import matplotlib.pyplot as plt

def CreateData(fileList,filename):
    pathList = fileList #['test1.root', 'test2.root']
    signalTreeList = TList()
    backgroundTreeList=TList()
    outputFile = TFile(filename, 'recreate')
    pyfilelist = []
    pySignalTreelist = []
    pyBackgroundTreelist = []

    for path in pathList:
	    print("Path", path)
	    inputFile = TFile(path, 'read')
	    pyfilelist.append(inputFile) # Make this TFile survive the loop!
	    inputSignalTree = inputFile.Get('Signal')
	    inputBackgroundTree = inputFile.Get('Background')
	    pySignalTreelist.append(inputSignalTree) # Make this TTree survive the loop!
	    pyBackgroundTreelist.append(inputBackgroundTree) # Make this TTree survive the loop!
	    outputSignalTree = inputSignalTree.CloneTree() #instead of extensive processing
	    outputBackgroundTree = inputBackgroundTree.CloneTree() #instead of extensive processing
	    signalTreeList.Add(inputSignalTree)
	    backgroundTreeList.Add(inputBackgroundTree)

    outputFile.cd()
    outputSignalTree = TTree.MergeTrees(signalTreeList)
    outputBackgroundTree = TTree.MergeTrees(backgroundTreeList)
    outputFile.Write()
    outputFile.Close()


def getData(filename):
    arr=root2array(filename,"Signal")
    return arr

def getSignalData(filename):
    arr=root2array(filename,"Signal")
    return arr

def getBackgroundData(filename):
    arr=root2array(filename,"Background")
    return arr


def getSignal(filename):
    infile=TFile(filename)
    signal=infile.Signal
    '''
    count=0
    for e in signal:
        count=count+1
        print(count)
        print(e.deltaThetaX)
    '''
    return signal

def getBackground(filename):
    infile=TFile(filename)
    background=infile.Background
    return background
    
'''
def reshapeData2(dataArr):
	supList=[]
	for e in dataArr:
		subList=[]
		for val in e:
			subList.append(val)
		supList.append(subList)
	return np.array(supList)

'''
#Using same logic as above but using built in function .tolist()	
def reshapeData2(dataArr):
    print(type(dataArr[0]))
    print(dataArr[0].dtype)
    supList=[]
    for e in dataArr:
	    supList.append(e.tolist())
    return np.array(supList)


def getDataFromTree(filename,treename):
    arr=root2array(filename,treename)
    return arr
    #return reshapeData2(arr)
    #return np.asarray(arr)

def happyBirthday():
    print("Happy Birthday to you!")

def main():
    #signal=getSignal()
    #for e in signal:
    #    print(e.deltaThetaX)
    #happyBirthday()
    #print(getData())

    signal=getData()
    for e in signal:
        print(e)

def load_data(dataFileName):
    #dataFileName="trainingSet.root"
    signal=getSignalData(dataFileName)
    background=getBackgroundData(dataFileName)
    return signal, background

def load_training_data(dataFileName,onlysignal=False):
    signal=getSignalData(dataFileName)
    background=getBackgroundData(dataFileName)
    #dataArr=np.array(np.concatenate((signal,background),axis=0))
    print('SignalSize')
    print(len(signal))
    print('Backgroundsize')
    print(len(background))
    if(onlysignal):
    	dataArr=signal
    else:
    	dataArr=np.concatenate((signal,background),axis=0)

    #targetList=[]
    #for e in dataArr:
    #    targetList.append(e[8])
    #targetArr=np.array(targetList)
    return reshapeData(dataArr) #, reshapeData(targetArr)
    

def reshapeData(dataArr):
    print(type(dataArr[0]))
    supList=[]
    for e in dataArr:
    	subList=[]
        for i in range(15):
        	subList.append(e[i])
        supList.append(subList)
	return np.array(supList)

    '''
    for e in signal:
        targetList.append(e[8])

    for e in background:
        targetList.append(e[8])
    '''
    
def PairCheck(channelList):
    dataLength=len(channelList)
    print("Length of channelList : "+format(dataLength))
    pairCounter=0
    for indexNo in range(dataLength):
        index=indexNo
        if(index%2==0):
            #if(abs(channelList[index]-channelList[index+1])==1):
            if( (channelList[index]+1) == channelList[index+1] or channelList[index] == (channelList[index+1]+1)):
                pairCounter=pairCounter+1
            else:
                print("Pair Not Found : Index : "+format(index))
                break
                
    print("Number of Pairs : "+format(pairCounter))
    return True

#main()

if __name__ == "__main__":
#    getSignal()
    treeData=getDataFromTree("testismran.root","ftree")
    print("Shape of numpy array of TTree "+format(treeData.shape))
    print(type(treeData[0]))
    print(treeData[0].dtype)
	#print(treeData[0].fBrCh)
	
    print(treeData[320:350])
    print(treeData[0][0])
    print(treeData[0][1])
    print(treeData[0][2])
    print(treeData[0][3])
    print("====================")

    treeDataLength=len(treeData)
    print("Tree Data Length : "+format(treeDataLength) )
    tsList=[]
    channelList=[]
    for index in range(treeDataLength):
        tsList.append(treeData[index][1])
        channelList.append(treeData[index][0])
    

    PairCheck(channelList)
    '''
    #Commenting for the time being, otherwise working logic    
    minimumTS=min(tsList)
    print("Minimum TS : "+format(minimumTS))
    
    histDurationInSec=120
    histDurationInPicoSec=histDurationInSec*1e+12
    steps=1000000000
    print("histDurationInPicoSec : "+format(histDurationInPicoSec))
    bins=np.arange(minimumTS , (minimumTS+histDurationInPicoSec) , steps )
    print("Bins Shape  : "+format(bins.shape))
    plt.hist(tsList,bins=bins)
    plt.show()
    '''
    
    

    #signal=getSignal()
    #for e in signal:
    #    print(e.deltaThetaX)

