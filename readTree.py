import numpy as np
from root_numpy import root2array, tree2array
from ROOT import TFile, TTree, TList

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
    supList=[]
    for e in dataArr:
        #print(e)
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

#main()

#if __name__ == "__main__":
#    getSignal()
    #signal=getSignal()
    #for e in signal:
    #    print(e.deltaThetaX)

