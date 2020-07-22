from readTree import *
from Classifiers import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

#dataArr=load_training_data("trainPb.root")
trainfilename="train.root"
#CreateData(["TMVA_Fe.root","TMVA_Al.root","TMVA_Pb.root","TMVA_U.root"],trainfilename)
CreateData(["TMVA_Al_100_100_20_20K.root","TMVA_Fe_100_100_20_20K.root","TMVA_Pb_100_100_20_20K.root"],trainfilename)
#CreateData(["TMVA_Pb.root","TMVA_U.root"],trainfilename)
onlySignal=True
dataArr=load_training_data("train.root",onlysignal=onlySignal)
testDataArr=load_training_data("test.root",onlysignal=onlySignal)
np.random.shuffle(dataArr)
np.random.shuffle(testDataArr)

X=dataArr[:,0:6]
Y=dataArr[:,9:10]
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)


#using real test data
X_train=X
Y_train=Y
X_test=testDataArr[:,0:9]
Y_test=testDataArr[:,9:10]


KerasClassfier(X_train,Y_train,X_test,Y_test,50)

Y_train=Y_train.reshape(X_train.shape[0])

#GaussianMixtureModel(X_train,Y_train,X_test,Y_test,writeToFile=True)


NearestNeighbours(X_train,Y_train,X_test,Y_test,writeToFile=True)
#RandomForest(X_train,Y_train,X_test,Y_test,writeToFile=True,num_estimators=100)
#print("========= Printing Returned Predicted Values +=============")
#print(X_pred)
#GradientBoosting(X_train,Y_train,X_test,Y_test,writeToFile=True)
#DecisionTree(X_train,Y_train,X_test,Y_test,True)
#LDA(X_train,Y_train,X_test,Y_test,True)
#NearestNeighbours(X_train,Y_train,X_test,Y_test,True)
#MLP(X_train,Y_train,X_test,Y_test)
#AdaBoost(X_train,Y_train,X_test,Y_test,writeToFile=True)
#Bagging(X_train,Y_train,X_test,Y_test,writeToFile=True)

#CalibrationPlot(X_train,Y_train,X_test,Y_test)
#Ensemble(X_train,Y_train,X_test,Y_test,writeToFile=True)
#NoveltyDetection()
