from readTree import *
from Classifiers import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#dataArr=load_training_data("trainPb.root")
dataArr=load_training_data("train.root")
testDataArr=load_training_data("test.root")
X=dataArr[:,0:6]
Y=dataArr[:,9:10]
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)


#using real test data
X_train=X
Y_train=Y
X_test=testDataArr[:,0:6]
Y_test=testDataArr[:,9:10]


#KerasClassfier(X_train,Y_train,X_test,Y_test,5)

Y_train=Y_train.reshape(X_train.shape[0])
RandomForest(X_train,Y_train,X_test,Y_test)
#GradientBoosting(X_train,Y_train,X_test,Y_test)
#DecisionTree(X_train,Y_train,X_test,Y_test)
#LDA(X_train,Y_train,X_test,Y_test)
#NearestNeighbours(X_train,Y_train,X_test,Y_test)
#MLP(X_train,Y_train,X_test,Y_test)
#AdaBoost(X_train,Y_train,X_test,Y_test)
