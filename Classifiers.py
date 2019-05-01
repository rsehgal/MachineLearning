from sklearn.metrics import confusion_matrix
from sklearn import tree

def KerasClassfier(X_train,Y_train,X_test,Y_test,num_epoch=15):
    print("========== Using Keras =========")
    from keras.models import Sequential
    from keras.layers import Dense,Dropout
    from keras.utils import to_categorical

    inputShape=X_train.shape[1]
    model = Sequential()
    model.add(Dense(8, input_shape=(inputShape,) , activation = 'relu'))
  #  model.add(Dropout(0.2))

    model.add(Dense(10, activation = 'relu'))
  #  model.add(Dropout(0.2))

    model.add(Dense(10, activation = 'relu'))
  #  model.add(Dropout(0.2))

    model.add(Dense(10, activation = 'relu'))
  #  model.add(Dropout(0.1))

    model.add(Dense(4, activation = 'softmax'))

    Y_train=to_categorical(Y_train)
    Y_test=to_categorical(Y_test)

    model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )
    model.fit(X_train, Y_train, epochs = num_epoch, batch_size = 64 ,validation_split=0.20)
    scores = model.evaluate(X_test, Y_test)
    Y_pred = model.predict(X_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    matrix = confusion_matrix(Y_test.argmax(axis=1), Y_pred.argmax(axis=1))
    print(matrix)

def MLP(X_train,Y_train,X_test,Y_test,num_iter=100,act_func='tanh'):
    print("========== MLP Classifier =========")
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(max_iter=num_iter,activation=act_func)
    clf = clf.fit(X_train, Y_train)
    X_pred=clf.predict(X_test)
    score = clf.score(X_test, Y_test)
    print(score)
    print(confusion_matrix(Y_test,X_pred))

def RandomForest(X_train,Y_train,X_test,Y_test,num_estimators=100):
    print("========== Random Forest Classifier =========")
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=num_estimators)
    clf = clf.fit(X_train, Y_train)
    X_pred=clf.predict(X_test)
    print(X_pred)
    score = clf.score(X_test, Y_test)
    print(score)
    print(confusion_matrix(Y_test,X_pred))
    '''
    print(X_pred.shape)
    supList=[]
    for i in range(X_pred.shape[0]):
        #print(X_pred[i])
        #if(X_pred[i]!=0.):
            #print("Raman")
            subList=[]
            #subList=testDataArr[i,6:9]
            for j in range(6,9):
                subList.append(testDataArr[i,j])
            subList.append(X_pred[i])
            #print(subList)
            supList.append(subList)

    import numpy as np
    filtDataArr=np.array(supList)
    np.savetxt("filteredTestPt.txt",filtDataArr,delimiter=",")
    '''
def GradientBoosting(X_train,Y_train,X_test,Y_test,num_estimators=100):
    print("========== Gradient Boosting Classifier =========")
    from sklearn.ensemble import  GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=num_estimators)
    clf = clf.fit(X_train, Y_train)
    X_pred=clf.predict(X_test)
    score = clf.score(X_test, Y_test)
    print(score)
    print(confusion_matrix(Y_test,X_pred))


def DecisionTree(X_train,Y_train,X_test,Y_test):
    print("========== DecisionTree Classifier =========")
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, Y_train)
    X_pred=clf.predict(X_test)
    score = clf.score(X_test, Y_test)
    print(score)
    print(confusion_matrix(Y_test,X_pred))

def AdaBoost(X_train,Y_train,X_test,Y_test):
    print("========== AdaBoost Classifier =========")
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(    tree.DecisionTreeClassifier(max_depth=15),    n_estimators=100,    learning_rate=1.5,    algorithm="SAMME")
    clf = clf.fit(X_train, Y_train)
    X_pred=clf.predict(X_test)
    score = clf.score(X_test, Y_test)
    print(score)
    print(confusion_matrix(Y_test,X_pred))

def LDA(X_train,Y_train,X_test,Y_test):
    print("========== Linear Discriminant Classifier =========")
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis()
    clf = clf.fit(X_train, Y_train)
    X_pred=clf.predict(X_test)
    score = clf.score(X_test, Y_test)
    print(score)
    print(confusion_matrix(Y_test,X_pred))

def NearestNeighbours(X_train,Y_train,X_test,Y_test,num_neighbours=3):
    print("========== K Nearest Neighbour Classifier =========")
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=num_neighbours)
    Y_train=Y_train.reshape(X_train.shape[0])
    neigh.fit(X_train,Y_train )
    X_pred=neigh.predict(X_test)
    #print(X_pred.shape)
    #print(X_test.shape)
    score = neigh.score(X_test, Y_test)
    print(score)
    print(confusion_matrix(Y_test,X_pred))
