from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.metrics import roc_curve, auc
from scipy import interp
import numpy as np
import matplotlib.pyplot as plt

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

def RandomForest(X_train,Y_train,X_test,Y_test,num_estimators=50):
    print("========== Random Forest Classifier =========")
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=num_estimators)
    clf = clf.fit(X_train, Y_train)
    X_pred=clf.predict(X_test)
    print(X_pred)
    score = clf.score(X_test, Y_test)
    #score = clf.predict_proba(X_test)

    #fpr,tpr,thres= roc_curve(Y_test, score)
    #roc_auc = auc(fpr, tpr)
    print("========= ROC ===========")
    #print(roc_auc)
    print(score)
    print(confusion_matrix(Y_test,X_pred))


    #===============================================================
    model = clf
    y_test = Y_test
    y_predict_proba = model.predict_proba(X_test)

    # Compute ROC curve and ROC AUC for each class
    n_classes = 3 
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    all_y_test_i = np.array([])
    all_y_predict_proba = np.array([])
    for i in range(n_classes):
	y_test_i = map(lambda x: 1 if x == i else 0, y_test)
	all_y_test_i = np.concatenate([all_y_test_i, y_test_i])
	all_y_predict_proba = np.concatenate([all_y_predict_proba, y_predict_proba[:, i]])
	fpr[i], tpr[i], _ = roc_curve(y_test_i, y_predict_proba[:, i])
	roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["average"], tpr["average"], _ = roc_curve(all_y_test_i, all_y_predict_proba)
    roc_auc["average"] = auc(fpr["average"], tpr["average"])


    # Plot average ROC Curve
    plt.figure()
    plt.plot(fpr["average"], tpr["average"],
	     label='Average ROC curve (area = {0:0.2f})'
		   ''.format(roc_auc["average"]),
	     color='deeppink', linestyle=':', linewidth=4)

    # Plot each individual ROC curve
    for i in range(n_classes):
	plt.plot(fpr[i], tpr[i], lw=2,
		 label='ROC curve of class {0} (area = {1:0.2f})'
		 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


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
