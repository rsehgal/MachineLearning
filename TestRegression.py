import pandas as pd  
import numpy as np  
#import matplotlib.pyplot as plt  
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.metrics import roc_curve, auc
from scipy import interp
import numpy as np
import matplotlib.pyplot as plt
from readTree import *
from sklearn.mixture import GaussianMixture

#base learners for adaboost
from sklearn.svm import SVC
from sklearn import metrics

#import seaborn as seabornInstance 
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(15, input_dim=9, kernel_initializer='normal', activation='relu'))
	model.add(Dense(12,  kernel_initializer='normal', activation='relu'))
	model.add(Dense(10,  kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


dataset = pd.read_csv("train.csv",names=["ScatteringAngle", "DeltaThetaX", "DeltaThetaY", "DeltaX", "DeltaY", "IncomingThetaX " , "IncomingThetaY", "OutgoingThetaX", "OutgoingThetaY" ,  "Momentum","Material"])
dataset = dataset[dataset["Material"]!=0]
print(dataset.head(10))
print(dataset.shape)
print(dataset.describe())
bins=np.linspace(-0.2,0.2,100)
#plt.hist(dataset["ScatteringAngle"],bins=bins)
bins=np.linspace(0,25,1000)
#plt.hist(dataset["Momentum"],bins=bins)
X=dataset[["ScatteringAngle", "DeltaThetaX", "DeltaThetaY", "DeltaX", "DeltaY","IncomingThetaX " , "IncomingThetaY", "OutgoingThetaX", "OutgoingThetaY"]].values
#X=dataset[["ScatteringAngle"]].values #, "DeltaThetaX", "DeltaThetaY", "DeltaX", "DeltaY"]].values
y=dataset["Momentum"].values
print(y[0:10])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)

print(X_train.shape)
print(y_train.shape)
#plt.scatter(X_train, y_train, color = "red", s=1)
#coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
#print(coeff_df)

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(100)
print(df1)

df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("========= Now trying Keras ================")
'''
estimator = KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
y_pred = estimator.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)
print(df1)
'''

model = Sequential()
model.add(Dense(50, input_dim=9, kernel_initializer='normal', activation='relu'))
model.add(Dense(40,  kernel_initializer='normal', activation='relu'))
model.add(Dense(35,  kernel_initializer='normal', activation='relu'))
model.add(Dense(30,  kernel_initializer='normal', activation='relu'))
model.add(Dense(25,  kernel_initializer='normal', activation='relu'))
model.add(Dense(20,  kernel_initializer='normal', activation='relu'))
model.add(Dense(15,  kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train,y_train,epochs=30,batch_size=8)
y_pred = model.predict(X_test)
y_pred=y_pred.flatten()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


print("Shape of Y_test : "+format(y_test.shape))
print("Shape of Y_Pred : "+format(y_pred.shape))

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(100)
print(df1)


df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()



#print("10% of Mean value of Momentum "+form(0.1*))

#plt.show()



