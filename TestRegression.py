import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
#import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
dataset = pd.read_csv("trainFe.csv",names=["ScatteringAngle", "DeltaThetaX", "DeltaThetaY", "DeltaX", "DeltaY","Momentum","Material"])
dataset = dataset[dataset["Material"]!=0]
print(dataset.head(10))
print(dataset.shape)
print(dataset.describe())
bins=np.linspace(-0.2,0.2,100)
plt.hist(dataset["ScatteringAngle"],bins=bins)
X=dataset[["ScatteringAngle", "DeltaThetaX", "DeltaThetaY", "DeltaX", "DeltaY"]].values
y=dataset["Momentum"].values
print(y[0:10])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
#coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
#print(coeff_df)

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)
print(df1)

df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#print("10% of Mean value of Momentum "+form(0.1*))

#plt.show()

