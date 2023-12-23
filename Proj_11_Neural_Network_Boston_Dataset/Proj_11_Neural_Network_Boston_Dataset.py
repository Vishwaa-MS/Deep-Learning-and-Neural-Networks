from keras.models import Sequential #layers in sequential order
from keras.layers import Dense #ouput of prev layer in connected to all neurons
from sklearn.preprocessing import StandardScaler # satndardising => mean = 0 , standard deviation = 1
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

BosData = pd.read_csv('BostonHousing.csv')
X = BosData.iloc[:,0:13]
y = BosData.iloc[:,13] # MEDV: Median value of owner-occupied homes in $1000s

ss = StandardScaler()
X = ss.fit_transform(X)

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size = 0.2)

model = Sequential()
model.add(Dense(15, input_dim=13, activation='softplus'))
model.add(Dense(12, activation='softplus'))
model.add(Dense(1))
model.compile(loss='mean_squared_error')

history = model.fit(Xtrain, ytrain, epochs=150, batch_size=10) #data is divided into data/10 size,  so here the data is divides into 41 batches and for each epoch every batch is trained {forward and backwrd propagation and update the weights}
# so overall for one epoch the model is trained for 41 times and total 41 x 150 times teh model is trained overall
ypred = model.predict(Xtest)
ypred = ypred[:,0]

error = np.sum(np.abs(ytest-ypred))/np.sum(np.abs(ytest))*100
print('Prediction Error is',error,'%')