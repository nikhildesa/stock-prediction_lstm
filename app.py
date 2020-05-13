import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('all_stocks_5yr.csv')
data = data[data['Name'] == 'AAL']
df = data['close']
df = df.to_frame()

print(df.isnull().values.any())
print(df.isna().sum())

df = df.dropna()


#<------------ Regression---------------->

future_days = 30
df['prediction'] = df['close'].shift(-future_days)

df.head()
df.tail()

#<-------  X ------------>

X = df.drop(['prediction'],1)
X = np.array(X)

X = X[:-future_days]

#<-------- Y ---------------->

y = df.drop(['close'],1)
y = np.array(y)

y = y[:-future_days]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
model = SVR(kernel='rbf',C=1.0, epsilon=0.2)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_pred = y_pred.reshape(-1,1)
print(model.score(X_test,y_test))

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print(lr.score(X_test,y_test))



#<--------------  LSTM----------------------------->

data = pd.read_csv('all_stocks_5yr.csv')
data = data[data['Name'] == 'AAL']
close_price = data['close']
close_price = close_price.to_frame()

plt.plot(close_price)


scaler = MinMaxScaler(feature_range=(0, 1))
close_price = scaler.fit_transform(close_price)

# Do not shuffle the data as we have to learn from the time series.

train_size = int(len(close_price) * 0.80)
train = close_price[0:train_size,:]
test = close_price[train_size:,:]

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Model
import time
import tensorflow as tf
from subprocess import check_output
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

model = Sequential()

model.add(LSTM(
    input_dim=1,
    output_dim=50,
    return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
    100,
    return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(
    output_dim=1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print ('compilation time : ', time.time() - start)

model.fit(
    trainX,
    trainY,
    batch_size=128,
    nb_epoch=10,
    validation_split=0.05)


y_pred = model.predict(testX)



#plot
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(trainY)
plt.plot(testY)
plt.plot(y_pred)
plt.show()


