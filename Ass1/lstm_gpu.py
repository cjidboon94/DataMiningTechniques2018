import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "10"

datasets = ['moodformood', '4things', 'allstuff']
dataset = datasets[2]

fold = 1
user = 'AS14.01.csv'

train = pd.read_csv('data_rnn/fold_{}/train/{}'.format(fold, user))
test = pd.read_csv('data_rnn/fold_{}/test/{}'.format(fold, user))

trainY, testY = np.array(train['true_mood']), np.array(test['true_mood']) 
test = test.drop(labels=['predict_day', 'user', 'true_mood', 'window'], axis=1)
train = train.drop(labels=['predict_day', 'user', 'true_mood', 'window'], axis=1)
trainX, testX = np.array(train), np.array(test) 
 
#trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#######

model = Sequential()
model.add(LSTM(4, input_shape=(1, trainX.shape[1])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(trainX, trainY, epochs=350, batch_size=1, verbose=1)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

print([[trainPredict[i], trainY[i]] for i in range(len(trainY))])
print([[testPredict[i], testY[i]] for i in range(len(testY))])
