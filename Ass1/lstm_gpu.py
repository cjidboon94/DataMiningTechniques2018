import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "10"

datasets = ['moodformood', '4things', 'allstuff']
dataset = datasets[1]

user = 'AS14.01.csv'
fold = 1
testfold0 = pd.read_csv('data_normalized/fold_{}/test/{}'.format(fold, user))

fold = 1
train = pd.read_csv('data_normalized/fold_{}/train/{}'.format(fold, user))
test = pd.read_csv('data_normalized/fold_{}/test/{}'.format(fold, user))

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return dataX, np.array(dataY)

look_back = 1

def reshape_etc(data):
    size = len(data)
    data = np.array(data).reshape(size, 1)
    return data

##### prepare data
trainX, trainY = create_dataset(np.array(train['mood']), look_back)
testX, testY = create_dataset(np.array(test['mood']), look_back)
trainX, trainY = reshape_etc(trainX), reshape_etc(trainY)

if dataset is not 'moodformood':
    valence_train_X, valence_train_Y = create_dataset(np.array(train['valence']), look_back)
    arousal_train_X, arousal_train_Y = create_dataset(np.array(train['arousal']), look_back)
    activity_train_X, activity_train_Y = create_dataset(np.array(train['activity']), look_back)

    valence_test_X, valence_test_Y = create_dataset(np.array(test['valence']), look_back)
    arousal_test_X, arousal_test_Y = create_dataset(np.array(test['arousal']), look_back)
    activity_test_X, activity_test_Y = create_dataset(np.array(test['activity']), look_back)

    valence_train_X = reshape_etc(valence_train_X)
    arousal_train_X = reshape_etc(arousal_train_X)
    activity_train_X = reshape_etc(activity_train_X)
    valence_test_X = reshape_etc(valence_test_X)
    arousal_test_X = reshape_etc(arousal_test_X)
    activity_test_X = reshape_etc(activity_test_X)

    trainX = np.column_stack((trainX, valence_train_X, arousal_train_X, activity_train_X))

    testX = np.column_stack((testX, valence_test_X, arousal_test_X, activity_test_X))     
 
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#######

model = Sequential()
model.add(LSTM(4, input_shape=(look_back, 4)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=500, batch_size=1, verbose=1)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

print([[trainPredict[i][0], trainY[i][0]] for i in range(len(trainY))])
print([[testPredict[i][0], testY[i][0]] for i in range(len(testY))])
