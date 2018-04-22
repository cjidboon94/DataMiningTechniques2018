import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
#import matplotlib.pyplot as plt

checkpoint_path = '/home/s1924192/lstmweights_weekdays/lstm.hdf5'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "13"

datasets = ['moodformood', '4things', 'allstuff']
dataset = datasets[2]

userfiles = os.listdir('data_normalized/fold_1/train')

for fold in [1,2,3,4]:
    for user in userfiles:
        train = pd.read_csv('data_rnn_weekdays/fold_{}/train/{}'.format(fold, user))
        test = pd.read_csv('data_rnn_weekdays/fold_{}/test/{}'.format(fold, user))

        trainY, testY = np.array(train['true_mood']), np.array(test['true_mood']) 
        
        train_days = np.array((train['predict_day'])).reshape(-1)
        test_days = np.array((test['predict_day'])).reshape(-1)
        
        test = test.drop(labels=['predict_day', 'user', 'true_mood', 'window'], axis=1)
        train = train.drop(labels=['predict_day', 'user', 'true_mood', 'window'], axis=1)
        trainX, testX = np.array(train), np.array(test) 

        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        model = Sequential()
        model.add(LSTM(4, input_shape=(1, trainX.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0,
                                     save_best_only=True, mode='min')
        stopearly = EarlyStopping(monitor='val_loss', min_delta=0, patience=50,
                                  verbose=0, mode='min')
        callbacks_list = [checkpoint, stopearly]

        history = model.fit(trainX, trainY, validation_data=[testX, testY], epochs=500, batch_size=1, callbacks=callbacks_list, verbose=1)

        trainPredict = model.predict(trainX).reshape(-1)
        testPredict = model.predict(testX).reshape(-1)

        trainY = trainY.reshape(-1)
        testY = testY.reshape(-1)
        # errors
        train_errors = trainY - trainPredict
        test_errors = testY - testPredict

        pd.DataFrame([train_days, train_errors, test_days, test_errors]).to_csv('data_normalized_res/rnn_weekdays/fold_{}/{}'.format(fold, user))