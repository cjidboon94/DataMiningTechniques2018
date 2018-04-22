'''
- Creates interval data (window=[2,3,4]) train and test splits (fold 1-4)
- Uses train/test splits from data_complete
- Removes data from user AS14.32
- Only considers mood, arousal, valence and activity from previous days as predictor
'''

import numpy as np
import pandas as pd
import os
import datetime

trainfiles = os.listdir('data_complete/fold_{}/train'.format(1))
testfiles = os.listdir('data_complete/fold_{}/test'.format(1))

try:
    trainfiles.remove('AS14.32.csv')
    testfiles.remove('AS14.32.csv')
except:
    pass


for user in trainfiles:
    for fold in [1, 2, 3, 4]:
        for window in [2, 3, 4]:
            train_dataset  = pd.read_csv('data_complete/fold_{}/train/'.format(fold) + user)
            test_dataset = pd.read_csv('data_complete/fold_{}/test/'.format(fold) + user)

            test_indices = list(test_dataset['Unnamed: 0'])
            print test_indices
            train_indices = list(train_dataset['Unnamed: 0'])
            print train_indices

            df = pd.concat([train_dataset, test_dataset])
            df = df.sort_values('Unnamed: 0')
            
            normalized_train = pd.read_csv('data_normalized/fold_{}/train/'.format(fold) + user)
            normalized_test = pd.read_csv('data_normalized/fold_{}/test/'.format(fold) + user)
            vars1 = list(normalized_train.columns)
            vars1.remove('Unnamed: 0')
            print vars1
            vars2 = ['predict_day', 'user', 'window', 'true_mood', 'indexje', 'weekday', 'weekend_day']
            cols=vars2+vars1
            
            dataset_df = pd.DataFrame(columns=cols)
            days = list(df['Unnamed: 0.1'])
            for i in range(len(df)):
                if i + window <= len(df) - 1:
                    windowstart = days[i]
                    windowend = days[i+window-1]
                    predictday = days[i+window]

                    day_ints = [int(x) for x in predictday.split('-')]
                    datum = datetime.date(day_ints[0], day_ints[1], day_ints[2])
                    if datum.weekday() in [4,5,6]:
                        weekenddag = 1
                        weekdag = 0
                    elif datum.weekday() in [0,1,2,3]:
                        weekenddag = 0
                        weekdag = 1

                    avg_mood = df[i:i+window]['mood'].mean()
                    avg_arousal = df[i:i+window]['arousal'].mean()
                    avg_valence = df[i:i+window]['valence'].mean()
                    avg_activity = df[i:i+window]['activity'].mean()
                    
                    normalized_row = normalized_test.loc[normalized_test['Unnamed: 0'] == days[i]]
                    if normalized_row.empty:
                        normalized_row = normalized_train.loc[normalized_train['Unnamed: 0'] == days[i]]

                    normalized_df = pd.DataFrame(columns=normalized_row.columns)
                    for j in range(window):
                        normalized_row = normalized_test.loc[normalized_test['Unnamed: 0'] == days[i+j]]
                        if normalized_row.empty:
                            normalized_row = normalized_train.loc[normalized_train['Unnamed: 0'] == days[i+j]]
                        normalized_df = pd.concat([normalized_df, normalized_row])
                
                    vardict = {}
                    for var in vars1:
                        vardict['predict_day'] = predictday
                        vardict['mood'] = float(avg_mood)
                        vardict['arousal'] = float(avg_arousal)
                        vardict['valence'] = float(avg_valence)
                        vardict['activity'] = float(avg_activity)
                        vardict['true_mood'] = list(df['mood'])[i+window]
                        vardict['indexje'] = list(df['Unnamed: 0'])[i]
                        vardict['user'] = user[:-4]
                        vardict['window'] = '{}-{}'.format(windowstart, windowend)
                        vardict['weekday'] = weekdag
                        vardict['weekend_day'] = weekenddag
                        if var not in ['mood', 'arousal', 'valence', 'activity', 'Unnamed: 0']:
                            vardict[var] = float(normalized_df[var].mean())

                    dataset_df = dataset_df.append(vardict, ignore_index=True)

            dataset_df.indexje = dataset_df.indexje.astype(int)
            dataset_df.index = dataset_df['indexje']

            test_data = dataset_df.loc[[a-window for a in test_indices]]
            train_data = dataset_df.loc[[a-window for a in train_indices]]

            train_data = train_data[pd.notnull(train_data['predict_day'])]
            test_data = test_data[pd.notnull(test_data['predict_day'])]
            train_data = train_data.drop('indexje', axis=1)
            test_data = test_data.drop('indexje', axis=1)
            train_data.to_csv('interval_datasets_done+weekdays/window_{}/fold_{}/train/{}'.format(window, fold, user), index=False)
            test_data.to_csv('interval_datasets_done+weekdays/window_{}/fold_{}/test/{}'.format(window, fold, user), index=False)

            print user, fold, window