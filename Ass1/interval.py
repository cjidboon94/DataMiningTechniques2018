'''
- Creates interval data (window=[2,3,4]) train and test splits (fold 1-4)
- Uses train/test splits from data_complete
- Removes data from user AS14.32
- Only considers mood, arousal, valence and activity from previous days as predictor
'''

import numpy as np
import pandas as pd
import os

trainfiles = os.listdir('data_normalized/fold_{}/train'.format(1))
testfiles = os.listdir('data_normalized/fold_{}/test'.format(1))
#trainfiles.remove('AS14.32.csv')
#testfiles.remove('AS14.32.csv')

for user in trainfiles:
    for fold in [1, 2, 3, 4]:
        for window in [2, 3, 4]:
            train_dataset  = pd.read_csv('data_normalized/fold_{}/train/'.format(fold) + user)
            test_dataset = pd.read_csv('data_normalized/fold_{}/test/'.format(fold) + user)

            test_indices = list(test_dataset['Unnamed: 0'])
            print test_indices
            train_indices = list(train_dataset['Unnamed: 0'])
            print train_indices

            df = pd.concat([train_dataset, test_dataset])
            df = df.sort_values('Unnamed: 0')

            dataset_df = pd.DataFrame(columns=['predict_day', 'user', 'window', 'mood', 'arousal', \
                                                   'valence', 'activity', 'true_mood', 'indexje'])

            print df
            days = list(df['Unnamed: 0.1'])
            for i in range(len(df)):
                if i + window <= len(df) - 1:
                    windowstart = days[i]
                    windowend = days[i+window-1]
                    predictday = days[i+window]

                    avg_mood = df[i:i+window]['mood'].mean()
                    avg_arousal = df[i:i+window]['arousal'].mean()
                    avg_valence = df[i:i+window]['valence'].mean()
                    avg_activity = df[i:i+window]['activity'].mean()

                    dataset_df = dataset_df.append({'predict_day': predictday, 'user': user[:-4], 'window': '{}-{}'.format(windowstart, \
                                                     windowend), 'mood': avg_mood, 'arousal': avg_arousal, \
                                                     'valence': avg_valence, 'activity': avg_activity, \
                                                     'true_mood': list(df['mood'])[i+window], 'indexje': list(df['Unnamed: 0'])[i]}, ignore_index=True)

            dataset_df.indexje = dataset_df.indexje.astype(int)
            dataset_df.index = dataset_df['indexje']

            test_data = dataset_df.loc[[a-window for a in test_indices]]
            train_data = dataset_df.loc[[a-window for a in train_indices]]

            train_data = train_data[pd.notnull(train_data['predict_day'])]
            test_data = test_data[pd.notnull(test_data['predict_day'])]
            train_data = train_data.drop('indexje', axis=1)
            test_data = test_data.drop('indexje', axis=1)
            train_data.to_csv('interval_datasets_normalized/window_{}/fold_{}/train/{}'.format(window, fold, user), index=False)
            test_data.to_csv('interval_datasets_normalized/window_{}/fold_{}/test/{}'.format(window, fold, user), index=False)
            print train_data
