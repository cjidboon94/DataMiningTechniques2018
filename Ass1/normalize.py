import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

datafilestrain1 = os.listdir('data_complete')
datafiles = [a for a in datafilestrain1 if a.startswith('AS')]
datafiles.remove('AS14.32.csv')

for datafile in datafiles:
    dataset = pd.read_csv('data_complete/' + datafile)

    # bug data fix
    if datafile == 'AS14.07.csv':
        newval = np.mean(list(dataset['builtin'])[:36] + list(dataset['builtin'])[37:])
        dataset['builtin'][36] = newval

    dataset['gsm'] = dataset['sms'] + dataset['calls']
    dataset['amusement'] = dataset['game'] + dataset['entertainment']
    dataset['system'] = dataset['utilities'] + dataset['builtin']
    dataset = dataset.drop(['Unnamed: 0', 'unknown', 'sms', 'calls', 'game', 'entertainment', \
                            'utilities', 'builtin', 'other'], axis=1)

    variables = dataset.columns
    print variables
    print '\nUser: {}. Dataset size: {}'.format(datafile, len(dataset))

    for variable in variables:
        if variable in ['valence', 'arousal', 'mood', 'activity']:
            pass
        else:
            #print ([int(x) for x in dataset[variable]])[1:]
            print '{}: {}'.format(variable, sum(np.bincount([x for x in dataset[variable]])[1:]))
            if sum(np.bincount([x for x in dataset[variable]])[1:]) < len(dataset)/10:
                print variable
                dataset = dataset.drop(variable, axis=1)

    timevars = [u'screen', u'communication', u'finance', u'office', u'social', u'travel', \
                    u'weather', u'amusement', u'system']

    # set all (remaining) time variables as percentages
    for var in timevars:
        try:
            dataset[var] = dataset[var]/86400
        except:
            pass

    #minmax scale gsm
    x = dataset['gsm']
    dataset['gsm'] = list(x-min(x))/(max(x)-min(x))
    dataset.to_csv('data_normalized/' + datafile, index=False)
