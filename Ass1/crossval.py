import numpy as np
import pandas as pd
import os

datafiles = os.listdir('data_normalized')
datafiles = [a for a in datafiles if a.startswith('AS')]
nfolds = 5
sizes = []

for datafile in datafiles:
    dataset = pd.read_csv('data_normalized/' + datafile)
    user = datafile[:-4]

    #print dataset
    datasize = len(dataset)
    foldsize = datasize/nfolds

    print datasize
    sizes.append(datasize)

    #test_0 = data[-foldsize:]
    test_0 =range(datasize)[-foldsize:]

    foldsequences = []
    for fold in range(nfolds):
        foldsequences.append([x-(datasize/nfolds * fold) for x in test_0])

    foldsequences = list(reversed(foldsequences))
    allindices = sum(foldsequences, [])

    for fold in range(nfolds):
        test = dataset.iloc[foldsequences[fold]]
        train = dataset.iloc[[a for a in allindices if a not in foldsequences[fold]]]
        # print test
        test.to_csv('data_normalized/fold_{}/test/{}.csv'.format(fold,user))
        train.to_csv('data_normalized/fold_{}/train/{}.csv'.format(fold,user))
    print foldsequences
