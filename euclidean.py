import numpy as np
import sys


def oneNN(testData, trainData):
    length = len(testData.columns)
    results = dict()

    for i in range(len(testData)):
        test = testData.ix[i]
        minDist = sys.maxsize
        for index, row in trainData.iterrows():
            train = trainData.ix[index]
            dist = np.linalg.norm(train[1:] - test[1:])
            if dist < minDist:
                minDist = dist
                results[i] = train[0]

    # calculate error rate
    errors = 0
    for i in range(len(testData)):
        if results[i] != testData[0][i]:
            errors += 1
    return errors / len(testData)
