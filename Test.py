import pandas as pd
from euclidean import oneNN
from helpers import loadDataset
import os

# Loop through archive
names = os.listdir("UCR_TS_Archive_2015")
for name in names:
    # load dataset
    testData, trainData = loadDataset(name)

    # Array of classes
    classes = trainData[0].unique()

    # Dict of averaged class representatives
    d = dict()
    for c in classes:
        d[c] = trainData[trainData[0] == c].mean()
    averagedTrainData = pd.DataFrame.from_dict(d, orient="index")

    print(name, oneNN(testData, averagedTrainData))
