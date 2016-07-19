import pandas as pd
from euclidean import oneNN
from helpers import loadDataset


# load dataset
dataset = "BeetleFly"
testData, trainData = loadDataset(dataset)

# Array of classes
classes = trainData[0].unique()

# Dict of averaged class representatives
d = dict()
for c in classes:
    d[c-1] = trainData[trainData[0]==c].mean()
averagedTrainData = pd.DataFrame.from_dict(d, orient="index")

print(testData.head())
print(averagedTrainData.head())

print(oneNN(testData,averagedTrainData))