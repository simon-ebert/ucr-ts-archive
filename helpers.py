import pandas as pd


def loadDataset(dataset):
    src = "UCR_TS_Archive_2015/" + dataset + "/" + dataset
    testData = pd.read_csv(src + "_TEST", header=None)
    trainData = pd.read_csv(src + "_TRAIN", header=None)
    return testData, trainData