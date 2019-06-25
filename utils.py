import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle

# global variables
eta = 0.1
numOfClasses = 4
inputDim = 784


def fetchData():
    # read data
    mnist = fetch_mldata("MNIST original", data_home="./data")

    X, Y = mnist.data[:60000] / 255., mnist.target[:60000]
    x = [np.reshape(ex, (1, 784)) for ex, ey in zip(X, Y) if ey in [0, 1, 2, 3]]
    y = [ey for ey in Y if ey in [0, 1, 2, 3]]
    # suffle examples
    x, y = shuffle(x, y, random_state=1)

    # since we don't want to work with generator (python3 zip function) thus using list
    return list(zip(x, y))


def loadDevData():
    dset = np.loadtxt('./x_test.txt')
    labels = np.loadtxt('./y_test.txt')
    devset = []

    for x in dset:
        devset.append(np.reshape(x, (1, 784)))

    return list(zip(devset, labels))


def loadTestData():
    tset = np.loadtxt('./x4pred.txt')
    testset = []

    for x in tset:
        testset.append(x.reshape((1, 784)))

    return testset


'''
 since we have multiclass data but wish to train binray classifier
 we need to have a function that transforms classes to binray classification
 parameter: data is a list of pairs of x, y
            positiveClass is the class that will be mapped to +1 all other will be
                          mapped to -1
'''


def transformToBinaryClasses(data, positiveClass):
    newData = []
    for x, y in data:
        if (y in positiveClass):
            newData.append((x, 1))
        else:
            newData.append((x, -1))

    return newData


def createOutputVectors(x, models, distanceMetric="Hamming"):
    outputVec = []  # for Hamming Distance
    predVec = []  # for Loss Base Decoding

    for model in models:
        y_tag, pred = model.inference(x)
        outputVec.append(y_tag)
        predVec.append(pred)

    if (distanceMetric == "Hamming"):
        return outputVec
    else:
        return predVec


def HammingDistance(ecocMat, outputVec):
    rows = ecocMat.shape[0]
    distance = []

    # iterate over rows - each row is a class
    for r in range(rows):
        # iterate over the columns of the ith row - each column is a classifier
        val = 0
        for i, s in enumerate(ecocMat[r]):
            val += (1 - np.sign(s * outputVec[i])) / 2
        distance.append(val)

    return np.argmin(distance)


def lossBaseDecoding(ecocMat, predVec):
    rows = ecocMat.shape[0]
    distance = []

    # iterate over rows - each row is a class
    for r in range(rows):
        # iterate over the columns of the ith row - each column is a classifier
        val = 0
        for i, s in enumerate(ecocMat[r]):
            tmp = 1 - (s * predVec[i])
            val += np.max([tmp, 0])

        distance.append(val)

    return np.argmin(distance)


def validate(dataset, distFunc, ecocMat, filePath, classifiers, distanceMetric="Hamming"):
    correct = 0
    incorrect = 0

    for x, y in dataset:
        vec = createOutputVectors(x, classifiers, distanceMetric)

        y_tag = distFunc(ecocMat, vec)
        if (y_tag == y):
            correct += 1
        else:
            incorrect += 1

    acc = 1. * correct / len(dataset)
    print("Validation {} : correct: {} incorrect: {} total: {}\n accuracy: {}".format( \
        distanceMetric, correct, incorrect, len(dataset), acc))


def evaluate(dataset, distFunc, ecocMat, filePath, classifiers, distanceMetric="Hamming"):
    predictions = []

    for x in dataset:
        vec = createOutputVectors(x, classifiers, distanceMetric)

        y_tag = distFunc(ecocMat, vec)
        predictions.append(str(y_tag))

    # save results
    with open('./pred/' + filePath, 'w+') as f:
        f.write("\n".join(predictions))

    print("Test finish")


class SVM(object):
    def __init__(self, inputDim, eta, lambdaP, epochs):
        self.W = np.zeros((inputDim, 1))  # weights
        self.eta = eta  # learning rate
        self.lambdaP = lambdaP  # regularization parameter
        self.epochs = epochs

    def train(self, trainset, printLoss=False):
        # iterate over epochs
        for t in range(1, self.epochs + 1):
            np.random.shuffle(trainset)
            # we want to diminish eta the more we progress, to make "small steps", no to loose min
            eta = self.eta / np.sqrt(t)
            loss = 0

            # iterate over trainset
            for x, y in trainset:
                pred = y * (np.dot(x, self.W)[0][0])  # get the scalar
                reg = (eta * self.lambdaP) * self.W  # regularization
                if (1 - pred >= 0):  # check hinge loss
                    lossAddition = (eta * y * x)
                    # need to reshape since x and W are not the same shape
                    self.W += np.reshape(lossAddition, (784, 1)) - reg
                else:
                    self.W -= reg
                loss += 1 - pred

            # for sanity checks and testing
            if (printLoss):
                print("Loss for #{} epoch is {}".format(t, loss))

    def inference(self, x):
        # since we use binary classifier
        pred = np.dot(x, self.W)
        return np.sign(pred)[0][0], pred