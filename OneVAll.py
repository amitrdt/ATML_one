# One vs All
import numpy as np
import utils

# error correcting output codes matrix
ecocMat = (np.ones(utils.numOfClasses) * -1) + (np.eye(utils.numOfClasses) * 2)

def main():
    classifiers = []
    trainset = utils.fetchData()
    devset = utils.loadDevData()
    testset = utils.loadTestData()

    # train OvA classifiers
    for i in range(utils.numOfClasses):
        binData = utils.transformToBinaryClasses(trainset, positiveClass=[i])
        model = utils.SVM(utils.inputDim, utils.eta, 1, 50)
        model.train(binData)
        classifiers.append(model)
        print("finished with #{} model".format(i))

    # Validation - Evaluate Test Data by Hamming Distance
    utils.validate(devset
                     , utils.HammingDistance
                     , ecocMat
                     , 'test.onevall.ham.pred'
                     , classifiers
                     , distanceMetric="Hamming")

    # Validation - Evaluate Test Data by Loss Base Decoding
    utils.validate(devset
                     , utils.lossBaseDecoding
                     , ecocMat
                     , 'test.onevall.ham.pred'
                     , classifiers
                     , distanceMetric="LBD")

    # Test - Evaluate test data by Hamming Distance
    utils.evaluate(testset
                     , utils.HammingDistance
                     , ecocMat
                     , 'test.onevall.ham.pred'
                     , classifiers
                     , distanceMetric="Hamming")

    # Test - Evaluate test data by Loss Base Decoding
    utils.evaluate(testset
                     , utils.lossBaseDecoding
                     , ecocMat
                     , 'test.onevall.loss.pred'
                     , classifiers
                     , distanceMetric="LBD")

if __name__ == "__main__":
    main()