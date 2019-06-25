import numpy as np
import itertools
import random
import utils


def filter_data(data, classes_to_filter):
    filtered_data = []
    for x, y in data:
        if classes_to_filter.__contains__(y):
            filtered_data.append((x, y))

    return filtered_data


def create_ecoc_matrix(rows, columns):
    ecoc_matrix = np.zeros((rows, columns), dtype=int)
    for j in range(columns):
        # make sure that at least one positive and one negetive
        seen_one = False
        seen_neg_one = False
        while True:
            for i in range(rows):
                item = random.randint(-1, 1)
                if item == 1:
                    seen_one = True
                elif item == -1:
                    seen_neg_one = True
                ecoc_matrix[i][j] = item

            if seen_one and seen_neg_one:
                break

            # need to initialize it again to False
            seen_one = False
            seen_neg_one = False

    return ecoc_matrix


def main():
    rows = utils.numOfClasses
    columns = random.randint(4, 8)
    ecoc_matrix = create_ecoc_matrix(rows, columns)
    classifiers = []
    trainset = utils.fetchData()
    print("total train set data len: {}".format(str(len(trainset))))
    devset = utils.loadDevData()
    testset = utils.loadTestData()

    lambda_p = 1
    epoch_number = 20
    print(ecoc_matrix)

    print(len(devset), len(testset))

    for j in range(columns):
        positive = []
        negetive = []
        for i in range(rows):
            if ecoc_matrix[i][j] == 1:
                positive.append(i)
            elif ecoc_matrix[i][j] == -1:
                negetive.append(i)

        print(j, " positive: ", positive, "negetive:", negetive)
        filtered_data = filter_data(trainset, negetive + positive)
        print("filtered data", len(filtered_data))
        # need to change this function to support list
        binary_data = utils.transformToBinaryClasses(filtered_data, positiveClass=positive)
        model = utils.SVM(utils.inputDim, utils.eta, lambda_p, epoch_number)
        model.train(binary_data)
        classifiers.append(model)

    # Validation - Evaluate Test Data by Hamming Distance
    utils.validate(devset
                   , utils.HammingDistance
                   , ecoc_matrix
                   , 'test.random.ham.pred'
                   , classifiers
                   , distanceMetric="Hamming")

    # Validation - Evaluate Test Data by Loss Base Decoding
    utils.validate(devset
                   , utils.lossBaseDecoding
                   , ecoc_matrix
                   , 'test.random.loss.pred'
                   , classifiers
                   , distanceMetric="LBD")

    # Test - Evaluate Test Data by Hamming Distance
    utils.evaluate(testset
                   , utils.HammingDistance
                   , ecoc_matrix
                   , 'test.random.ham.pred'
                   , classifiers
                   , distanceMetric="Hamming")

    # Test - Evaluate Test Data by Loss Base Decoding
    utils.evaluate(testset
                   , utils.lossBaseDecoding
                   , ecoc_matrix
                   , 'test.random.loss.pred'
                   , classifiers
                   , distanceMetric="LBD")


if __name__ == "__main__":
    main()