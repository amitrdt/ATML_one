import numpy as np
import itertools
import utils


def get_all_pairs(classes_number):
    return itertools.combinations(range(classes_number), 2)


def filter_data(data, classes_to_filter):
    filtered_data = []
    for x, y in data:
        if classes_to_filter.__contains__(y):
            filtered_data.append((x, y))

    return filtered_data


def update_ecoc_matrix(ecoc_matrix, classifier_index, y0, y1):
    ecoc_matrix[y0, classifier_index] = 1
    ecoc_matrix[y1, classifier_index] = -1


def main():
    k = utils.numOfClasses
    columns = int(k * (k-1) / 2)
    ecoc_matrix = np.zeros((k, columns), dtype=int)
    classifiers = []
    trainset = utils.fetchData()
    devset = utils.loadDevData()
    print("total train set data len: {}".format(str(len(trainset))))
    testset = utils.loadTestData()

    lambda_p = 1
    epoch_number = 20
    pair_index = 0

    # Train All-Pair Classifiers
    for y0, y1 in get_all_pairs(utils.numOfClasses):
        update_ecoc_matrix(ecoc_matrix, pair_index, y0, y1)
        print("working on pair {},{}".format(y0, y1))
        pair_index = pair_index + 1
        filtered_data = filter_data(trainset, (y0, y1))
        print("pair relevant data: {}".format(str(len(filtered_data))))
        binary_data = utils.transformToBinaryClasses(filtered_data, positiveClass=[y0])
        model = utils.SVM(utils.inputDim, utils.eta, lambda_p, epoch_number)
        model.train(binary_data)
        classifiers.append(model)
        print("finished with #{} model".format(pair_index))

    print(ecoc_matrix)

    # Evaluate Test Data by Hamming Distance
    utils.validate(devset
                     , utils.HammingDistance
                     , ecoc_matrix
                     , 'test.allpairs.ham.pred'
                     , classifiers
                     , distanceMetric="Hamming")

    # Evaluate Test Data by Loss Base Decoding
    utils.validate(devset
                     , utils.lossBaseDecoding
                     , ecoc_matrix
                     , 'test.allpairs.loss.pred'
                     , classifiers
                     , distanceMetric="LBD")

    # Evaluate Test Data by Hamming Distance
    utils.evaluate(testset
                     , utils.HammingDistance
                     , ecoc_matrix
                     , 'test.allpairs.ham.pred'
                     , classifiers
                     , distanceMetric="Hamming")

    # Evaluate Test Data by Loss Base Decoding
    utils.evaluate(testset
                     , utils.lossBaseDecoding
                     , ecoc_matrix
                     , 'test.allpairs.loss.pred'
                     , classifiers
                     , distanceMetric="LBD")


if __name__ == "__main__":
    main()