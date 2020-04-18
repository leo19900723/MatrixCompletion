import collections
import numpy


def printProgressBar(iteration, total, delimiter=None, prefix="", suffix="", decimals=1, length=100, fill="█", printEnd="\r"):
    if iteration == total or delimiter is None or iteration % (delimiter * (total / 100)) == 0:
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + "-" * (length - filledLength)
        print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end=printEnd)
        # Print New Line on Complete
        if iteration == total:
            print()


def matrixFactorization(A_orgInputMatrix):
    epoch = 11
    lambda_regularizationController = 0.02
    gamma_biasModificationController = 0.0000003

    print("Matrix Factorization...")
    AShape = A_orgInputMatrix.shape
    matrixEuclideanDistance = lambda matrixA, matrixB: numpy.sum(numpy.power(matrixA - numpy.clip(numpy.round(matrixB), 1, 5), 2))

    print("Squaring the A_orgInputMatrix...")
    A_orgInputMatrix = numpy.pad(A_orgInputMatrix, ((0, AShape[1] - AShape[0] if AShape[1] > AShape[0] else 0), (0, AShape[0] - AShape[1] if AShape[0] > AShape[1] else 0)), "constant", constant_values=numpy.nan)
    A_maskedOrgInputMatrix = numpy.ma.masked_array(A_orgInputMatrix, numpy.isnan(A_orgInputMatrix))

    mu_avgOfInputMatrix = numpy.mean(A_maskedOrgInputMatrix)

    print("Calculating the SVD of A_orgInputMatrix...")
    U_userToConcept, S_singularValueArray, VT_itemToConcept = numpy.linalg.svd((A_maskedOrgInputMatrix - mu_avgOfInputMatrix).filled(0), full_matrices=False)
    S_singularValueArray = numpy.diag(S_singularValueArray)
    bu, bi, K = 0, 0, int(numpy.count_nonzero(S_singularValueArray))

    print("Training...")
    for round in range(epoch):
        R_returnInputMatrix = mu_avgOfInputMatrix + U_userToConcept[:, :K] @ VT_itemToConcept[:K, :]
        err = matrixEuclideanDistance(A_maskedOrgInputMatrix, R_returnInputMatrix)

        U_userToConcept = U_userToConcept + gamma_biasModificationController * (err * VT_itemToConcept.T - lambda_regularizationController * U_userToConcept)
        VT_itemToConcept = (VT_itemToConcept.T + gamma_biasModificationController * (err * U_userToConcept - lambda_regularizationController * VT_itemToConcept.T)).T

        print("Epoch:", round, "\tErr:", err)

    R_returnInputMatrix = numpy.clip(numpy.round(R_returnInputMatrix[:AShape[0], :AShape[1]]), 1, 5)
    print("----\nRMSE: ", numpy.sqrt(numpy.nanmean((A_orgInputMatrix[:AShape[0], :AShape[1]] - R_returnInputMatrix)**2)))

    return R_returnInputMatrix, int(mu_avgOfInputMatrix.round())


def _unitTest():
    print("Loading Files...")
    trainningData = numpy.genfromtxt("InputFiles/train.csv", delimiter=",", dtype=int)[1:, :-1]
    testingData = numpy.genfromtxt("InputFiles/test.csv", delimiter=",", dtype="U10")
    A_orgInputHash = collections.defaultdict(lambda: {})
    userDict, itemDict = {}, {}

    print("Loading the input file to A_orgInputHash (users * items)...")
    for rowIndex in range(len(trainningData)):
        printProgressBar(rowIndex + 1, len(trainningData), delimiter=5, prefix="\tProgress:", suffix="Complete", length=50)
        A_orgInputHash[trainningData[rowIndex, 1]][trainningData[rowIndex, 0]] = trainningData[rowIndex, 2]
        if trainningData[rowIndex, 0] not in itemDict:
            itemDict[trainningData[rowIndex, 0]] = len(itemDict)
        if trainningData[rowIndex, 1] not in userDict:
            userDict[trainningData[rowIndex, 1]] = len(userDict)

    print("Convert A_orgInputHash to A_orgInputMatrix...")
    A_orgInputMatrix = numpy.full((len(userDict), len(itemDict)), numpy.nan)
    for progressIndex, user in enumerate(A_orgInputHash.keys()):
        printProgressBar(progressIndex + 1, len(A_orgInputHash), delimiter=5, prefix="\tProgress:", suffix="Complete", length=50)
        for item in A_orgInputHash[user].keys():
            A_orgInputMatrix[userDict[user], itemDict[item]] = A_orgInputHash[user][item]

    print("\n#Users:", len(userDict), "#Items: ", len(itemDict))
    del A_orgInputHash

    R_returnInputMatrix, mu_avgOfInputMatrix = matrixFactorization(A_orgInputMatrix)

    with open("OutputFiles/Yi-Chen Liu_preds_matrix.txt", "w", encoding="utf-8") as outputFile:
        for rowIndex in range(len(testingData)):
            if int(testingData[rowIndex, 1]) in userDict and int(testingData[rowIndex, 0]) in itemDict:
                outputFile.write(testingData[rowIndex, 0] + "," + testingData[rowIndex, 1] + "," + str(int(R_returnInputMatrix[userDict[int(testingData[rowIndex, 1])], itemDict[int(testingData[rowIndex, 0])]])) + "," + testingData[rowIndex, 3] + "\n")
            else:
                outputFile.write(testingData[rowIndex, 0] + "," + testingData[rowIndex, 1] + "," + str(mu_avgOfInputMatrix) + "," + testingData[rowIndex, 3] + "\n")
        outputFile.close()

    return


if __name__ == "__main__":
    _unitTest()
