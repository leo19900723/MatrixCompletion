import collections
import math
import numpy
from scipy.optimize import minimize
from scipy.sparse import csgraph
from scipy.cluster.vq import vq, kmeans, whiten
from scipy.spatial import distance


# Print iterations progress
def printProgressBar(iteration, total, prefix="", suffix="", decimals=1, length=100, fill="█", printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def matrixDistance(matrixA, matrixB):
    ans = 0

    def euclideanDistanceWithoutZero(vector1, vector2, power=1):
        ans = 0
        for index in range(len(vector1)):
            if vector1[index] and vector2[index]:
                ans += (vector1[index] - vector2[index]) ** 2

        return math.sqrt(ans) ** power

    for vectorIndex in range(len(matrixA[0])):
        ans += euclideanDistanceWithoutZero(matrixA[:, vectorIndex], matrixB[:, vectorIndex], 2)

    return ans


def l2RegularizedMatrixFactorization(A_orgInputMatrix):
    AShape = A_orgInputMatrix.shape
    lambda_regularizationController = 0

    def objectiveFunction(inputMatrix):
        inputMatrix = numpy.reshape(inputMatrix, AShape)

        mu_avgOfInputMatrix = numpy.average(inputMatrix)
        accuDeviationErr = 0
        accuSVDErr = 0

        U_userToConcept, S_singularValueMatrix, VT_itemToConcept = numpy.linalg.svd(inputMatrix, full_matrices=False)
        V_itemToConcept = VT_itemToConcept.transpose()

        R_returnPrediction = numpy.empty((len(inputMatrix), len(inputMatrix[0])))

        for userIndex in range(len(R_returnPrediction)):
            for itemIndex in range(len(R_returnPrediction[userIndex])):
                bu = numpy.std(inputMatrix[userIndex])
                bi = numpy.std(inputMatrix[:, itemIndex])
                R_returnPrediction[userIndex, itemIndex] = mu_avgOfInputMatrix + bu + bi + V_itemToConcept[itemIndex].dot(U_userToConcept[userIndex])
                accuDeviationErr += bu ** 2 + bi ** 2

        for user_i in range(len(U_userToConcept)):
            for user_j in range(user_i + 1, len(U_userToConcept)):
                accuSVDErr += distance.euclidean(U_userToConcept[user_i], U_userToConcept[user_j])

        for item_i in range(len(VT_itemToConcept)):
            for item_j in range(item_i + 1, len(VT_itemToConcept[item_i])):
                accuSVDErr += distance.euclidean(VT_itemToConcept[item_i], VT_itemToConcept[item_j])

        return matrixDistance(R_returnPrediction, A_orgInputMatrix) + lambda_regularizationController * (accuDeviationErr + accuSVDErr)

    print(objectiveFunction(A_orgInputMatrix))
    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    res = minimize(objectiveFunction, A_orgInputMatrix, method="SLSQP", bounds=[(0, 6)]*AShape[0]*AShape[1])
    return numpy.reshape(res.x, AShape)


def main():
    print("Loading Files...")
    trainningData = numpy.genfromtxt("InputFiles/1.csv", delimiter=",", dtype=int)[1:, :-1]
    testingData = numpy.genfromtxt("InputFiles/1_test.csv", delimiter=",", dtype="U10")
    numOfMovies, numOfUsers = max(trainningData[:, 0]), max(trainningData[:, 1])

    print("Calculating A_orgInputMatrix...")
    A_orgInputMatrix = numpy.zeros((numOfUsers, numOfMovies))
    for userIndex in range(len(trainningData)):
        if not (userIndex % (len(trainningData) // 20)) or userIndex + 1 == len(trainningData):
            printProgressBar(userIndex + 1, len(trainningData), prefix="\tProgress:", suffix="Complete", length=50)
        A_orgInputMatrix[trainningData[userIndex, 1] - 1, trainningData[userIndex, 0] - 1] = trainningData[userIndex, 2]

    R_returnPrediction = l2RegularizedMatrixFactorization(A_orgInputMatrix)
    print(R_returnPrediction)

    '''
    with open("OutputFiles/ans.csv", "w", encoding="utf-8") as outputFile:
        for rowIndex in range(len(testingData)):
            outputFile.write(testingData[rowIndex, 0] + "," + testingData[rowIndex, 1] + "," + str(R_returnPrediction[int(testingData[rowIndex, 1]) - 1, int(testingData[rowIndex, 0]) - 1]) + "," + testingData[rowIndex, 3] + "\n")
        outputFile.close()
    '''

    return


if __name__ == "__main__":
    main()
