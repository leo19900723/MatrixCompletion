import math
import numpy
from scipy.optimize import minimize
from scipy.sparse import csgraph
from scipy.cluster.vq import vq, kmeans, whiten
from scipy.spatial import distance


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
    print("L2-Regularized Matrix Factorization...")
    AShape = A_orgInputMatrix.shape
    mu_avgOfInputMatrix = numpy.average(numpy.nonzero(A_orgInputMatrix))
    lambda_regularizationController = 0

    def objectiveFunction(inputParameters):
        inputMatrix = numpy.reshape(inputParameters[AShape[0] + AShape[1] + 1:], AShape)

        U_userToConcept, S_singularValueMatrix, VT_itemToConcept = numpy.linalg.svd(inputMatrix, full_matrices=False)

        bu = numpy.sum(A_orgInputMatrix - U_userToConcept @ VT_itemToConcept - numpy.array([inputParameters[AShape[0] + 1: AShape[0] + AShape[1] + 1], ] * AShape[0]), axis=1)
        bi = numpy.sum(A_orgInputMatrix - U_userToConcept @ VT_itemToConcept - numpy.array([inputParameters[1: AShape[0] + 1], ] * AShape[1]).transpose(), axis=0)
        accuDeviationErr = numpy.sum(numpy.power(bu, 2)) + numpy.sum(numpy.power(bi, 2))
        accuSVDErr = numpy.sum(numpy.power(numpy.linalg.norm(U_userToConcept, axis=0), 2)) + numpy.sum(numpy.power(numpy.linalg.norm(VT_itemToConcept, axis=0), 2))

        R_returnPrediction = numpy.full(AShape, mu_avgOfInputMatrix) + numpy.array([bu, ]*AShape[1]).transpose() + numpy.array([bi, ]*AShape[0]) + U_userToConcept @ VT_itemToConcept

        return matrixDistance(R_returnPrediction, A_orgInputMatrix) + inputParameters[0] * (accuDeviationErr + accuSVDErr)

    print("Start to optimize the model...")
    inputHead = [lambda_regularizationController] + [0] * (AShape[0] + AShape[1])
    res = minimize(objectiveFunction, numpy.append(inputHead, numpy.ndarray.flatten(A_orgInputMatrix)), method="TNC", bounds=[None] * (AShape[0] + AShape[1] + 1) + [(1, 5)]*AShape[0]*AShape[1])
    print("Done!, Lambda: ", res.x[0], ", Bias table: ", res.x[:AShape[0] + AShape[1] + 1])
    return numpy.reshape(res.x[AShape[0] + AShape[1] + 1:], AShape)


def _unitTest():
    print("Loading Files...")
    trainningData = numpy.genfromtxt("InputFiles/1.csv", delimiter=",", dtype=int)[1:, :-1]
    numOfMovies, numOfUsers = max(trainningData[:, 0]), max(trainningData[:, 1])

    print("Calculating A_orgInputMatrix...")
    A_orgInputMatrix = numpy.zeros((numOfUsers, numOfMovies))
    for userIndex in range(len(trainningData)):
        A_orgInputMatrix[trainningData[userIndex, 1] - 1, trainningData[userIndex, 0] - 1] = trainningData[userIndex, 2]

    print(l2RegularizedMatrixFactorization(A_orgInputMatrix))

    return


if __name__ == "__main__":
    _unitTest()
