import collections
import math
import numpy
from scipy.sparse import csgraph
from scipy.cluster.vq import vq, kmeans, whiten
from scipy.spatial import distance
import matplotlib.pyplot as plt


# Print iterations progress
def printProgressBar(iteration, total, prefix="", suffix="", decimals=1, length=100, fill="█", printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def euclideanDistanceWithoutZero(vector1, vector2, power=1):
    ans = 0
    for index in range(len(vector1)):
        if vector1[index] and vector2[index]:
            ans += (vector1[index] - vector2[index]) ** 2

    return math.sqrt(ans) ** power


def averageWithoutZero(vector):
    ans = 0
    length = 0
    for item in vector:
        if item:
            ans += item
            length += 1

    return (ans / length) if length else math.inf


def spectralClustering(A_orgInputMatrix, K):
    sigma_parameter = 1
    clustersCtoU = collections.defaultdict(lambda: set())
    clustersUtoC = {}
    R_returnPrediction = numpy.copy(A_orgInputMatrix)

    print("Calculating W_affinityMatrix...")
    W_affinityMatrix = numpy.ones((len(A_orgInputMatrix), len(A_orgInputMatrix)))
    for user_i in range(len(A_orgInputMatrix)):
        if not (user_i % (len(A_orgInputMatrix) // 20)) or user_i + 1 == len(A_orgInputMatrix):
            printProgressBar(user_i + 1, len(A_orgInputMatrix), prefix="\tProgress:", suffix="Complete", length=50)
        for user_j in range(user_i + 1, len(A_orgInputMatrix)):
            W_affinityMatrix[user_i, user_j] = numpy.exp(-1 * euclideanDistanceWithoutZero(A_orgInputMatrix[user_i, :], A_orgInputMatrix[user_j, :], 2) ** 2 / (sigma_parameter ** 2))
            W_affinityMatrix[user_j, user_i] = W_affinityMatrix[user_i, user_j]

    print("Calculating L_laplacianMatrix...")
    L_laplacianMatrix = csgraph.laplacian(W_affinityMatrix)

    print("Calculating L_eigenvalues & L_eigenvectors...")
    L_eigenvalues, L_eigenvectors = numpy.linalg.eig(L_laplacianMatrix)
    L_eigenvaluesOrderedIndex = numpy.argsort(L_eigenvalues)

    print("\tEigenvalues: ", sorted(L_eigenvalues))
    centroids, distortion = kmeans(L_eigenvectors[:, L_eigenvaluesOrderedIndex[:K]], K)

    print("Assigning Clusters...")
    for userIndex in range(len(L_eigenvectors)):
        assignedCluster = [0, math.inf]
        for centroidIndex in range(len(centroids)):
            currDist = distance.euclidean(L_eigenvectors[userIndex, L_eigenvaluesOrderedIndex[:K]], centroids[centroidIndex]) ** 2
            if currDist < assignedCluster[1]:
                assignedCluster[0] = centroidIndex
                assignedCluster[1] = currDist
        clustersCtoU[assignedCluster[0]].add(userIndex)
        clustersUtoC[userIndex] = assignedCluster[0]

    print("\tAssigned Clusters: ", dict(clustersCtoU))

    print("Calculating R_returnPrediction...")
    for userIndex in range(len(R_returnPrediction)):
        for itemIndex in range(len(R_returnPrediction[userIndex])):
            if R_returnPrediction[userIndex, itemIndex] == 0:
                R_returnPrediction[userIndex, itemIndex] = averageWithoutZero(R_returnPrediction[list(clustersCtoU[clustersUtoC[userIndex]]), itemIndex])

    print("Done")
    return R_returnPrediction


def main():
    print("Loading Files...")
    trainningData = numpy.genfromtxt("InputFiles/train.csv", delimiter=",", dtype=int)[1:, :-1]
    testingData = numpy.genfromtxt("InputFiles/test.csv", delimiter=",", dtype="U10")
    numOfMovies, numOfUsers = max(trainningData[:, 0]), max(trainningData[:, 1])

    print("Calculating A_orgInputMatrix...")
    A_orgInputMatrix = numpy.zeros((numOfUsers, numOfMovies))
    for userIndex in range(len(trainningData)):
        if not (userIndex % (len(trainningData) // 20)) or userIndex + 1 == len(trainningData):
            printProgressBar(userIndex + 1, len(trainningData), prefix="\tProgress:", suffix="Complete", length=50)
        A_orgInputMatrix[trainningData[userIndex, 1] - 1, trainningData[userIndex, 0] - 1] = trainningData[userIndex, 2]

    R_returnPrediction = spectralClustering(A_orgInputMatrix, 2)

    with open("OutputFiles/Yi-Chen Liu_preds_clustering.txt", "w", encoding="utf-8") as outputFile:
        for rowIndex in range(len(testingData)):
            outputFile.write(testingData[rowIndex, 0] + "," + testingData[rowIndex, 1] + "," + str(R_returnPrediction[int(testingData[rowIndex, 1]) - 1, int(testingData[rowIndex, 0]) - 1]) + "," + testingData[rowIndex, 3] + "\n")
        outputFile.close()

    return


if __name__ == "__main__":
    main()
