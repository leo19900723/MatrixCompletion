import collections
import math
import numpy
from scipy.sparse import csgraph
from scipy.cluster.vq import vq, kmeans, whiten
from scipy.spatial import distance
from Tools import printProgressBar


def spectralClustering(A_orgInputMatrix, K):
    sigma_parameter = 1
    print("Start to proceed Spectral Clustering...")
    AShape = A_orgInputMatrix.shape

    clustersCtoU = collections.defaultdict(lambda: set())
    clustersUtoC = {}

    A_maskedOrgInputMatrix = numpy.ma.masked_array(A_orgInputMatrix, numpy.isnan(A_orgInputMatrix))
    mu_avgOfInputMatrix = numpy.mean(A_maskedOrgInputMatrix)
    R_learningInput = A_maskedOrgInputMatrix.filled(mu_avgOfInputMatrix)

    print("Calculating W_affinityMatrix...")
    W_affinityMatrix = numpy.ones((AShape[0], AShape[0]))
    for user_i in range(AShape[0]):
        printProgressBar(user_i + 1, AShape[0], prefix="\tProgress:", suffix="Complete", length=50)
        for user_j in range(user_i + 1, AShape[0]):
            W_affinityMatrix[user_i, user_j] = math.exp(-1 * distance.euclidean(R_learningInput[user_i, :], R_learningInput[user_j, :]) ** 2 / (sigma_parameter ** 2))
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
    return clustersCtoU, clustersUtoC, int(round(mu_avgOfInputMatrix))


def _unitTest():
    print("Loading Files...")
    trainningData = numpy.genfromtxt("InputFiles/1.csv", delimiter=",", dtype=int)[1:, :-1]
    testingData = numpy.genfromtxt("InputFiles/1_test.csv", delimiter=",", dtype="U10")
    A_orgInputHash = collections.defaultdict(lambda: {})
    clusterCtoI_Mu = collections.defaultdict(lambda: {})
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

    clusterCtoU, clusterUtoC, mu_avgOfInputMatrix = spectralClustering(A_orgInputMatrix, 2)

    print("Writing test result...")
    with open("OutputFiles/Yi-Chen Liu_preds_clustering.txt", "w", encoding="utf-8") as outputFile:
        for rowIndex in range(len(testingData)):
            printProgressBar(rowIndex + 1, len(testingData), delimiter=5, prefix="\tProgress:", suffix="Complete", length=50)
            if int(testingData[rowIndex, 1]) in userDict and int(testingData[rowIndex, 0]) in itemDict:
                if not numpy.isnan(A_orgInputMatrix[userDict[int(testingData[rowIndex, 1])], itemDict[int(testingData[rowIndex, 0])]]):
                    predictRating = int(A_orgInputMatrix[userDict[int(testingData[rowIndex, 1])], itemDict[int(testingData[rowIndex, 0])]])
                elif clusterUtoC[int(testingData[rowIndex, 1])] in clusterCtoI_Mu and int(testingData[rowIndex, 0]) in clusterCtoI_Mu[clusterUtoC[int(testingData[rowIndex, 1])]]:
                    predictRating = clusterCtoI_Mu[clusterUtoC[userDict[int(testingData[rowIndex, 1])]]][itemDict[int(testingData[rowIndex, 0])]]
                else:
                    predictRating = int(round(numpy.nanmean(A_orgInputMatrix[numpy.array(list(clusterCtoU[clusterUtoC[userDict[int(testingData[rowIndex, 1])]]])), itemDict[int(testingData[rowIndex, 0])]])))
                    clusterCtoI_Mu[clusterUtoC[userDict[int(testingData[rowIndex, 1])]]][itemDict[int(testingData[rowIndex, 0])]] = predictRating
                outputFile.write(testingData[rowIndex, 0] + "," + testingData[rowIndex, 1] + "," + str(predictRating) + "," + testingData[rowIndex, 3] + "\n")
            else:
                outputFile.write(testingData[rowIndex, 0] + "," + testingData[rowIndex, 1] + "," + str(mu_avgOfInputMatrix)+ "," + testingData[rowIndex, 3] + "\n")
        outputFile.close()
    return


if __name__ == "__main__":
    _unitTest()
