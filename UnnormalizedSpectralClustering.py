import collections
import math
import numpy
from scipy.sparse import csgraph
from scipy.cluster.vq import vq, kmeans, whiten
from scipy.spatial import distance
import matplotlib.pyplot as plt


def spectralClustering(A_orgInputMatrix, K):
    sigma_parameter = 1
    clusters = collections.defaultdict(lambda: set())

    W_affinityMatrix = numpy.ones((len(A_orgInputMatrix), len(A_orgInputMatrix)))
    for user_i in range(len(A_orgInputMatrix)):
        for user_j in range(user_i + 1, len(A_orgInputMatrix)):
            W_affinityMatrix[user_i, user_j] = numpy.exp(-1 * distance.euclidean(A_orgInputMatrix[user_i, :], A_orgInputMatrix[user_j, :]) ** 2 / (sigma_parameter ** 2))
            W_affinityMatrix[user_j, user_i] = W_affinityMatrix[user_i, user_j]

    L_laplacianMatrix = csgraph.laplacian(W_affinityMatrix)

    L_eigenvalues, L_eigenvectors = numpy.linalg.eig(L_laplacianMatrix)
    L_eigenvaluesOrderedIndex = numpy.argsort(L_eigenvalues)

    centroids, distortion = kmeans(L_eigenvectors[:, L_eigenvaluesOrderedIndex[:K]], K)

    test = collections.defaultdict(lambda: [])
    for userIndex in range(len(L_eigenvectors)):
        assignedCluster = [0, math.inf]
        for centroidIndex in range(len(centroids)):
            currDist = distance.euclidean(L_eigenvectors[userIndex, L_eigenvaluesOrderedIndex[:K]], centroids[centroidIndex])
            test[userIndex].append(currDist)
            if currDist < assignedCluster[1]:
                assignedCluster[0] = centroidIndex
                assignedCluster[1] = currDist
        clusters[assignedCluster[0]].add(userIndex)

    print(clusters)

    plt.scatter(L_eigenvectors.transpose()[:, 0], L_eigenvectors.transpose()[:, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="*", c="r")
    plt.show()

    return


def main():
    trainningData = numpy.genfromtxt("InputFiles/1.csv", delimiter=",", dtype=int)[1:, :-1]
    numOfMovies, numOfUsers = max(trainningData[:, 0]), max(trainningData[:, 1])

    A_orgInputMatrix = numpy.zeros((numOfUsers, numOfMovies))
    for row in trainningData:
        A_orgInputMatrix[row[1] - 1, row[0] - 1] = row[2]

    spectralClustering(A_orgInputMatrix, 3)

    return


if __name__ == "__main__":
    main()
