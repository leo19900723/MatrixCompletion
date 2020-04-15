import collections
import math
import numpy
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


def l2RegularizedMatrixFactorization(A_orgInputMatrix):
    return


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

    R_returnPrediction = l2RegularizedMatrixFactorization(A_orgInputMatrix, 2)

    with open("OutputFiles/ans.csv", "w", encoding="utf-8") as outputFile:
        for rowIndex in range(len(testingData)):
            outputFile.write(testingData[rowIndex, 0] + "," + testingData[rowIndex, 1] + "," + str(R_returnPrediction[int(testingData[rowIndex, 1]) - 1, int(testingData[rowIndex, 0]) - 1]) + "," + testingData[rowIndex, 3] + "\n")
        outputFile.close()

    return


if __name__ == "__main__":
    main()