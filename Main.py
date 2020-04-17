from UnnormalizedSpectralClustering import *
from l2RegularizedMatrixFactorization import *


def printProgressBar(iteration, total, delimiter=None, prefix="", suffix="", decimals=1, length=100, fill="█", printEnd="\r"):
    if iteration == total or delimiter is None or iteration % (total * delimiter // 100) == 0:
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + "-" * (length - filledLength)
        print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end=printEnd)
        # Print New Line on Complete
        if iteration == total:
            print()


def main():
    print("Loading Files...")
    trainningData = numpy.genfromtxt("InputFiles/1.csv", delimiter=",", dtype=int)[1:, :-1]
    testingData = numpy.genfromtxt("InputFiles/1_test.csv", delimiter=",", dtype="U10")
    A_orgInputHash = collections.defaultdict(lambda: {})
    userDict, itemDict = {}, {}

    print("Loading the input file to A_orgInputHash...")
    for rowIndex in range(len(trainningData)):
        printProgressBar(rowIndex + 1, len(trainningData), delimiter=5, prefix="\tProgress:", suffix="Complete", length=50)
        A_orgInputHash[trainningData[rowIndex, 1]][trainningData[rowIndex, 0]] = trainningData[rowIndex, 2]
        if trainningData[rowIndex, 1] not in userDict:
            userDict[trainningData[rowIndex, 1]] = len(userDict)
        if trainningData[rowIndex, 0] not in itemDict:
            itemDict[trainningData[rowIndex, 0]] = len(itemDict)

    print("Convert A_orgInputHash to A_orgInputMatrix...")
    A_orgInputMatrix = numpy.zeros((len(userDict), len(itemDict)))
    for progressIndex, user in enumerate(A_orgInputHash.keys()):
        printProgressBar(progressIndex + 1, len(A_orgInputHash), delimiter=5, prefix="\tProgress:", suffix="Complete", length=50)
        for item in A_orgInputHash[user].keys():
            A_orgInputMatrix[userDict[user], itemDict[item]] = A_orgInputHash[user][item]

    print("\n#Users:", len(A_orgInputMatrix), "#Items: ", len(A_orgInputMatrix[0]))
    del A_orgInputHash

    def outputResult(fileName, resultMatrix):
        with open("OutputFiles/" + fileName, "w", encoding="utf-8") as outputFile:
            mu = str(int(round(numpy.average(resultMatrix))))
            for rowIndex in range(len(testingData)):
                if int(testingData[rowIndex, 1]) in userDict and int(testingData[rowIndex, 0]) in itemDict:
                    outputFile.write(testingData[rowIndex, 0] + "," + testingData[rowIndex, 1] + "," + str(int(round(resultMatrix[userDict[int(testingData[rowIndex, 1])], itemDict[int(testingData[rowIndex, 0])]]))) + "," + testingData[rowIndex, 3] + "\n")
                else:
                    outputFile.write(testingData[rowIndex, 0] + "," + testingData[rowIndex, 1] + "," + mu + "," + testingData[rowIndex, 3] + "\n")
            outputFile.close()
        return

    methodChoice = input("Choose your method:\n1. Spectral Clustering\n2. L2-Regularized Matrix Factorization\n3. All of above\n")
    if methodChoice in {"1", "3"}:
        R_clustering = spectralClustering(A_orgInputMatrix, 2)
        outputResult("Yi-Chen Liu_preds_clustering.txt", R_clustering)

    if methodChoice in {"2", "3"}:
        R_matrix = l2RegularizedMatrixFactorization(A_orgInputMatrix)
        outputResult("Yi-Chen Liu_preds_matrix.txt", R_matrix)

    return


if __name__ == "__main__":
    main()
