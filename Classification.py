import math
import numpy as np
from Data import Data


class kNN:
    
    def __init__(self, trainData, features=None):
        """
        Initialize the values of trainX, labels and features used to fit
        the kNN classifier
        """
        self.traindata = trainData
        self.trainX = trainData.getNumpyMatrix(list(features))
        reference = trainData.reference
        if reference:
            self.labels = np.array(trainData.dataDict[reference])
        else:
            self.labels = None
        self.features = features

    def _euclidian_distance(self, p1, p2):
        """
        Computes euclidian distance between given two points
        """
        dim, sum_ = len(p1), 0
        for index in range(dim - 1):
            sum_ += math.pow(p1[index] - p2[index], 2)
        return math.sqrt(sum_)

    def computeDistanceMatrix(self, testX):
        """
        [20 points]
        This method computes pairwise distance between
        every row in test data (testX) and every row 
        in the training data.
        Inputs
            •	testX: numpy matrix corresponding to the 
                test data object. 
        Outputs
            •	Distance matrix, D representing pairwise distance
                between every row in testX to every row in trainX.
        
        The element D(i,j) in the resultant distance matrix, 
        D would correspond to the distance between ith row 
        in testX and jth row in trainX.

        """
        distance_matrix = np.empty(shape=(len(testX), len(self.trainX)))
        for i, m in enumerate(testX):
            for j, n in enumerate(self.trainX):
                dist = self._euclidian_distance(m, n)
                distance_matrix[i][j] = dist
        return distance_matrix
        
    def classify(self, testData, k):
        """
        [30 points]
        This method computes prediction for every row in the test data.
        Hints: 
            •	Refer the pseudocode provided
            •	Call the computeDistanceMatrix method from this method
            •	List of useful numpy matrix operations 
                o	numpy.argsort
                o	numpy.unique
                o	numpy.sum
                o	numpy.argmax
        Inputs
            •	testData: data object corresponding to the test data. 
            •	k: Number of nearest neighbors to consider for prediction
        Outputs
            •	kNN based predictions for each row in the test data.

        """
        distance_matrix = self.computeDistanceMatrix(testData)
        k_neighbors = np.argsort(distance_matrix)[:, :k]
        k_shape = k_neighbors.shape
        k_labels = np.empty_like(k_neighbors)
        test_labels = []
        for i in range(k_shape[0]):
            for j in range(k_shape[1]):
                k_labels[i][j] = self.traindata.dataDict[self.traindata.reference][k_neighbors[i][j]]
            test_labels.append(np.bincount(k_labels[i]).argmax())
        return test_labels


# inputFile1 = 'trainData.csv'
# inputFile2 = 'testData.csv'
# trainData = Data(inputFile=inputFile1, reference='diagnosis')
# testData = Data(inputFile=inputFile2, reference='diagnosis')
# kNNClassifier = kNN(trainData)
# # D = kNNClassifier.computeDistanceMatrix(testData.getNumpyMatrix())
# predictions=kNNClassifier.classify(testData.getNumpyMatrix(), 5)
# print(predictions)
