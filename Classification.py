import numpy as np
class kNN:    
    
    def __init__(self,trainData, features=None):
        """
        Initialize the values of trainX, labels and features used to fit
        the kNN classifier
        """
        self.trainX = trainData.getNumpyMatrix(features)
        self.labels = np.array(trainData.dataDict[trainData.reference])
        self.features = features
        
    
    
    def computeDistanceMatrix(self,testX):
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
        pass
    
        
    def classify(self,testData,k):   
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
        pass
    
    
        