from Data import Data
def main():
    
    # load the datasets    
    # Update the paths of the datasets based on your directory structure
    inputFile1 = 'iris_train.csv'
    inputFile2 = 'iris_test.csv'    
    trainData = Data(inputFile=inputFile1,reference='Species')
    testData = Data(inputFile=inputFile2)    
        
    
    
    pl_accuracy, pw_accuracy, sl_accuracy, sw_accuracy = findIndividualAccuracies(trainData,testData)
        
    min_feature, max_feature = findMinMaxFeatures(pl_accuracy, pw_accuracy, sl_accuracy, sw_accuracy)
    
    knn_predictions,final_accuracy = findOveralAccuracy(trainData,testData)
        
    visualizePredictions(testData,knn_predictions)
    
    print('Prediction accuracy using Petal length = ',round(pl_accuracy,3))
    print('Prediction accuracy using Petal width = ',round(pw_accuracy,3))
    print('Prediction accuracy using Sepal length = ',round(sl_accuracy,3))
    print('Prediction accuracy using Sepal width = ',round(sw_accuracy,3))
    print(min_feature,'has the minimum and',max_feature,'has the maximum accuracy')
    print('Prediction accuracy using all the features = ',round(final_accuracy,3))
        


def findIndividualAccuracies(trainData,testData):
    """
    Find the accuracy of predicting species 
    in iris_test.csv using 
    a. Petal length
    b. Petal width
    c. Sepal length
    d. Sepal width
    """
    pass

def findMinMaxFeatures(pl_accuracy, pw_accuracy, sl_accuracy, sw_accuracy):
    """
    Which features have the best and the worst accuracies?
    """
    pass
def findOveralAccuracy(trainData,testData):
    """
    What is the accuracy of predicting species in iris_test.csv 
    using all four features? 
    """
    pass

def visualizePredictions(testData,knn_predictions):
    """
    Visualize Petal length vs. Petal width
    """
    pass

if __name__ == "__main__":main()
