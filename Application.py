from Data import Data
from Classification import kNN


def main():
    
    # load the datasets    
    # Update the paths of the datasets based on your directory structure
    inputFile1 = 'iris_train.csv'
    inputFile2 = 'iris_test.csv'    
    trainData = Data(inputFile=inputFile1, reference='Species')
    testData = Data(inputFile=inputFile2, reference='Species')
        
    
    
    pl_accuracy, pw_accuracy, sl_accuracy, sw_accuracy = findIndividualAccuracies(trainData,testData)
        
    min_feature, max_feature = findMinMaxFeatures(pl_accuracy, pw_accuracy, sl_accuracy, sw_accuracy)
    
    knn_predictions,final_accuracy = findOveralAccuracy(trainData, testData)
        
    visualizePredictions(testData,knn_predictions)
    
    print('Prediction accuracy using Petal length = ',round(pl_accuracy,3))
    print('Prediction accuracy using Petal width = ',round(pw_accuracy,3))
    print('Prediction accuracy using Sepal length = ',round(sl_accuracy,3))
    print('Prediction accuracy using Sepal width = ',round(sw_accuracy,3))
    print(min_feature,'has the minimum and',max_feature,'has the maximum accuracy')
    print('Prediction accuracy using all the features = ',round(final_accuracy,3))
        

def compute_accuracy(train_label, test_label):
    match = 0
    total = len(train_label)
    for i in range(total):
        if train_label[i] == test_label[i]:
            match += 1
    return match*100.0/total


def findIndividualAccuracies(trainData, testData):
    """
    Find the accuracy of predicting species 
    in iris_test.csv using 
    a. Petal length
    b. Petal width
    c. Sepal length
    d. Sepal width
    """
    accuracies = []
    features = trainData.header
    if trainData.reference and trainData.reference in features:
        features.remove(trainData.reference)
    for feature in features:
        kNNClassifier = kNN(trainData, features=[feature])
        prediction = kNNClassifier.classify(testData.getNumpyMatrix([feature]), 5)
        actual = testData.dataDict[testData.getReference()]
        accuracies.append(compute_accuracy(actual, prediction))
    return accuracies


def findMinMaxFeatures(pl_accuracy, pw_accuracy, sl_accuracy, sw_accuracy):
    """
    Which features have the best and the worst accuracies?
    """
    return min([pl_accuracy, pw_accuracy, sl_accuracy, sw_accuracy]), max([pl_accuracy, pw_accuracy, sl_accuracy, sw_accuracy])


def findOveralAccuracy(trainData,testData):
    """
    What is the accuracy of predicting species in iris_test.csv 
    using all four features? 
    """
    features = trainData.header
    if trainData.reference and trainData.reference in features:
        features.remove(trainData.reference)
    kNNClassifier = kNN(trainData, features=features)
    prediction = kNNClassifier.classify(testData.getNumpyMatrix(), 5)
    actual = testData.dataDict[testData.getReference()]
    return prediction, compute_accuracy(actual, prediction)


def visualizePredictions(testData, knn_predictions):
    """
    Visualize Petal length vs. Petal width
    """
    testData.visualize.scatterPlot('Petal length', 'Petal width')


if __name__ == "__main__":
    main()

