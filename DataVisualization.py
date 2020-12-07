import matplotlib.pyplot as plt


class DataVisualization:
    
    def __init__(self, data):
        self.data = data

    def scatterPlot(self, feature1, feature2):
        """
        [20 points]
        •	This method plots the relationship between 
            two features, feature1 and feature2 as a scatter plot. 
        •	If the reference for the data is set, this 
            method colors the scattered points using
            the underlying reference array.
        •	The color scheme should be set to “jet”.
        """
        plt.figure(figsize=(7,7))
        if self.data.getReference() is None:
            plt.scatter(self.data.dataDict[feature1], self.data.dataDict[feature2], cmap='jet')
        else:
            plt.scatter(self.data.dataDict[feature1], self.data.dataDict[feature2], cmap='jet', c=self.data.dataDict[self.data.getReference()])

        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title('scatter plot')
        plt.show()
