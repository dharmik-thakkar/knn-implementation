import csv
from DataVisualization import DataVisualization
import numpy as np


class Data:
    def __init__(self,inputFile=None, reference=None):
        """        
        Inputs
            •	inputFile: Name of the input file. Default value is None
            •	reference: Name of the reference column. Default value is None. 
        Local variables set:
            •	dataDict
            •	header
            •	reference
            •	stats 
            •	visualize
        Outputs: None
        """
        # Check if inputFile is None
        if inputFile is None:
            # Initialize all the variables to None
            self.dataDict = None            
            self.reference = None     
            self.header = None
            self.numColumns = None
            self.numRows = None
            self.inputFile = None
        else:
            # else call the load function
            self.load(inputFile,reference)
        
        # initialize DataStatistics and DataVisualization objects        
        self.visualize = DataVisualization(self)

        
    def load(self, inputFile, reference=None):
        """        
        Inputs
            •	inputFile: Name of the input file. 
            •	reference: Name of the reference column. Default value is None. 
        Local variables set:
            •	dataDict
            •	header
            •	reference
        Outputs: None
        """
        # Set values of inputFile and reference
        self.inputFile = inputFile
        self.reference = reference        
        # Initialize header data and columnDict
        headerData = []              
        columnDict = {}
    
        # Read csv file
        with open(inputFile) as csvFile:
            csvReader = csv.reader(csvFile)
            # Iterate through all the rows in the csv file
            for i, row in enumerate(csvReader):
                # Header row
                if i == 0:
                    headerData = row
                    for header in headerData:
                        columnDict[header] = []
                
                # Remaining rows
                else:
                    for j, data in enumerate(row):
                        try:
                            columnDict[headerData[j]].append(float(data))
                        except:
                            # For strings 
                            columnDict[headerData[j]].append(data)                                     
        
        # Set the class instance variables
        self.dataDict = columnDict
        self.header = headerData
        self.numColumns = len(headerData)
        self.numRows = len(columnDict[headerData[0]])
    
    def getNumpyMatrix(self, features=None):
        """        
        This function computes a numpy matrix by concatenating 
        the columns corresponding to the features provided by 
        the user as input. If the user provides no input or None, 
        the function computes the numpy matrix using all the features.
        Note: The output should not include the reference column

        Inputs
            •	features: List of features to compute the 
                corresponding numpy matrix. The default value is None.         
        Output 
            •	A numpy matrix computed by concatenating the 
                columns represented by the input variable, features.
        """
        # dataDict = self.dataDict
        # reference = self.reference
        # if features is None, use all the features
        if not features:
            features = self.header
        if self.reference is not None:
            features.remove(self.reference)
        numpy_matrix = []
        data_size = len(self.dataDict[features[0]])
        for row_id in range(data_size):
            matrix_row = []
            for feature in features:
                matrix_row.append(self.dataDict[feature][row_id])
            numpy_matrix.append(matrix_row)
        return numpy_matrix
        # XList = []
        # # Iterate through all the features
        # for i in dataDict:
        #     # Skip reference
        #     if i == reference:
        #         continue
        #     elif i in features:
        #         # Append the list
        #         XList.append(dataDict[i])
        #
        # X = np.array(XList)
        #
        # # Return transpose to ensure matrix shape is same as
        # # shape of data in the csv file
        # return X.T
    
    def getReference(self):
        return self.reference
    
    
    def setReference(self,reference):
        """
        Inputs
            •	reference: Name of the reference column. The default value is None. 
        Local variables set:
            •	reference
        Outputs: None
        """
        if reference in self.header:
            self.reference = reference
        else:
            print(reference + 'does not exist in the dataset')

inputFile = 'ExampleData.csv'
breastData = Data()
breastData.load(inputFile)
print (breastData.getNumpyMatrix())