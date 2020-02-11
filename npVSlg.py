import NaiveBayes as nb 
import LogisticRegression as lr 

def main():
    file = open("CleanDatasets/Cancer_Numpy_Array.txt", "r")
    cancer_data = np.loadtxt(file) # data loaded into numpy array
    
    file = open("CleanDatasets/Ionosphere_Numpy_Array.txt", "r")
    ionosphere_data = np.loadtxt(file) # data loaded into numpy array
    ionosphere_data = np.delete(ionosphere_data, 1, 1) # removing 0 columns
    ionosphere_data = ionosphere_data[:, 1:]
    
    file = open("CleanDatasets/Ozone_Numpy_Array.txt", "r")
    ozone_data = np.loadtxt(file) # data loaded into numpy array
    
    file = open("CleanDatasets/Adult_Numpy_Array.txt", "r")
    adult_data = np.loadtxt(file) # data loaded into numpy array
    unbiased_adult_data = np.delete(adult_data, 8, 1) # removing gender
    unbiased_adult_data = np.delete(unbiased_adult_data, 8, 1) # removing race
    
    
    

