import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from NaiveBayes import NaiveBayes
import LogisticRegression

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
    
    nbaccs = [None]*4
    lraccs = [None]*4
    
    nbaccs[0] = np.average(np.around(NaiveBayes.eval(cancer_data, 5)[0], 3))
    nbaccs[1] = np.average(np.around(NaiveBayes.eval(ionosphere_data, 5)[0], 3))
    nbaccs[2] = np.average(np.around(NaiveBayes.eval(ozone_data, 5)[0], 3))
    nbaccs[3] = np.average(np.around(NaiveBayes.eval(adult_data, 5)[0], 3))


    lraccs[0] = 100*LogisticRegression.calculateAccuracy(cancer_data)
    lraccs[1] = 100*LogisticRegression.calculateAccuracy(ionosphere_data)
    lraccs[2] = 100*LogisticRegression.calculateAccuracy(ozone_data)
    lraccs[3] = 100*LogisticRegression.calculateAccuracy(adult_data)
    
    plt.title("Logistic Regression vs. Naive Bayes")
    plt.plot(['Cancer Dataset', 'Ionosphere Dataset', 'Ozone Dataset', 'Adult Dataset'], nbaccs, 'b-o')
    plt.plot(['Cancer Dataset', 'Ionosphere Dataset', 'Ozone Dataset', 'Adult Dataset'], lraccs, 'ro-')
    plt.ylabel('Accuracy (%) ')
    plt.xlabel("Datasets")
    plt.ylim(0, 100)     # set the ylim to bottom, top
    red_patch = mpatches.Patch(color='red', label='Logistic Regression')
    blue_patch = mpatches.Patch(color='blue', label='Naive Bayes')
    plt.legend(handles=[red_patch, blue_patch])
    #plt.savefig('a.pdf')
    plt.show()
    
if __name__ == "__main__":
    main()
    
    
    
    

