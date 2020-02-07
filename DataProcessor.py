import numpy as np

def main():
    
    file = open("Datasets/ionosphere.data", "r")
    lines = file.readlines()
    data = np.empty([351, 35])
    for i in range(len(lines)):
        features =  lines[i].split(",")
        for j in range(len(features) - 1):
            data[i,j] = float(features[j])
        if(features[-1][0] == "g"):
            data[i, -1] = 1
        else:
            data[i, -1] = 0
    print(data[-1])
    np.savetxt("Ionosphere_Numpy_Array.txt", data, fmt='%1.5f')
    print(data)
##################################################

###################################################
    # file1 = open("Datasets/Chess/krkopt.data", "r")
    # lines1 = file1.readlines()
    # data1 = np.empty([28056, 6])
    # for i in range(len(lines1)-1):
    #     features1 =  lines1[i].split(",")
    #     for j in range(len(features1) - 1):
    #        if j%2==0:
    #            data1[i,j] = (features1[j])
    #           data1[i,j] = (features1[j])
    # np.savetxt("Chess_Numpy_Array.txt", data1, fmt='%1.5f')




##################################################################


    #Use one-hot encoding for categorical features      
    workclass = ["Private", "Self-emp-not-inc", "Self-emp-inc", 
    "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
    education = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", 
    "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]
    maritalStatus = ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
    occupation = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", 
    "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
    relationship = []
    race = []
    sex = []
    nativeCountry = []


    file = open("Datasets/adult.data", "r")
    lines = file.readlines()
    data = np.empty([48842, 15], str)        # initializes stuff to numbers by default, need to explicitly typecast to string
    for i in range(len(lines)):
        features =  str(lines[i].split(","))
        for j in range(len(features) - 1):
            data[i,j] = features[j]
    np.savetxt("Adult_Numpy_Array.txt", data)
    print(data)

if (__name__ == "__main__"):
    main()
    