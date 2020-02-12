import numpy as np

class DataProcessor():

    @staticmethod
    def cleanIonosphereData():
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
        np.savetxt("CleanDatasets/Ionosphere_Numpy_Array.txt", data, fmt='%1.5f')

    @staticmethod
    def cleanAdultData():
        atts = [None]*14
        atts[0] = [] # a0 cont
        atts[1] = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
        atts[2] = [] # a2 cont
        atts[3] = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"] 
        atts[4] = [] # a4 cont
        atts[5] = ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
        atts[6] = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
        atts[7] = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
        atts[8] = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
        atts[9] = ["Female", "Male"]
        atts[10] = [] # a10 cont
        atts[11] = [] # a11 cont
        atts[12] = [] # a12 cont
        atts[13] = ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]
        file = open("Datasets/adult.data", "r")
        lines = file.readlines()
        data = np.empty([48842, 15])
        for i in range(len(lines)):
            features =  lines[i].split(",")
            for j in range(len(features) - 1):
                if not j in [0, 2, 4, 10, 11, 12]:
                    for k in range(len(atts[j])):
                        if atts[j][k]==(features[j]).strip():
                            data[i][j] = k
                else:
                    data[i][j] = features[j]
            if(features[-1].strip() == "<=50K"):
                data[i, -1] = 0
            else:
                data[i, -1] = 1
        data = data[:32561]
        np.savetxt("CleanDatasets/Adult_Numpy_Array.txt", data, fmt='%1.5f')


    # 1,0 | two classes 1: ozone day, 0: normal day
    @staticmethod
    def cleanOzoneData():
        file = open("Datasets/Ozone/onehr.data", "r")
        lines = file.readlines()
        data = np.empty([2536, 73])
        listOfIndex = []

        for i in range(len(lines)):
            features =  lines[i].split(",")

            if ("?" in features) == True:   # you don't populate the data, you insert a flag, -1
                listOfIndex.append(i)
            else:
                for j in range(len(features) - 1):
                    data[i,j] = float(features[j+1])    # ignore first column, useless feature


        
        data = np.delete(data, listOfIndex, 0) # delete the rows missing features
        np.savetxt("CleanDatasets/Ozone_Numpy_Array.txt", data, fmt='%1.5f')


    @staticmethod
    def cleanCancerData():
        file = open("Datasets/Cancer/breast-cancer-wisconsin.data", "r")
        lines = file.readlines()
        data = np.empty([699, 10], dtype=int)
        listOfIndex = []

        for i in range(len(lines)):
            features =  lines[i].split(",")

            if ("?" in features) == True:   # you don't populate the data, you insert flag, -1
                listOfIndex.append(i)
            else:
                for j in range(len(features) - 1):
                    data[i,j] = int(features[j+1])    # ignore first column, useless feature

            if(features[-1][0] == "2"):
                data[i, -1] = 1
            else:
                data[i, -1] = 0


        
        data = np.delete(data, listOfIndex, 0)      # delete the rows missing features
        np.savetxt("CleanDatasets/Cancer_Numpy_Array.txt", data, fmt='%1.5f')

def main():

    DataProcessor.cleanAdultData()
    DataProcessor.cleanIonosphereData()
    DataProcessor.cleanOzoneData()
    DataProcessor.cleanCancerData()




if (__name__ == "__main__"):
    main()

