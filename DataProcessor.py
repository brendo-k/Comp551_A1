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
        #print(data[-1])
        np.savetxt("Ionosphere_Numpy_Array.txt", data, fmt='%1.5f')
        #print(data)




    @staticmethod
    def cleanAdultData():
        # Use one-hot encoding for categorical features | 6 continuous, 8 nominal      
        workclass = ["Private", "Self-emp-not-inc", "Self-emp-inc", 
        "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
        education = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", 
        "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]
        maritalStatus = ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
        occupation = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", 
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
        relationship = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
        race = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
        sex = ["Female", "Male"]
        nativeCountry = ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", 
        "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", 
        "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France","Dominican-Republic", 
        "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", 
        "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]

        #workclass_to_int = dict((c, i) for i, c in enumerate(workclass))
        #int_to_workclass = dict((i, c) for i, c in enumerate(workclass))


        file = open("Datasets/adult.data", "r")
        lines = file.readlines()
        data = np.zeros([48842, 6 + len(workclass) + len(education) + len(maritalStatus) + len(occupation)
        + len(relationship) + len(race) + len(sex) + len(nativeCountry) + 1])        

        print(len(data[0]))
        for i in range(len(lines)):
            features = lines[i].split(",")
            if ("?" in features) == True:   # you don't populate the data, you leave it with zeros
                break
            else:
                for j in range(len(features) - 1):
                    if features[j].isdigit() == True:   # numerical data
                        data[i,j] = features[j]
                    else:                               # categorical data
                        if j == 1:     # workclass
                            index = np.where(workclass == features[j])         # index of the feature in the one hot matrix
                            # one_hotFeature = np.zeros([len(workclass), 0])
                            # one_hotFeature[index] += 1
                            # print(len(one_hotFeature))
                            data[i,j + index[0]] += 1
                        
                        if j == 3:      # education
                            index = np.where(education == features[j])
                            data[i,j + len(workclass) + index[0]] += 1



                    
                if(features[-1][0] == ">50K"):
                    data[i, -1] = 1
                    print("here")
                else:
                    data[i, -1] = 0

        # remove all rows containing just zeros: data[~np.all(data == 0, axis=1)]

        # np.savetxt("Adult_Numpy_Array.txt", data) prints a too large file 
        print(data)


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
        np.savetxt("Ozone_Numpy_Array.txt", data, fmt='%1.5f')
        #print(data)


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
        np.savetxt("Cancer_Numpy_Array.txt", data, fmt='%1.5f')

def main():

    #DataProcessor.cleanIonosphereData()
    #DataProcessor.cleanAdultData()
    #DataProcessor.cleanOzoneData()
    #DataProcessor.cleanCancerData()




if (__name__ == "__main__"):
    main()
    
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

