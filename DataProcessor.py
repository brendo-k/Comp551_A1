import numpy as np

def main():
    file = open("Datasets/ionosphere.data", "r")
    lines = file.readlines()
    data = np.empty([351, 35])
    for i in range(len(lines)-1):
        features =  lines[i].split(",")
        for j in range(len(features) - 2):
            data[i,j] = float(features[j])
        if(features[-1][0] == "g"):
            data[i, -1] = 1
        else:
            data[i, -1] = 0
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
    file = open("Datasets/adult.data", "r")
    lines = file.readlines()
    data = np.empty([48842, 14], str)        ## initialize stuff to int by default?
    for i in range(len(lines)):
        features =  lines[i].split(",")
        for j in range(len(features) - 1):
            data[i,j] = features[j]
    np.savetxt("Adult_Numpy_Array.txt", data, fmt='%1.5f')
    print(data)

if (__name__ == "__main__"):
    main()
    