import numpy as np

def main():
    file = open("Datasets/ionosphere.data", "r")
    lineList = []
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

if (__name__ == "__main__"):
    main()
