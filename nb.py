import numpy as np

# as of right now, this only uploads ionosphere,               #
# as this will be the first dataset i try using naive bayes on #


def main():
    file = file = open("Ionosphere_Numpy_Array.txt", "r")
    array = np.loadtxt(file) # data loaded into numpy array


if (__name__ == "__main__"):
    main() # running main 