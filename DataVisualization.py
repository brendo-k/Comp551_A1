import matplotlib.pyplot as plt
# %matplotlib inline

class DataVisualization():

    @staticmethod
    def displayIonosphereData():
        file = open("CleanDatasets/Ionosphere_Numpy_Array.txt", "r")
        lines = file.readlines()

        plt.title("My Plot")
        #x is 0, 1, 2, 3
        plt.plot([1,2,3,4])
        plt.ylabel('some numbers')
        plt.xlabel('some numbers')
        plt.savefig('a.pdf')
        plt.show()







def main():

    DataVisualization.displayIonosphereData()
    # DataVisualization.displayAdultData()
    # DataVisualization.displayCancerData()
    # DataVisualization.displayOzoneData()


if (__name__ == "__main__"):
    main()
