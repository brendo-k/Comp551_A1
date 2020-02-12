import matplotlib.pyplot as plt


class DataVisualization():

    @staticmethod
    def displayIonosphereData():
        xMin = -1
        xMax = 1

        file = open("CleanDatasets/Ionosphere_Numpy_Array.txt", "r")
        data = file.readlines()

        with open("CleanDatasets/Ionosphere_Numpy_Array.txt") as file:
            lines = file.readlines()
            x = [line.split()[0] for line in lines]
            y = [line.split()[1] for line in lines]


        # display the distribution of positive and negative classes
        plt.title("Distribution of classes")
        #x is 0, 1, 2, 3
        plt.plot([0, 1], data[:][-1])
        # plt.axis([0, 1])
        plt.xlabel("Classes")
        plt.ylabel('Instancses')
        plt.savefig("BasicStatistics/classes.pdf")
        plt.show()


        file.close()




def main():

    DataVisualization.displayIonosphereData()
    # DataVisualization.displayAdultData()
    # DataVisualization.displayCancerData()
    # DataVisualization.displayOzoneData()


if (__name__ == "__main__"):
    main()
