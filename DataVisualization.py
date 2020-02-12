import matplotlib.pyplot as plt
import numpy as np

class DataVisualization():

    @staticmethod
    def displayIonosphereData():
        positiveCount = 0
        negativeCount = 0

        with open("CleanDatasets/Ionosphere_Numpy_Array.txt") as file:
            lines = file.readlines()
            x = [line.split()[0: -1] for line in lines]
            y = [line.split()[-1] for line in lines]

        for element in y:
                if (float(element) == 1):
                    positiveCount += 1
                else:
                    negativeCount += 1

        # display the distribution of positive and negative classes
        plt.title("Distribution of classes")
        plt.bar([0, 1], [negativeCount, positiveCount])
        plt.xticks([0, 1], [0, 1])
        plt.xlabel("Classes")
        plt.ylabel('Instances')
        plt.savefig("BasicStatistics/IonosphereClasses.pdf")
        plt.show()

        file.close()



    @staticmethod
    def displayAdultData():
        positiveCount = 0
        negativeCount = 0

        with open("CleanDatasets/Ionosphere_Numpy_Array.txt") as file:
            lines = file.readlines()
            x = [line.split()[0: -1] for line in lines]
            y = [line.split()[-1] for line in lines]

        for element in y:
                if (float(element) == 1):
                    positiveCount += 1
                else:
                    negativeCount += 1

        # display the distribution of positive and negative classes
        plt.title("Distribution of classes")
        plt.bar([0, 1], [negativeCount, positiveCount])
        plt.xticks([0, 1], [0, 1])
        plt.xlabel("Classes")
        plt.ylabel('Instances')
        plt.savefig("BasicStatistics/AdultClasses.pdf")
        plt.show()

        file.close()


def main():

    DataVisualization.displayIonosphereData()
    DataVisualization.displayAdultData()
    # DataVisualization.displayCancerData()
    # DataVisualization.displayOzoneData()



if (__name__ == "__main__"):
    main()
