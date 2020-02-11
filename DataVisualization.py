import matplotlib.pyplot as plt
# %matplotlib inline

class DataVisualization():

    @staticmethod
    def displayIonosphereData():
        file = open("Datasets/ionosphere.data", "r")
        lines = file.readlines()







def main():

    DataVisualization.displayIonosphereData()
    # DataVisualization.displayAdultData()
    # DataVisualization.displayCancerData()
    # DataVisualization.displayOzoneData()


if (__name__ == "__main__"):
    main()
