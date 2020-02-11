import numpy as np
import random
import matplotlib.pyplot as plt
import math

class LogisticRegression():

    weights = np.zeros(0)  #1XD
    X = np.zeros(0)     #N x D
    Y = np.zeros(0) #N x 1  
    reg = 0

    def __init__(self, X, Y, reg = 0):
        self.weights = np.zeros(X.shape[1])
        self.X = X
        self.Y = Y
        self.reg = reg
    

    
    def gradientDescent(self, X, Y, learning, threshold, regularization = 0):

        step = 0
        while True:#np.linalg.norm(newWeigths - self.weights) > 0.1:
            if(regularization == 0):
                newWeigths = self.weights - learning * LogisticRegression.gradient(X, Y, self.weights)
            elif(regularization == 1):
                newWeigths = self.weights - self.L2Gradient(X, Y, self.weights, learning)

            if(np.linalg.norm(self.weights - newWeigths) < threshold):
                break
            self.weights = newWeigths
            step += 1
        print(step)
        


    def fit(self, data):
        y = np.dot(data, self.weights.T)
        y = LogisticRegression.logistic(y)
        lables = LogisticRegression.label(y)
        return lables

    @staticmethod
    def crossValidation(data, k, lambd = 0):
        partions = np.array_split(data, k, 0)
        accAverage = np.zeros(k)
        trainAccuracy = np.zeros(k)
        print (lambd)
        for i in range(len(partions)):
            train = partions.copy()
            train = np.concatenate(train, 0)
            X = train[:, 0:-1]
            Y = train[:,-1]
            lg = LogisticRegression(X, Y, lambd)
            lg.gradientDescent(X, Y, 0.01, 1000, 1)



            xValidate = partions[i][:, 0:-1]
            yValidate = partions[i][:, -1]
            validLabels = lg.fit(xValidate)
            [TP, FN, TN, FP] = LogisticRegression.confusionTable(validLabels, yValidate)
            accAverage[i] = float(TP + TN)/float(len(validLabels))

            trainLabels = lg.fit(X)
            [TP, FN, TN, FP] = LogisticRegression.confusionTable(trainLabels, Y)
            trainAccuracy[i] = float(TP + TN)/float(len(trainLabels))
            

        accAverage = np.mean(accAverage)
        trainAccuracy = np.mean(trainAccuracy)
        print("Average validation accuracy: {0}".format(np.mean(accAverage)))
        print("Average train accuracy: {0}".format(np.mean(trainAccuracy)))
        return accAverage, trainAccuracy


    
    def L2Gradient(self, X, Y, W, learning):
        logis = LogisticRegression.logistic(np.dot(X, W))
        gradient = learning * np.dot(X.T, logis - Y) + self.reg*W
        return gradient

    #X N x D
    #Y D x 1
    #W 1 x D
    @staticmethod
    def gradient(X, Y, W):
        logis = LogisticRegression.logistic(np.dot(X, W.T))
        gradient = np.dot(X.T,  logis - Y)
        return gradient

    @staticmethod
    def logistic(logit):
        return 1 / (1 + np.exp(-logit))

    @staticmethod
    def label(y):
        newY = np.zeros(len(y))
        for i in range(len(y)):
            if (y[i] > 0.5):
                newY[i] = 1
            else:
                newY[i] = 0

        return newY

    @staticmethod
    def confusionTable(yClassified, Y):
        TP = 0
        FN = 0
        TN = 0
        FP = 0
        for i in range(yClassified.T.shape[0]):
            if(Y[i] == 1 and Y[i] == yClassified[i]):
                TP += 1
            elif(Y[i] == 0 and Y[i] != yClassified[i]):
                FP += 1
            elif(Y[i] == 0 and Y[i] == yClassified[i]):
                TN += 1
            else:
                FN += 1


        return TP, FN, TN, FP 



def main():
    iono = np.loadtxt("Ionosphere_Numpy_Array.txt")
    cancer = np.loadtxt("Ionosphere_Numpy_Array.txt")

    x = data[:, 0:-1]
    y = data[:, -1]
    normalizeData = nomralize(data[:, 0:-1])
    normalizeData = np.append(normalizeData, data[:, -1:] , axis=1 )
    threshold = np.linspace(0.01, 1, 20)
    for i in threshold:
        data = np.array_split(data, 10, 0)
    
        index = 0
    test = data[index]

    del data[index]
    data = np.concatenate(data, 0)
    LogisticRegression.crossValidation(data, 5)

    X = data[:, 0:-1]
    Y = data[:, -1]
    lg = LogisticRegression(X,Y)
    lg.gradientDescent(X, Y, 0.01, )
    labels = lg.fit(test[:, 0:-1])
    TP, FN, TN, FP = LogisticRegression.confusionTable(labels, test[:, -1])

    acc = float(TP + TN)/len(labels)
    print("final accuracy without regularization: {0}".format(acc))




    
    normalizeData = np.array_split(normalizeData, 10, 0)
    test = normalizeData[index]
    del normalizeData[index]

    normalizeData = np.concatenate(normalizeData, 0)
    regulization = np.linspace(0, 10, 20)

    #accuracy = []
    #for lambd in regulization:
    #    acc, _ = LogisticRegression.crossValidation(normalizeData, 5, lambd)
    #    accuracy.append(acc)
    #plt.plot(regulization.T, accuracy)
    #plt.show()
    
    

    



def nomralize(input):
    data = np.copy(input)
    average = np.mean(data, 0)
    std = np.std(data, 0)
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            if(std[i] == 0):
                data[j,i] = 0
            else:
                data[j,i] = (data[j,i] - average[i])/math.sqrt(std[i])

    return data


if __name__ == "__main__":
    main()