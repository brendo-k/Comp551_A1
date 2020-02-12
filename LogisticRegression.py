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
        self.weights = np.zeros([1,X.shape[1]])
        self.X = X
        self.Y = Y
        self.reg = reg
    

    
    def gradientDescent(self, X, Y, learning, steps = -1, threshold = -1):
        
        step = 0
        
        while True:
            if(self.reg == 0):
                lr = learning * self.gradient(X,Y)
                newWeigths = self.weights - lr
            else:
                reg = self.reg * self.weights
                newWeigths = self.weights - learning * self.gradient(X, Y) + reg
            if(np.linalg.norm(self.weights - newWeigths) < threshold):
                break
            elif(step > steps):
                break
            self.weights = newWeigths

            step += 1
        return step
        


    def fit(self, data):
        y = np.matmul(data, self.weights.T)
        y = LogisticRegression.logistic(y)
        lables = LogisticRegression.label(y)
        return lables

    @staticmethod
    def crossValidation(data, k, learning, thresh, lambd = 0):
        print(lambd)
        partions = np.array_split(data, k, 0)
        accAverage = np.zeros(k)
        trainAccuracy = np.zeros(k)
        steps = np.zeros(k)
        for i in range(len(partions)):
            
            train = partions.copy()
            del train[i]
            train = np.concatenate(train, 0)
            X = train[:, 0:-1]
            Y = train[:,-1:]
            lg = LogisticRegression(X, Y, lambd)
            steps[i] = lg.gradientDescent(X, Y, learning, 1000, threshold=thresh)
            xValidate = partions[i][:, 0:-1]
            yValidate = partions[i][:, -1:]
            validLabels = lg.fit(xValidate)
            [TP, FN, TN, FP] = LogisticRegression.confusionTable(validLabels, yValidate)
            accAverage[i] = float(TP + TN)/float(len(validLabels))

            trainLabels = lg.fit(X)
            [TP, FN, TN, FP] = LogisticRegression.confusionTable(trainLabels, Y)
            trainAccuracy[i] = float(TP + TN)/float(len(trainLabels))
            

        accAverage = np.mean(accAverage)
        trainAccuracy = np.mean(trainAccuracy)
        steps = np.mean(steps)
        print("Average validation accuracy: {0}".format(np.mean(accAverage)))
        print("Average train accuracy: {0}".format(np.mean(trainAccuracy)))
        print("Average steps:  {0}".format(steps) )
        return accAverage, trainAccuracy, steps

    #X N x D
    #Y D x 1
    #W 1 x D
    def gradient(self, X, Y):
        logis = LogisticRegression.logistic(np.dot(X, self.weights.T))
        gradient = np.dot(X.T,  logis - Y)
        return gradient.T

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
    iono = np.loadtxt("CleanDatasets/Ionosphere_Numpy_Array.txt")
    cancer = np.loadtxt("CleanDatasets/Cancer_Numpy_Array.txt")
    ozone = np.loadtxt("CleanDatasets/Ozone_Numpy_Array.txt")
    adult = np.loadtxt("CleanDatasets/Adult_Numpy_Array.txt")

    data = ozone
    #calculateAccuracy(data)
    #learningRate = np.linspace(0.001, 0.01 , 20)
    #steps = []
    #validateAccuracy = []
    #testAcc = []
    #count = 0
    #for i in learningRate:
        #trainAcc, validateAcc, step = LogisticRegression.crossValidation(data, 5, i, 0.005)
        #validateAccuracy.append(validateAcc)
        #steps.append(step)
        #count += 1
        #X = data[:, 0:-1]
        #Y = data[:, -1:]
        #lg = LogisticRegression(X,Y)
        #lg.gradientDescent(X, Y, i, 3000)
        #labels = lg.fit(test[:, 0:-1])
        #TP, FN, TN, FP = LogisticRegression.confusionTable(labels, test[:, -1])
        #acc = float(TP + TN)/len(labels)
        #testAcc.append(acc)
    #print(testAcc) 

    #plt.plot(learningRate, steps)
    #plt.title("Steps vs learning rate")
    #plt.xlabel("learning rate")
    #plt.ylabel("Steps")
    #print(steps)
    #plt.show()

#    print("final accuracy without regularization: {0}".format(acc))



    normalizeData = nomralize(data[:, 0:-1])
    normalizeData = np.concatenate((normalizeData, data[:, -1:]), 1)
    index = 0
    normalizeData = np.array_split(normalizeData, 10, 0)
    test = normalizeData[index]
    del normalizeData[index]

    normalizeData = np.concatenate(normalizeData, 0)

    regulization = np.linspace(0, 1, 20)

    accuracy = []
    for lambd in regulization:
       acc, validationAcc, _ = LogisticRegression.crossValidation(normalizeData, 5, 0.004, 0.005, lambd=lambd)
       accuracy.append(acc)

    #lg = LogisticRegression(normalizeData[:, 0:-1], normalizeData[:, -1:], 0.01)
    #lg.gradientDescent(normalizeData[:, 0:-1], normalizeData[:, -1:], 0.04, 1000)
    #labels = lg.fit(test[:, 0:-1])
    #TP, FN, TN, FP = LogisticRegression.confusionTable(labels, test[:, -1:])
    #acc = float(TP + TN)/len(labels)
    #print("final accuracy reg: {0}".format(acc))
    
    plt.plot(regulization.T, accuracy)
    plt.title("Accuracy vs regularization strength")
    plt.xlabel("lambda")
    plt.ylabel("Accuracy")
    plt.show()
    
def calculateAccuracy(data):    
    data = np.array_split(data, 10, 0)
    
    index = 0
    test = data[index]

    del data[index]
    data = np.concatenate(data, 0)

    trainAcc, validationAcc, steps = LogisticRegression.crossValidation(data, 5, 0.004, 0.005)
    
    X = data[:, 0:-1]
    Y = data[:, -1:]
    lg = LogisticRegression(X,Y)
    lg.gradientDescent(X, Y, 0.004, 1000)
    labels = lg.fit(test[:, 0:-1])
    TP, FN, TN, FP = LogisticRegression.confusionTable(labels, test[:, -1:])
    acc = float(TP + TN)/len(labels)
    print("final accuracy: {0}".format(acc))
    return acc

def variableTrainSize(data):
    data = np.array_split(data, 10, 0)
    
    index = 0
    test = data[index]

    del data[index]
    data = np.concatenate(data, 0)


    

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