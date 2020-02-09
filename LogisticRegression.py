import numpy as np

class LogisticRegression():

    weights = np.zeros(0)  #1XD
    steps = 0    #number of steps
    X = np.zeros(0)     #N x D
    Y = np.zeros(0) #N x 1  

    def __init__(self, steps, X, Y):
        self.steps = steps
        self.weights = np.zeros(X.shape[1])
        self.X = X
        self.Y = Y
    

    
    def gradientDescent(self, X, Y, learning ):

        step = 0
        while step < self.steps:
            self.weights -= learning * LogisticRegression.gradient(X,Y, self.weights)
            step += 1
        
        

    def fit(self, data):
        y = np.dot(data, self.weights.T)
        y = LogisticRegression.logistic(y)
        lables = LogisticRegression.label(y)
        return lables



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


def main():
    X = np.loadtxt("Ionosphere_Numpy_Array.txt")
    Y = X[:, -1]
    X = X[:, 0:-1]
    lg = LogisticRegression(10000, X, Y)
    lg.gradientDescent(X,Y, 0.01)
    
    yClassified = lg.fit(X)

    #print(Y)
    #print(yClassified)

    TP = 0
    FN = 0
    TN = 0
    FP = 0
    for i in range(yClassified.T.shape[0]):
        if(Y[i] == 1 and Y[i] == yClassified[i]):
            TP += 1
        elif(Y[i] == 0 and Y[i] != yClassified[i]):
            FN += 1
        elif(Y[i] == 0 and Y[i] == yClassified[i]):
            TN += 1
        else:
            FP += 1
    
    acc = float(TP + TN)/float(len(yClassified))

    print(" True positive: %d \n False Negative: %d \n True Negative: %d \n False Positive: %d" % (TP, FN, TN, FP))
    print(" Accuracy: %f " % (acc))



if __name__ == "__main__":
    main()