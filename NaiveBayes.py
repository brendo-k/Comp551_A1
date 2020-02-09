import numpy as np

# as of right now, this only uploads ionosphere,               #
# as this will be the first dataset i try using naive bayes on #

class NaiveBayes():
    
    X = np.zeros(0)     # N x D
    Y = np.zeros(0) # N x 1 
    
    def __init__(self, prior, likelihood, evidence, X, Y):
        self.prior = prior
        self.likelihood = likelihood
        self.evidence = evidence
        self.X = X
        self.Y = Y
    
    @staticmethod    
    def fit(summs, row): # copied, needs revision
        num_instances = np.sum([summs[label][0][2] for label in summs])
        probs = dict()
        for label, class_summs in summs.items():
            probs[label] = (summs[label][0][2] / float(num_instances))
            for i in range(len(class_summs)):
                mean, std, count = class_summs[i]
                probs[label] *= NaiveBayes.like(row[i], mean, std)
        return probs
    
    @staticmethod
    def predict(data):
        y = [None]*len(data)
        for i in range(len(data)):
            probs = NaiveBayes.fit(NaiveBayes.classSumms(data), data[i])
            if probs[0]>=probs[1]:
                y[i] = 0
            else:
                y[i] = 1
        return y
     
    @staticmethod    
    def like(x, mean, std):
        exp = np.exp(-((x-mean)**2)/(2 * (std**2)))
        prob = exp * (1 / (np.sqrt(2 * np.pi) * std))
        return prob
    
    @staticmethod
    def splitData(data): # organizing the data by class (dict: label -> list of instances with that label)
        split = dict()
        for i in range(len(data)):
            instance = data[i]
            label = (instance[-1])
            if (label not in split):
                split[label] = list()         
            split[label].append(instance)
        return split
            
    @staticmethod            # creates an array of tuples that summarizes the data:
    def summData(data): # first el of the tuple is mean, second is std. dev.
        summs= [(np.mean(col), np.std(col), len(col)) for col in zip(*data)]
        del(summs[-1])
        return summs
    
    @staticmethod
    def classSumms(data): 
        split = NaiveBayes.splitData(data)
        summs = dict()
        for label, instances in split.items():
            summs[label] = NaiveBayes.summData(instances)
        return summs

def main():
    file = file = open("Ionosphere_Numpy_Array.txt", "r")
    X = np.loadtxt(file) # data loaded into numpy array
    X_clean = np.delete(X, 1, 1)
    yClassified = X[:, -1] # extracting labels
    X_clean = X_clean[:, 1:] # removing the labels
    yPred = NaiveBayes.predict(X_clean)
    TP = 0
    FN = 0
    TN = 0
    FP = 0
    for i in range(yClassified.shape[0]):
        if(yPred[i] == 1 and yPred[i] == yClassified[i]):
            TP += 1
        elif(yPred[i] == 0 and yPred[i] != yClassified[i]):
            FN += 1
        elif(yPred[i] == 0 and yPred[i] == yClassified[i]):
            TN += 1
        else:
            FP += 1
    
    acc = float(TP + TN)/float(len(yClassified))

    print(" True positive: %d \n False Negative: %d \n True Negative: %d \n False Positive: %d" % (TP, FN, TN, FP))
    print(" Accuracy: %.3f%% " % (acc*100))
    
if (__name__ == "__main__"):
    main() # running main 
    