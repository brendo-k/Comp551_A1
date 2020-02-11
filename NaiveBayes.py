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
    
      
    def fit(summaries, instance, Y): # calculates class probabilities
        num_instances = np.sum([summaries[label][0][2] for label in summaries])
        probs = dict()
        for label, class_summs in summaries.items():
            probs[label] = (summaries[label][0][2] / float(num_instances))
            probs[label] = NaiveBayes.probLabel(label, Y)
            for i in range(len(class_summs)):
                mu, sigma, count = class_summs[i]
                feature = instance[i]
                probs[label] *= NaiveBayes.gaussProb(feature, mu, sigma)
        return probs
    
    @staticmethod
    def predict(data, Y): # this will give a final prediction
        y = [None]*len(data)
        summaries = NaiveBayes.summarizeClass(data)
        for i in range(len(data)):
            instance = data[i]
            instanceProbs = NaiveBayes.fit(summaries, instance, Y)
            bestLabel, bestProb = None, -1.0
            for label, prob in instanceProbs.items():
                if (bestLabel is None) or (prob > bestProb):
                    bestProb = prob
                    bestLabel = label
            y[i] = bestLabel
        return y
        
    @staticmethod    
    def gaussProb(x, mean, std):
        exp = np.exp(-((x-mean)**2)/(2 * (std**2)))
        prob = exp * (1 / (np.sqrt(2 * np.pi) * std))
        return prob
    
    @staticmethod
    def probLabel(lab, Y):
        count = 0
        for i in range(len(Y)):
            if Y[i]==lab:
                count+=1
        return count/len(Y)
    
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
    def summarizeData(data): # first el of the tuple is mean, second is std. dev.
        summs= [(np.mean(col), np.std(col), len(col)) for col in zip(*data)]
        del(summs[-1])
        return summs
    
    @staticmethod
    def summarizeClass(data): 
        split = NaiveBayes.splitData(data)
        summs = dict()
        for label, instances in split.items():
            summs[label] = NaiveBayes.summarizeData(instances)
        return summs

def main():
    file = open("Ionosphere_Numpy_Array.txt", "r")
    X = np.loadtxt(file) # data loaded into numpy array
    X_clean = np.delete(X, 1, 1)
    yClassified = X[:, -1] # extracting labels
    X_clean = X_clean[:, 1:] # removing the labels
    yPred = NaiveBayes.predict(X_clean, yClassified)

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
    
if __name__ == "__main__":
    main()