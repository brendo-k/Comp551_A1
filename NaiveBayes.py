import numpy as np
import random as rand

# as of right now, this only uploads ionosphere,               #
# as this will be the first dataset i try using naive bayes on #

class NaiveBayes():
    
    X = np.zeros(0)     # N x D
    Y = np.zeros(0) # N x 1 
    
    def __init__(self, prior, likelihood, evidence, X, Y):
        self.X = X
        self.Y = Y
    
    @staticmethod  
    def fit(summaries, instance): # calculates class probabilities
        num_instances = np.sum([summaries[label][0][2] for label in summaries])
        probs = dict()
        for label, class_summs in summaries.items():
            probs[label] = summaries[label][0][2] / float(num_instances)
            for i in range(len(class_summs)):
                mu, sigma, count = class_summs[i]
                feature = instance[i]
                probs[label] *= NaiveBayes.gaussProb(feature, mu, sigma)
        return probs
    
    @staticmethod
    def predict(summaries, instance): # this will give a final prediction
        probs = NaiveBayes.fit(summaries, instance)
        best_label, best_prob = None, -1.0
        for label, prob in probs.items():
            if (best_label is None) or (prob > best_prob):
                best_prob = prob
                best_label = label
        return best_label
    
    @staticmethod
    def nb(training_data, test_data):
        trained = NaiveBayes.summarizeClass(training_data)
        pred = list()
        for instance in test_data:
            label = NaiveBayes.predict(trained, instance)
            pred.append(label)
        return pred
        
        
    @staticmethod    
    def gaussProb(x, mean, std):
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
    
    @staticmethod
    def crossValidation(data, n):
        data_folded = list()
        data_copy = list(data) #list?
        fold_size = int(len(data) / n)
        for _ in range(n):
            fold = list()
            while len(fold) < fold_size:
                i = rand.randrange(len(data_copy))
                fold.append(data_copy.pop(i))
            data_folded.append(fold)
        return data_folded
    
    @staticmethod
    def removeFold(folds, fold):
        i = 0
        size = len(folds)
        while i != size and not np.array_equal(folds[i], fold):
            i+=1
        if i!= size:
            folds.pop(i)
        
    
    @staticmethod
    def eval(data, n): # requires data INCLUDING labels
        folds = NaiveBayes.crossValidation(data, n)
        vals = list()
        for fold in folds:
            train_set = list(folds)
            NaiveBayes.removeFold(train_set, fold)
            train_set = sum(train_set, [])
            test_set = list()
            for instance in fold:
                instance_copy = list(instance) # list?
                test_set.append(instance_copy)
                instance_copy[-1] = None 
            pred = NaiveBayes.nb(train_set, test_set)
            real = [instance[-1] for instance in fold]
            accuracy = NaiveBayes.getAccuracy(real, pred)
            vals.append(accuracy)
        return vals, pred, real
            
    @staticmethod
    def getAccuracy(real, pred): # takes in actual results and predicted results
        score = 0
        for i in range(len(pred)):
            if real[i]==pred[i]:
                score += 1
        return score * 100 / float(len(real)) 
        
def main():
    file = open("Adult_Numpy_Array.txt", "r")
    X = np.loadtxt(file) # data loaded into numpy array
    Y = X[:, -1]
    X_clean = np.delete(X, 1, 1)
    X_unbiased = np.delete(X, 8, 1)
    X_unbiased = np.delete(X_unbiased, 8, 1)
    X_clean = X_clean[:, 1:] # YOU MUST USE X_clean FOR IONOSPHERE!!!
    accuracy, yPred, yClassified = NaiveBayes.eval(X_unbiased, 5)
    accuracy = list(np.around(accuracy, 3))
    av = np.round(np.average(accuracy), 3)
    for i in range(len(accuracy)):
        accuracy[i] = str(accuracy[i]) + "% "
    print("")
    print("Accuracy per fold test: ")
    print(accuracy)
    print("")
    print("Average accuracy:")
    print(str(av)+"%")
    print("")
    print("Test Set Break-Down:")
    TP = 0
    FN = 0
    TN = 0
    FP = 0
    for i in range(len(yClassified)):
        if(yPred[i] == 1 and yPred[i] == yClassified[i]):
            TP += 1
        elif(yPred[i] == 0 and yPred[i] != yClassified[i]):
            FN += 1
        elif(yPred[i] == 0 and yPred[i] == yClassified[i]):
            TN += 1
        else:
            FP += 1
    
    print(" True positive: %d \n False Negative: %d \n True Negative: %d \n False Positive: %d" % (TP, FN, TN, FP))

    
if __name__ == "__main__":
    main()