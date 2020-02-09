import numpy as np

# as of right now, this only uploads ionosphere,               #
# as this will be the first dataset i try using naive bayes on #

class NaiveBayes():
    
    weights = np.zeros(0)  # 1 x D
    X = np.zeros(0)     # N x D
    Y = np.zeros(0) # N x 1 
    
    def __init__(self, prior, likelihood, evidence, X, Y):
        self.prior = prior
        self.likelihood = likelihood
        self.evidence = evidence
        self.X = X
        self.Y = Y
        
    def fit(self, x):
        log_p = np.log(self.prior) + np.sum(np.log(self.likelihood) * x[:,None], 0) + np.sum(np.log(1-self.likelihood) * (1-x[:,None]), 0)
        log_p -= np.max(log_p) 
        post = np.exp(log_p)
        post /= np.sum(post)
        return post
            

def main():
    file = file = open("Ionosphere_Numpy_Array.txt", "r")
    array = np.loadtxt(file) # data loaded into numpy array


if (__name__ == "__main__"):
    main() # running main 