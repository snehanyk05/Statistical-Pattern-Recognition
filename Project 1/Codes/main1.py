from  utils.mnist_reader import * 
import pca
import lda
import numpy as np
from scipy.stats import multivariate_normal as mvn

class Bayes(object):
    def calc_Mean_Variance(self, X, Y, method, lam=0.01989):
        _, D = X.shape
        self.gaussians = dict()
        for c in range(0,10):
            
            
            x = X[Y == c]
            if(method==1):
                xmax, xmin = x.max(), x.min()
                x = (x - xmin)/(xmax - xmin)

            mean = x.mean(axis=0) 
            cov = np.cov(x.T) + np.eye(D)*lam

            self.gaussians[c] = {
                'mean': mean,
                'cov': cov,
            }
            
            
    def score(self, X, Y, type1):
        N,_ = X.shape
        P = np.zeros((10,N))
        # print(X[0].shape)
        for  (c,g) in (self.gaussians.items()):
                
                P[c,:] = mvn.logpdf(X, mean=g['mean'], cov=g['cov']) 
        P_c = np.argmax(P,axis=0)
    
        return np.round(np.mean(P_c == Y),4)


    
Xtrain_m, Ytrain = load_mnist('data/fashion', kind='train')
Xtest_m, Ytest = load_mnist('data/fashion', kind='t10k')
    


classifier = Bayes()

classifier.calc_Mean_Variance(Xtrain_m, Ytrain, 1)
    
print("Test accuracy Bayes:", classifier.score(Xtest_m, Ytest, 'Test')*100,'%')

Xtrain_m = Xtrain_m - np.mean(Xtrain_m) 
Xtest_m = Xtest_m - np.mean(Xtest_m) 

Xtrain, Xtest = pca.pca_inbuilt(Xtrain_m, Ytrain, Xtest_m, 45)
classifierPCA = Bayes()

classifierPCA.calc_Mean_Variance(Xtrain, Ytrain, 2)
    
print("Test accuracy Bayes with PCA Inbuilt:", classifierPCA.score(Xtest, Ytest, 'Test')*100,'%')




Xtrain, Xtest = lda.lda_inbuilt(Xtrain_m, Ytrain, Xtest_m, 9)

classifierLDA = Bayes()

classifierLDA.calc_Mean_Variance(Xtrain, Ytrain, 3)
    
print("Test accuracy Bayes with LDA Inbuilt:", classifierLDA.score(Xtest, Ytest, 'Test')*100,'%')

