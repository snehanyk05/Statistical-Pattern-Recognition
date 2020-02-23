from  utils.mnist_reader import * 
import pca
import lda
import numpy as np
import sys

class KNN(object):
    def euclidean_distance(self,row1, row2):
        r1=row1.shape
        r2=row2.shape
        if(r1 != r2):
            sys.exit('Dimensions not equal')
        dist = 0.0
        for i in range(len(row1)-1):
          
            try:
                dist += (float(row1[i]) - float(row2[i]))**2
            except:
                print(row1[i],row2[i])
        return np.sqrt(dist)

  
    def sortSecond(self,val): 
        return val[1] 
    def get_neighbors(self, train, test, y, k):
        N,_ = test.shape
        P = np.zeros((N)) 
        for i,test_row in enumerate(test):
            distances = list()
            for c,train_row in enumerate(train):
                dist = self.euclidean_distance(test_row, train_row)
                distances.append((train_row, dist,y[c] ))
           
            
            distances.sort(key=self.sortSecond)
            neighbors = list()
            for j in range(k):
                neighbors.append(distances[j][2])
            
            P[i]=neighbors[0]
        return P



Xtrain_m, Ytrain = load_mnist('data/fashion', kind='train')
Xtest_m, Ytest = load_mnist('data/fashion', kind='t10k')
Xtrain_m=Xtrain_m[0:500,:]
Xtest_m=Xtest_m[0:500,:]
Ytrain=Ytrain[0:500]
Ytest=Ytest[0:500]

classifier = KNN()

P = classifier.get_neighbors(Xtrain_m, Xtest_m, Ytrain, 1)
Ytest = Ytest.astype(np.float)

score = np.round(np.mean(P == (Ytest)),4)
print("Test accuracy 1-nn:", score*100,'%')

Xtrain, Xtest = pca.pca_inbuilt(Xtrain_m, Ytrain, Xtest_m, 85)
P = classifier.get_neighbors(Xtrain, Xtest, Ytrain, 1)

score = np.round(np.mean(P == (Ytest)),4)
    
print("Test accuracy 1-nn with PCA Inbuilt:", score*100,'%')




Xtrain, Xtest = lda.lda_inbuilt(Xtrain_m, Ytrain, Xtest_m, 9)

P = classifier.get_neighbors(Xtrain, Xtest, Ytrain, 1)

score = np.round(np.mean(P == (Ytest)),4)
    
print("Test accuracy 1-nn with LDA Inbuilt:", score*100,'%')