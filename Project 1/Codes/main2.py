from  utils.mnist_reader import * 
import pca
import lda
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
           

def score(classifier, X, Y, type1):
        P = classifier.predict(X)
        return np.round(np.mean(P == Y),4)

          
Xtrain_m, Ytrain = load_mnist('data/fashion', kind='train')
Xtest_m, Ytest = load_mnist('data/fashion', kind='t10k')
    

classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(Xtrain_m, Ytrain)

print("Test accuracy Knn Inbuilt:", score(classifier, Xtest_m, Ytest, 'Test')*100,'%')



Xtrain, Xtest = pca.pca_inbuilt(Xtrain_m, Ytrain, Xtest_m, 85)
classifier1 = KNeighborsClassifier(n_neighbors=3)
classifier1.fit(Xtrain, Ytrain)
    
print("Test accuracy Knn Inbuilt with PCA Inbuilt:", score(classifier1, Xtest, Ytest, 'Test')*100,'%')




Xtrain, Xtest = lda.lda_inbuilt(Xtrain_m, Ytrain, Xtest_m, 9)

classifier2 = KNeighborsClassifier(n_neighbors=1)
classifier2.fit(Xtrain, Ytrain)
    
print("Test accuracy Knn Inbuilt with LDA Inbuilt:", score(classifier2, Xtest, Ytest, 'Test')*100,'%')

