from  utils.mnist_reader import * 
import pca
import lda
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

def score(classifier, X, Y, type1):
        P = classifier.predict(X)
        return np.round(np.mean(P == Y),4)

          
Xtrain_m, Ytrain = load_mnist('data/fashion', kind='train')
Xtest_m, Ytest = load_mnist('data/fashion', kind='t10k')

xmean, xmax, xmin = np.mean(Xtrain_m), Xtrain_m.max(), Xtrain_m.min()
Xtrain_m = (Xtrain_m - xmean)/(xmax - xmin)
Xtest_m = (Xtest_m - xmean)/(xmax - xmin)

print("Reducing Dimensionality using LDA..")
Xtrain, Xtest = lda.lda_inbuilt(Xtrain_m, Ytrain, Xtest_m, 9)
print("Training...")

classifier = LinearSVC()
classifier.fit(Xtrain, Ytrain)   
print("Test accuracy Linear SVM with LDA Inbuilt:", score(classifier, Xtest, Ytest, 'Test')*100,'%')


classifier = SVC(kernel='linear', gamma='auto')
classifier.fit(Xtrain, Ytrain)   
print("Test accuracy Linear Kernel SVM  with LDA Inbuilt:", score(classifier, Xtest, Ytest, 'Test')*100,'%')




classifier = SVC(kernel='poly', gamma='auto')
classifier.fit(Xtrain, Ytrain)   
print("Test accuracy Poly Kernel SVM with LDA Inbuilt:", score(classifier, Xtest, Ytest, 'Test')*100,'%')



classifier = SVC(kernel='rbf', gamma='auto')
classifier.fit(Xtrain, Ytrain)   
print("Test accuracy RBF Kernel SVM with LDA Inbuilt:", score(classifier, Xtest, Ytest, 'Test')*100,'%')


print("Reducing Dimensionality using PCA..")
Xtrain, Xtest = pca.pca_inbuilt(Xtrain_m, Ytrain, Xtest_m, 50)
print("Training...")




classifier = LinearSVC()
classifier.fit(Xtrain, Ytrain)   
print("Test accuracy Linear SVM with PCA Inbuilt:", score(classifier, Xtest, Ytest, 'Test')*100,'%')


classifier = SVC(kernel='linear', gamma='auto')
classifier.fit(Xtrain, Ytrain)   
print("Test accuracy Linear Kernel SVM with PCA Inbuilt:", score(classifier, Xtest, Ytest, 'Test')*100,'%')


classifier = SVC(kernel='rbf', gamma='auto')
classifier.fit(Xtrain, Ytrain)   
print("Test accuracy RBF Kernel SVM with PCA Inbuilt:", score(classifier, Xtest, Ytest, 'Test')*100,'%')

classifier = SVC(kernel='poly', gamma='auto')
classifier.fit(Xtrain, Ytrain)   
print("Test accuracy Poly Kernel SVM with PCA Inbuilt:", score(classifier, Xtest, Ytest, 'Test')*100,'%')



