from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def lda_inbuilt(Xtrain,Ytrain, Xtest, n):
    lda = LinearDiscriminantAnalysis(n_components=n)
    Xtrain1 = lda.fit_transform(Xtrain,Ytrain)
    Xtest1 = lda.transform(Xtest)
    return Xtrain1, Xtest1