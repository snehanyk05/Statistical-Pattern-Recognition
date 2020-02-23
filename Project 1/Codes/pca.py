from sklearn.decomposition import PCA

def pca_inbuilt(Xtrain,Ytrain, Xtest, n):
    pca = PCA(n_components=n)
    Xtrain1 = pca.fit_transform(Xtrain,Ytrain)
    Xtest1 = pca.transform(Xtest)
    return Xtrain1, Xtest1