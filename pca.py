import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError

#This is Tea's code from Homework 3

class Standard_PCA:
    def __init__(self, n_components: int):
        
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.Mean = None
        self.STD = None
        self.isFitted = False
        

    def fit(self, X: np.ndarray):
        
        self.Mean = np.mean(X, axis=0)
        self.STD = np.std(X, axis=0)

        X_standard = (X - self.Mean)/self.STD

        self.pca.fit(X_standard)
        self.isFitted = True
        

    def fit_transform(self, X: np.ndarray) -> np.ndarray:

        self.fit(X)
        return self.transform(X)


    
    def transform(self, X: np.ndarray) -> np.ndarray:
        
        if not self.isFitted:
            raise NotFittedError("This Standard_PCA is not fitted. Please call fit() first.")

        X_standard = (X - self.Mean) / self.STD
        return self.pca.transform(X_standard)


def PCA_dim(X: np.ndarray, p: float) -> int:
    
    spca = Standard_PCA(n_components=X.shape[1])
    spca.fit(X)

    cum_var = np.cumsum(spca.pca.explained_variance_ratio_)
    d = np.argmax(cum_var >= p) + 1
    
    return d
