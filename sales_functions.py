import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
import matplotlib.pyplot as plt

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

#these are additional functions to help with finding the best depth for the decision tree regressor
def find_max_depth(X_train, y_train):
  best_score = 0 
  best_depth = 0 
  cv_scores = [] # create empty list for storing cross validation cores
  depths= list(range(1,10)) # a list of max depth to explore from 1 to 9

  # loops through each possibile max depth to find the one with the highest cross validation score
  for d in depths:
      T = DecisionTreeRegressor(max_depth=d, random_state=123)
      scores = cross_val_score(T, X_train, y_train, cv=10) # use a 10-fold cross validation
      mean_score = scores.mean() # get the mean of cross validation scores
      cv_scores.append(mean_score) # append them into the list

      if mean_score > best_score: # if the current mean score is higher than the best:
          best_score = mean_score # then update it to become the best
          best_depth = d # get the depth d corresponding to the best score
  return best_depth, best_score, cv_scores

def plot_best_depth(best_depth, cv_scores):
  #plot results for best depth
  fig, ax= plt.subplots(1)
  ax.set(xlabel = "tree depth",
        ylabel = "cross validation score")
  plt.scatter(range(1,10), cv_scores)
  plt.title("Best Depth: "+ str(best_depth))
  print(best_score)
