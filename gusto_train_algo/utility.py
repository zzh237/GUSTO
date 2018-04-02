import numpy as np
from sklearn.base import TransformerMixin,BaseEstimator

rng = np.random.RandomState(42)

class ImportanceSelect(BaseEstimator, TransformerMixin):
    #select the number of important features
    def __init__(self, model, n=1):
         self.model = model
         self.n = n
    def fit(self, *args, **kwargs):
         self.model.fit(*args, **kwargs)
         return self
    def transform(self, X):
         return X[:,self.model.feature_importances_.argsort()[::-1][:self.n]]

