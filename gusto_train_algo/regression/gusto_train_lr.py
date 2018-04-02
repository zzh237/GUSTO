import os
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from gusto_train_algo.utility import *
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats


class GustoTrainLR():
    def __init__(self, X, y, columns_X, dir_, title):
        self.X = X
        self.y = y
        self.columns_X = columns_X
        self.dir_ = dir_
        self.prefix = title

    def train(self):
        X2 = sm.add_constant(self.X)
        est = sm.OLS(self.y, X2)
        est2 = est.fit()
        p_values = est2.pvalues[1:]
        print(est2.summary())
        columns =  np.asarray(self.columns_X)
        index = [i for i,v in enumerate(p_values) if v <= 0.05]
        res = zip(columns[index], p_values[index])
        print res

        X = self.X[:, index]


        X2 = sm.add_constant(X)
        est = sm.OLS(self.y, X2)
        est2 = est.fit()
        print(est2.summary())

        # test colinearity
        corr = np.corrcoef(X, rowvar=0)
        w, v = np.linalg.eig(corr)

        print w
        index = w.argsort()[:1]
        print index
        # small_w = [i for i, v in enumerate(w) if v < 0.1]
        xx = v[:, index]
        print xx
        xx = abs(xx).flatten()
        print xx
        removeidex = xx.argsort()[::-1][:1]
        print removeidex[0]
        keepidex = [i for i in range(0, len(xx)) if i not in removeidex]
        print keepidex

        # removeidex = [i for i, (score, name) in enumerate(xxx) if name =='job_seniority_unknown'][0]
        X_train_trim = X[:, keepidex]

        X2 = sm.add_constant(X_train_trim)
        est = sm.OLS(self.y, X2)
        est2 = est.fit()
        print(est2.summary())





