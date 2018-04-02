import os
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from gusto_train_algo.utility import *
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class GustoTrainGB():
    def __init__(self, X, y, X_test, y_test, columns_X, dir_, title):
        self.X = X
        self.y = y
        self.columns_X = columns_X
        self.dir_ = dir_
        self.prefix = title
        self.X_test = X_test
        self.y_test = y_test

    def train(self):
        grid = {'n_estimators': [100],
                      'learning_rate': [0.01],
                      'max_depth':[4,6], 'min_samples_leaf':[3,5,9,17], 'max_features':[1.0,0.3,0.1] }

        estimator = GradientBoostingRegressor()

        gs = GridSearchCV(estimator, grid, verbose=10, cv=5)
        gs.fit(self.X, self.y)
        param = gs.best_params_
        best_est = gs.best_estimator_
        print best_est.score(self.X, self.y)
        print best_est.score(self.X_test, self.y_test)
        estimator = GradientBoostingRegressor(n_estimators=best_est.n_estimators,
                                              learning_rate=best_est.learning_rate)
        estimator.fit(self.X, self.y)

        fs = best_est.feature_importances_
        xxx = sorted(zip(map(lambda x: round(x, 4), fs), self.columns_X), reverse=True)
        print xxx
        res = dict()
        res['feature'] = zip(*xxx)[1]
        res['value'] = zip(*xxx)[0]
        tmp = pd.DataFrame(res)

        def plot_feature(df_):
            df = df_[:30]
            fig, ax = plt.subplots(1, 1, figsize=(8, 9))
            plt.subplots_adjust(left=0.15, right=0.95, bottom=0.45, top=0.9)
            ax = df['value'].plot(kind='bar', title="top 30 important features", legend=True, fontsize=12)
            ax.set_xticklabels(df['feature'], rotation=90)
            ax.legend().set_visible(False)
            fig.savefig(os.path.join(self.dir_['result'], '{} ft_imp_top30.png'.format(self.prefix)))

        plot_feature(tmp)



        n_components = [1, 5, 10, 20, 30, 60, 80, 100, 120, 150]
        filtr = ImportanceSelect(estimator)


        grid = {'filter__n': n_components}
        estimator = GradientBoostingRegressor(n_estimators=best_est.n_estimators, max_depth=best_est.max_depth,
                                              learning_rate=best_est.learning_rate,
                                              min_samples_leaf=best_est.min_samples_leaf,
                                              max_features=best_est.max_features)


        features = best_est.feature_importances_

        n = 10

        X_test_trim = self.X_test[:, features.argsort()[::-1][:n]]
        X_train_trim = self.X[:, features.argsort()[::-1][:n]]

        #test colinearity
        corr = np.corrcoef(X_train_trim, rowvar=0)
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
        keepidex = [i for i in range(0,10) if i not in removeidex]
        print keepidex

        # removeidex = [i for i, (score, name) in enumerate(xxx) if name =='job_seniority_unknown'][0]
        X_train_trim = X_train_trim[:,keepidex]
        X_test_trim = X_test_trim[:,keepidex]

        estimator = GradientBoostingRegressor(n_estimators=best_est.n_estimators, max_depth=best_est.max_depth,
                                              learning_rate=best_est.learning_rate,
                                              min_samples_leaf=best_est.min_samples_leaf,
                                              max_features=best_est.max_features)
        estimator.fit(X_train_trim, self.y)
        print estimator.score(X_train_trim, self.y)
        print estimator.score(X_test_trim, self.y_test)












