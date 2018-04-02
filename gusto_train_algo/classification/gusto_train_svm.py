import os
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib
from gusto_train_algo.utility import *
import sys


class GustoTrainSVM():
    def __init__(self, X, y, columns_X, dir_, title):
        self.X = X
        self.y = y
        self.columns_X = columns_X
        self.dir_ = dir_
        self.prefix = title

    def train(self):
        # rfc = GradientBoostingClassifier(n_estimators=100, random_state=rng)
        rfc = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=rng, n_jobs=4)

        fs = rfc.fit(self.X, self.y).feature_importances_
        xxx = sorted(zip(map(lambda x: round(x, 4), fs), self.columns_X), reverse=True)
        print xxx
        res = dict()
        res['feature'] = zip(*xxx)[1]
        res['value'] = zip(*xxx)[0]
        tmp = pd.DataFrame(res)
        tmp.to_csv(os.path.join(self.dir_['result'], '{} gusto scree.csv'.format(self.prefix)))

        # %% Data for 2
        n_components = [1, 10, 30, 60, 80, 100, 120, 150]
        filtr = ImportanceSelect(rfc)
        C_range = np.logspace(-2, 4, 7)
        gamma_range = np.logspace(-4, 3, 8)



        # n_components = [1, 10]
        # C_range = [0.1]
        # gamma_range = [0]

        grid = {'filter__n': n_components, 'variant__C':C_range, 'variant__gamma':gamma_range}
        svc = SVC()
        pipe = Pipeline([('filter', filtr), ('variant', svc)])
        gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

        gs.fit(self.X, self.y)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(os.path.join(self.dir_['result'], 'svm\\{} gusto dim red.csv'.format(self.prefix)))

        param = gs.best_params_
        print param
        print gs.best_score_
        print gs.best_index_
        pipe.set_params(**param)
        pipe.fit(self.X, self.y)
        if hasattr(pipe.named_steps['variant'], 'validation_scores_'):
            print pipe.named_steps['variant'].validation_scores_
        res = (gs.best_index_, pipe)
        joblib.dump(res, os.path.join(self.dir_['model'], 'svm\\{} gusto dim red model.pkl'.format(self.prefix)))

