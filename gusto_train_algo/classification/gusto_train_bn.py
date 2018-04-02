import os
import pandas as pd
from bayesnets import BayesNetsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from utility import *


class GustoTrain():
    def __init__(self, X, y, columns_X, dir_):
        self.X = X
        self.y = y
        self.columns_X = columns_X
        self.dir_ = dir_

    def train(self):
        rfc = GradientBoostingClassifier(n_estimators=100, random_state=rng)
        # rfc = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=rng, n_jobs=4)

        fs = rfc.fit(self.X, self.y).feature_importances_
        xxx = sorted(zip(map(lambda x: round(x, 4), fs), self.columns_X), reverse=True)
        print xxx
        res = dict()
        res['feature'] = zip(*xxx)[1]
        res['value'] = zip(*xxx)[0]
        tmp = pd.DataFrame(res)
        tmp.to_csv(os.path.join(self.dir_['result'], 'bn\\gusto scree.csv'))

        # %% Data for 2
        n_components = [1, 10, 30, 60, 80, 100, 120, 150]
        filtr = ImportanceSelect(rfc)
        bn_itr = [100,200,400]

        grid = {'filter__n': n_components, 'BN__n_iter': bn_itr}
        bn = BayesNetsClassifier(n_iter=400)
        pipe = Pipeline([('filter', filtr), ('BN', bn)])
        gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

        gs.fit(self.X, self.y)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(os.path.join(self.dir_['result'], 'bn\\gusto dim red.csv'))

        param = gs.best_params_
        print param
        print gs.best_score_
        print gs.best_index_
        print gs.best_estimator_.named_steps['NN'].loss_curve_
        pipe.set_params(**param)
        pipe.fit(self.X, self.y)
        if hasattr(pipe.named_steps['NN'], 'validation_scores_'):
            print pipe.named_steps['NN'].validation_scores_
        print pipe.named_steps['NN'].loss_curve_
        res = (gs.best_index_, pipe)
        joblib.dump(res, os.path.join(self.dir_['model'], 'bn\\gusto dim red model.pkl'))

