import os
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.externals import joblib
from gusto_train_algo.utility import *


class GustoTrainXGB():
    def __init__(self, X, y, columns_X, dir_, title):
        self.X = X
        self.y = y
        self.columns_X = columns_X
        self.dir_ = dir_
        self.prefix = title

    def train(self):
        rfc = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=rng, n_jobs=4)
        # rfc = GradientBoostingClassifier(n_estimators=100, random_state=rng)

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

        xgb_reg = [10 ** -x for x in range(1, 5)]
        gamma_range = np.logspace(-6, 3, 10)

        # n_components = [1,10]
        # xgb_reg = [10**-2]
        # gamma_range = [0.1]
        grid = {'filter__n': n_components, 'variant__reg_alpha':xgb_reg, 'variant__gamma':gamma_range}
        xgb = XGBClassifier()
        pipe = Pipeline([('filter', filtr), ('variant', xgb)])
        gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

        gs.fit(self.X, self.y)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(os.path.join(self.dir_['result'], 'xgb\\{} gusto dim red.csv'.format(self.prefix)))

        param = gs.best_params_
        print param
        print gs.best_score_
        print gs.best_index_
        # print gs.best_estimator_.named_steps['XGB'].loss_curve_
        pipe.set_params(**param)
        pipe.fit(self.X, self.y)
        if hasattr(pipe.named_steps['variant'], 'validation_scores_'):
            print pipe.named_steps['variant'].validation_scores_
        # print pipe.named_steps['XGB'].loss_curve_
        res = (gs.best_index_, pipe)
        joblib.dump(res, os.path.join(self.dir_['model'], 'xgb\\{} gusto dim red model.pkl'.format(self.prefix)))

