import os
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from gusto_train_algo.utility import *



class GustoTrainMLPR():
    def __init__(self, X, y, columns_X, dir_, title):
        self.X = X
        self.y = y
        self.columns_X = columns_X
        self.dir_ = dir_
        self.prefix = title

    def train(self):
        # rfc = GradientBoostingClassifier(n_estimators=100, random_state=rng)
        rfc = RandomForestRegressor(n_estimators=100, random_state=rng, n_jobs=4)

        fs = rfc.fit(self.X, self.y).feature_importances_
        xxx = sorted(zip(map(lambda x: round(x, 4), fs), self.columns_X), reverse=True)
        print xxx
        res = dict()
        res['feature'] = zip(*xxx)[1]
        res['value'] = zip(*xxx)[0]
        tmp = pd.DataFrame(res)
        tmp.to_csv(os.path.join(self.dir_['result'],'{} gusto scree.csv'.format(self.prefix)))

        # %% Data for 2
        n_components = [1, 10, 30, 60, 80, 100, 120, 150]
        filtr = ImportanceSelect(rfc)

        nn_arch = [(50, 50), (50,), (25,), (25, 25), (100, 25, 100), (21, 21, 21, 21)]
        nn_reg = [10 ** -x for x in range(1, 5)]

        # n_components = [1, 10]
        # nn_arch=[(50,50)]
        # nn_reg=[10**-2]

        grid = {'filter__n': n_components, 'variant__alpha': nn_reg, 'variant__hidden_layer_sizes': nn_arch}
        mlp = MLPRegressor(activation='relu', max_iter=2000, early_stopping=True, random_state=rng, solver='adam',
                            learning_rate_init=0.01)
        pipe = Pipeline([('filter', filtr), ('variant', mlp)])
        gs = GridSearchCV(pipe, grid, verbose=10, cv=5, scoring='r2')

        gs.fit(self.X, self.y)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(os.path.join(self.dir_['result'],'nn\\{} gusto dim red.csv'.format(self.prefix)))

        param = gs.best_params_
        print param
        print gs.best_score_
        print gs.best_index_
        # print gs.best_estimator_.named_steps['variant'].loss_curve_
        pipe.set_params(**param)
        pipe.fit(self.X, self.y)
        if hasattr(pipe.named_steps['variant'], 'validation_scores_'):
            print pipe.named_steps['variant'].validation_scores_
        # print pipe.named_steps['variant'].loss_curve_
        res = (gs.best_index_, pipe)
        joblib.dump(res, os.path.join(self.dir_['model'],'nn\\{} gusto dim red model.pkl'.format(self.prefix)))