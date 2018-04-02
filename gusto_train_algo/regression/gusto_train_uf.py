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
import numpy as np
from sklearn.feature_selection import SelectPercentile, f_classif
from scipy import stats
import matplotlib.pyplot as plt
import sys
from sklearn.feature_selection import f_regression, mutual_info_regression


class GustoTrainUF():
    def __init__(self, X, y, columns_X, dir_, title):
        self.X = X
        self.y = y
        self.columns_X = columns_X
        self.dir_ = dir_
        self.prefix = title

    def train(self):
        # Univariate feature selection with F-test for feature scoring
        # We use the default selection function: the 10% most significant features
        selector = SelectPercentile(f_classif, percentile=10)
        selector.fit(self.X, self.y)
        scores = -np.log10(selector.pvalues_)
        scores /= scores.max()
        xxx = sorted(zip(map(lambda x: round(x, 4), scores), self.columns_X), reverse=True)

        res = dict()
        res['feature'] = zip(*xxx)[1]
        res['value'] = zip(*xxx)[0]
        tmp = pd.DataFrame(res)

        def plot_feature(df_):
            df = df_[:30]
            fig, ax = plt.subplots(1, 1, figsize=(8, 9))
            plt.subplots_adjust(left=0.15, right=0.95, bottom=0.45, top=0.9)
            ax = df['value'].plot(kind='bar', title="Top 30 factors by Univariate score ($-Log(p_{value})$)", color='darkorange',
                                  label=r'Univariate score ($-Log(p_{value})$)', legend=True, fontsize=12)
            # ax.set_ylim(top=0.2)
            # ax.set_xlabel("Hour", fontsize=12)
            ax.set_xticklabels(df['feature'], rotation=90)
            # ax.set_ylabel("V", fontsize=12)
            ax.legend().set_visible(False)
            fig.savefig(os.path.join(self.dir_['result'], '{} uf_selection.png'.format(self.prefix)))

        plot_feature(tmp)


        mi = mutual_info_regression(self.X, self.y)
        mi /= np.max(mi)
        xxx = sorted(zip(map(lambda x: round(x, 4), mi), self.columns_X), reverse=True)

        res = dict()
        res['feature'] = zip(*xxx)[1]
        res['value'] = zip(*xxx)[0]
        tmp = pd.DataFrame(res)

        def plot_feature(df_):
            df = df_[:30]
            fig, ax = plt.subplots(1, 1, figsize=(8, 9))
            plt.subplots_adjust(left=0.15, right=0.95, bottom=0.45, top=0.9)
            ax = df['value'].plot(kind='bar', title="Top 30 factors by Univariate score MI value", color='darkorange',
                                  label=r'Univariate score MI value', legend=True, fontsize=12)
            # ax.set_ylim(top=0.2)
            # ax.set_xlabel("Hour", fontsize=12)
            ax.set_xticklabels(df['feature'], rotation=90)
            # ax.set_ylabel("V", fontsize=12)
            ax.legend().set_visible(False)
            fig.savefig(os.path.join(self.dir_['result'], '{} uf_selection_MI.png'.format(self.prefix)))

        plot_feature(tmp)







