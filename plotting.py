import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from utility import *

class Plotter():
    def __init__(self, dir, X_test, y_test, title):
        self.dir_ = dir

        self.X_test = X_test
        self.y_test = y_test
        self.prefix = title

    def plot(self):
        self.plot_feature()
        self.plot_error()

    def plot_feature(self):
        df_ = pd.read_csv(os.path.join(self.dir_['result'], '{} gusto scree.csv'.format(self.prefix)), index_col=0)
        df = df_[:30]
        fig, ax = plt.subplots(1, 1, figsize=(8, 9))
        plt.subplots_adjust(left=0.15, right=0.95, bottom=0.45, top=0.9)
        ax = df['value'].plot(kind='bar', title="top 30 important features",   legend=True, fontsize=12)
        # ax.set_ylim(top=0.2)
        # ax.set_xlabel("Hour", fontsize=12)
        ax.set_xticklabels(df['feature'], rotation = 90)
        # ax.set_ylabel("V", fontsize=12)
        ax.legend().set_visible(False)
        fig.savefig(os.path.join(self.dir_['result'], '{} ft_imp_top30.png'.format(self.prefix)))


        plt.close(fig)
        df = df_[-30:]
        fig, ax = plt.subplots(1, 1, figsize=(8, 9))
        plt.subplots_adjust(left=0.15, right=0.95, bottom=0.55, top=0.9)
        ax = df['value'].plot(kind='bar', title="least 30 important features", legend=True, fontsize=12)
        # ax.set_ylim(top=0.2)
        # ax.set_xlabel("Hour", fontsize=12)
        ax.set_xticklabels(df['feature'], rotation=90)
        # ax.set_ylabel("V", fontsize=12)
        ax.legend().set_visible(False)
        fig.savefig(os.path.join(self.dir_['result'], '{} ft_imp_bot30.png'.format(self.prefix)))



    def plot_error(self):
        res = defaultdict(dict)
        tracks = ['nn', 'svm', 'xgb']
        for track in tracks:
            out = os.path.join(self.dir_['result'], track)
            modelout = os.path.join(self.dir_['model'], track)
            print track
            (index, pipe) = joblib.load(os.path.join(modelout,'{} gusto dim red model.pkl'.format(self.prefix)))
            df = pd.read_csv(os.path.join(out,'{} gusto dim red.csv'.format(self.prefix)), index_col=0)
            print pipe.named_steps
            model = pipe.named_steps['variant']
            res[track]['training time'] = df.loc[index, 'mean_fit_time']
            res[track]['cv score'] = df.loc[index, 'mean_test_score']
            res[track]['train score'] = df.loc[index, 'mean_train_score']
            res[track]['cv score std'] = df.loc[index, 'std_test_score']
            res[track]['train score std'] = df.loc[index, 'std_train_score']
            res[track]['model'] = model

            features = pipe.named_steps['filter'].model.feature_importances_
            n = pipe.named_steps['filter'].n
            print features.argsort()[::-1][:n]
            print features.argsort()[-n:][::-1]
            testX = self.X_test[:, features.argsort()[::-1][:n]]
            pred_y = model.predict(testX)
            res[track]['test score'] = np.mean(pred_y == self.y_test)
            res[track]['test score std'] = 0.0

            # iterates = model.n_iter_
            # loss = model.loss_curve_
            # model.validation_scores_

        figname = os.path.join(self.dir_['result'],'{} gusto_dim_red_traintime.png'.format(self.prefix))
        title = "Training time for NN, SVM, XGB"
        xlabel = ""
        ylabel = "Time"
        learning_curves_time(figname, title, xlabel, ylabel, res, ylim=None)

        # the error plot for different algorithms

        figname = os.path.join(self.dir_['result'],'{} gusto_dim_red_accuracy.png'.format(self.prefix))
        title = "Accuracy rates for NN, SVM, XGB"
        xlabel = ""
        ylabel = "Accuracy"
        plot_accuracy_error(figname, title, xlabel, ylabel, res, ylim=None)



