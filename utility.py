import os
import sys
import cv2
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from scipy import linalg
import scipy.sparse as sps
from scipy.linalg import pinv
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.mixture import GaussianMixture as GMM
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin,BaseEstimator


rng = np.random.RandomState(42)


def plot_error_bar(axisticks, axislabels, value, color, width, ax=None):
    # the width of the bars

    mean = value['mean']
    std = value['std']

    rects = ax.bar(axisticks + width, mean, 0.1, color=color, yerr=std)

    # women_means = (25, 32, 34, 20, 25)
    # women_std = (3, 5, 2, 3, 3)
    # rects2 = ax.bar(ind + width, women_means, width, color='y', yerr=women_std)

    # add some text for labels, title and axes ticks

    ax.set_xticks(axisticks)
    ax.set_xticklabels(axislabels)

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            height = np.around(height, 2)
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    '%.2f' % float(height),
                    ha='center', va='bottom', fontsize=8)

    autolabel(rects)

    # plt.show()

    return ax

def reconstructionError(projections,X):
    W = projections.components_
    if sps.issparse(W):
        W = W.todense()
    p = pinv(W)
    reconstructed = (np.dot(np.dot(p,W),(X.T))).T # Unproject projected data
    errors = np.square(X-reconstructed)
    return np.nanmean(errors)

def pairwiseDistCorr(X1, X2):
    assert X1.shape[0] == X2.shape[0]
    d1 = pairwise_distances(X1)
    d2 = pairwise_distances(X2)
    return np.corrcoef(d1.ravel(), d2.ravel())[0, 1]


class GMMT(GMM):
    def transform(self,X):
        return self.predict_proba(X)

# http://datascience.stackexchange.com/questions/6683/feature-selection-using-feature-importances-in-random-forests-with-scikit-learn
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

def learning_curves_time(figname, title, xlabel, ylabel, values, ylim=None):
    N = len(values)
    ind = np.arange(N)  # the x locations for the groups

    allvalues = ((key, values[key]['training time']) for key in values.keys())

    xaxis_labels, train_time = zip(*allvalues)

    series = [(train_time,)]

    widths = [i * 0.15 for i in range(len(series))]

    fig, ax = plt.subplots(1, 1)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    ax.grid(False)

    # R = np.linspace(.1, 0.8, len(series))
    # G = np.linspace(.3, 0.1, len(series))
    # B = np.linspace(.7, 0.1, len(series))

    R = np.random.random_sample((len(series),))
    G = np.random.random_sample((len(series),))

    B = np.random.random_sample((len(series),))
    colors = zip(R, G, B)

    for i, item in enumerate(series):
        # train_sizes_abs, training_score, test_score, training_time, name = j
        value = dict()
        value['mean'] = item[0]
        value['std'] = 0
        width = widths[i]
        color = colors[i]

        ax = plot_error_bar(ind, xaxis_labels, value, color, width, ax=ax)

    # ax.legend((rects), ('Accuracy'))
    ax.legend().set_visible(False)
    # ax.legend('training time', loc='upper center', shadow=True, fontsize='small')

    # ax.legend(loc='upper center', shadow=True, fontsize='medium')

    ymin, ymax = ax.get_ylim()
    ymax = 1.2 * ymax
    ax.set_ylim((ymin, ymax))

    # plt.xticks(ind, xaxis_labels, rotation=90)
    plt.tight_layout()

    fig.savefig(figname, bbox_inches='tight')


def plot_accuracy_error(figname, title, xlabel, ylabel, values, ylim=None):
    N = len(values)
    ind = np.arange(N)  # the x locations for the groups



    allvalues = ((key, values[key]['cv score'], values[key]['cv score std'],
               values[key]['train score'], values[key]['train score std'],
                  values[key]['test score'], values[key]['test score std']  ) for key in values.keys())

    xaxis_labels, cv_error_mean, cv_error_std, train_error_mean, train_error_std, test_error_mean, test_error_std = zip(*allvalues)

    series = [(train_error_mean, train_error_std), (cv_error_mean, cv_error_std), (test_error_mean, test_error_std)]

    widths = [i * 0.15 for i in range(len(series))]

    fig, ax = plt.subplots(1, 1)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    ax.grid(False)



    # R = np.linspace(.1, 0.8, len(series))
    # G = np.linspace(.3, 0.1, len(series))
    # B = np.linspace(.7, 0.1, len(series))

    R = np.random.random_sample((len(series),))
    G = np.random.random_sample((len(series),))

    B = np.random.random_sample((len(series),))
    colors = zip(R, G, B)

    for i, item in enumerate(series):
        # train_sizes_abs, training_score, test_score, training_time, name = j
        value = dict()
        value['mean'] = item[0]
        value['std'] = item[1]
        width = widths[i]
        color = colors[i]

        ax = plot_error_bar(ind, xaxis_labels, value, color, width, ax=ax)

    # ax.legend((rects), ('Accuracy'))

    ax.legend(('train','cv', 'test'), loc='upper right', shadow=True, fontsize='small')

    # ax.legend(loc='upper center', shadow=True, fontsize='medium')

    ymin, ymax = ax.get_ylim()
    ymax = 1.2 * ymax
    ax.set_ylim((ymin, ymax))

    # plt.xticks(ind, xaxis_labels, rotation=90)
    plt.tight_layout()

    fig.savefig(figname, bbox_inches='tight')




def learning_iterations_performance(figname, title, xlabel, ylabel, values, ylim=None):

    allvalues = ((key, values[key]['model']) for key in list(values.keys()))
    xaxis_labels, model = zip(*allvalues)


    R = np.random.random_sample((len(model),))
    G = np.random.random_sample((len(model),))
    B = np.random.random_sample((len(model),))


    colors = zip(R, G, B)

    # colors=['g','r']

    linestyles = ['-.', ':', '-', '--',':']
    markers = ['.','p','v','o','*']
    print xaxis_labels

    fig, ax = plt.subplots(1, 1)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(False)

    for i, j in enumerate(model):
        mlp = j
        label = xaxis_labels[i]
        niter = mlp.n_iter_
        # ax.plot(np.arange(1, niter+1), np.asarray(mlp.loss_curve_), label=label, **plot_arg)
        ax.plot(mlp.validation_scores_, label=label, c=colors[i], marker = markers[i], linestyle=linestyles[i])


        # iterations, mean, std = j
        # name = labels[i]
        # value = dict()
        # value['mean'] = mean
        # value['std'] = 0.0
        # color = colors[i]
        #
        # ax = plot_line_chart(iterations, value, name, color, ax=ax)
    # labels, _ = zip(*values)
    labels = xaxis_labels

    ax.legend(ax.get_lines(), labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.savefig(figname,  bbox_inches='tight')