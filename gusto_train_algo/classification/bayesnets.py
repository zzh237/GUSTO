from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.externals import six
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np


class BayesNetsClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, n_iter):
        self.n_iter = n_iter

    def fit(self, X, y=None):
        # solve the bayes network structures
        self.class_edges = self.get_edges_for_each_class(X, y)
        # EM algorithms to solve the parameters
        self.class_thetas = self.get_thetas_for_each_class(X, y, self.class_edges)
        # the prior probability for the algorithm
        self.log_p_h = self.get_log_p_h(y)
        self.label_set = np.unique(y)

        return self

    def predict_proba(self, X):
        prob = self.evaluate_predictions(X, self.label_set, self.class_edges, self.class_thetas, self.log_p_h)

        return prob
    def predict(self, X):
        pred = self.evaluate_predictions(X, self.label_set, self.class_edges, self.class_thetas, self.log_p_h)[:, 0]
        print pred
        return pred


    def evaluate_predictions(self, test_features, label_set, class_edges, class_thetas, log_p_h):

        pred_lab = np.zeros((test_features.shape[0], 2))
        # the first column shoud be the predicted label, and the second column should be the probability of that label.
        for i in range(len(test_features)):
            x = test_features[i, :]
            res = self.p_h_given_x_theta(x, class_edges, class_thetas, log_p_h, label_set)

            # print(type(res))
            max_value = np.max(res)
            max_index = np.argmax(res)
            min_value = np.min(res)

            # predicted label
            max_p_label = label_set[max_index]
            pred_lab[i, 0] = max_p_label
            # probability of that label
            pred_lab[i, 1] = max_value

            # pred_lab[i, 0] = label_set[max_index]
            # probability of that label
            # pred_lab[i, 1] = max_value
        return pred_lab

    # def _get_label(self, ...):
        # Need to have a way of knowing which label being classified
        # by OneVsRestClassifier (self.class_label)

    def corr(self, x, y):
        if x.shape[0] == 0:
            rowvar = 1
        else:
            rowvar = 0
        return np.corrcoef(np.asarray(x), np.asarray(y), rowvar=rowvar)[0, 1]

    def mutual_information(self, x, y):
        return -0.5 * np.log(1 - self.corr(x, y) ** 2)

    def mutual_info_all(self, M):
        f_num = M.shape[1]  # feature number
        mi_ary = np.zeros((f_num, f_num))
        for i in range(f_num - 1):
            for j in range(i + 1, f_num):  # want to keep diag = 0
                x = M[:, i]
                y = M[:, j]
                mi_ary[i, j] = self.mutual_information(x, y)
        return mi_ary

    def chowliu(self, features):
        MI = self.mutual_info_all(features)
        adjacency_matrix = minimum_spanning_tree(-MI)
        return adjacency_matrix

    def get_edge_list(self, mat):
        f_num = mat.shape[0]
        edges = []
        for k in range(f_num):
            lst = np.nonzero(mat[k, :])[1]
            # k is parent, j is child
            new_edges = [(k, j) for j in lst]
            edges.extend(new_edges)
        return edges

    # np.random.seed(1)
    # x = np.random.randn(20)
    # y = x + 0.1 * np.random.randn(20)
    # z = -x + 0.1 * np.random.randn(20)
    # q = z + 0.1 * np.random.randn(20)
    # w = y + 0.1 * np.random.rand(20)
    # features = np.asmatrix([x, y, z, q, w]).transpose()
    # adjacency = chowliu(features)
    # names = ['x', 'y', 'z', 'q', 'w']
    # edges = get_edge_list(adjacency)
    # print('Edge list: ', [(names[i] + '->' + names[j]) for (i, j) in edges])
    #
    # edges = get_edge_list(adjacency)
    #
    # assert (set(edges) == set([(0, 1), (0, 2), (1, 4), (2, 3)]))

    def get_label_subsets(self, train_labels):
        label_set = np.unique(train_labels)  # get 5 label numbers
        label_sample_map = {}  # a label to sample index map
        for j in label_set:
            label_sample_map[j] = []
        for i in range(len(train_labels)):
            for j in label_set:
                if train_labels[i] == j:
                    label_sample_map[j].append(i)
        return label_sample_map

    def get_edges_for_each_class(self, train_features, train_labels):
        label_set = np.unique(train_labels)  # get 5 label numbers
        label_sample_map = self.get_label_subsets(train_labels)
        class_edges = {}
        for i in label_set:
            class_edges[i] = self.get_edge_list(self.chowliu(train_features[label_sample_map[i], :]))
        return class_edges



    #
    # d = 1000
    # x = np.zeros((d,2))
    # x[:,0] = 1.0 + np.random.randn(d,)
    # x[:,1] = 0.5 + 2.0*x[:,0] + np.random.randn(d,)
    # theta_j = [0.0,0.0]

    # compute log p(x_j|x_k,\theta_j), for the edge between j and k, given \theta_{j, 0} and \theta_{j,1}
    # Let sigma^2 =1
    def compute_lp_j_k(self, j, k, x, theta_j_0, theta_j_1):
        # temp = np.add([i - theta_j_0 for i in x[:,j]], [i*theta_j_1 for i in x[:,k]])
        # return -x.shape[0]*np.log(2*np.pi) - .5 * np.dot(temp, temp)
        return - .5 * np.log(2 * np.pi) - .5 * (x[j] - theta_j_0 - theta_j_1 * x[k]) ** 2

    # compute log p(x_r|\theta_r), for the root node, given theta_r.
    def compute_lp_r(self, x, theta_r):
        #     temp = [i - theta_r for i in x[:,0]]
        #     return -x.shape[0]*np.log(2*np.pi) - .5 * np.dot(temp, temp)
        return -.5 * np.log(2 * np.pi) - .5 * (x[0] - theta_r) ** 2

    def compute_lp_x_given_Theta(self, x, thetas, edges):
        lp = self.compute_lp_r(x, thetas[0, 0])
        for (k, j) in edges:
            # k is parent, j is child
            lp = lp + self.compute_lp_j_k(j, k, x, thetas[j, 0], thetas[j, 1])
        return lp

    # update for theta_{r,0}, r is the root_index (0 in our case)
    def compute_theta_r(self, x):  # x is the 403*1000 feature matrix
        return np.sum(x[:, 0]) / x.shape[0]

    # update for theta_{j, 0}, for link between j and k given theta_j_1.
    # k is parent, j is child
    def compute_theta_j_k_0(self, j, k, x, theta_j_1):
        return (np.sum(x[:, j]) - theta_j_1 * np.sum(x[:, k])) / x.shape[0]

    # update for theta_{j, 1}, for link between j and k given theta_j_0.
    # k is parent, j is child
    def compute_theta_j_k_1(self, j, k, x, theta_j_0):
        return (np.dot(x[:, j], x[:, k]) - theta_j_0 * np.sum(x[:, k])) / np.dot(x[:, k], x[:, k])

    def get_thetas_for_each_class(self, train_features, train_labels, class_edges):
        label_sample_map = self.get_label_subsets(train_labels)
        label_set = np.unique(train_labels)
        class_thetas = {}
        f_num = train_features.shape[1]
        for lab in label_set:
            c_samples = train_features[label_sample_map[lab], :]
            theta_r = self.compute_theta_r(c_samples)
            c_edge_list = class_edges[lab]
            thetas = np.zeros((f_num, 2))  # the first column shoud be j_0, and the second column should be j_1
            # the first row (thetas[0,:]) is for theta_r

            for (k, j) in c_edge_list:
                theta_j_1 = 0
                # should do coordinate ascent using the function
                # compute_theta_j_k_0 and compute_theta_j_k_1 here
                for z in range(self.n_iter):
                    theta_j_0 = self.compute_theta_j_k_0(j, k, c_samples, theta_j_1)
                    # print (z, theta_j_0)
                    theta_j_1 = self.compute_theta_j_k_1(j, k, c_samples, theta_j_0)
                    # print (z, theta_j_1)
                # set the optimal theta_j_0 and theta_j_1 for this the edge (k, j)
                thetas[j, 0] = theta_j_0
                thetas[j, 1] = theta_j_1
            thetas[0, 0] = theta_r
            # root has no parents
            thetas[0, 1] = np.nan
            class_thetas[lab] = thetas
        return class_thetas


    # get the prior probability rate
    def get_log_p_h(self, train_labels):
        label_set = np.unique(train_labels)
        log_p_h = np.zeros(len(np.unique(train_labels)))
        for k, i in enumerate(label_set):
            count = len(np.nonzero(train_labels == i)[0])
            log_p_h[k] = np.log(float(count) / float(len(train_labels)))
        return log_p_h



    def logsumexp(self, vec):
        m = np.max(vec, axis=0)
        return np.log(np.sum(np.exp(vec - m), axis=0)) + m

    def p_h_given_x_theta(self, x, class_edges, class_thetas, log_p_h, label_set):
        C = len(class_thetas)
        lognumerator = np.zeros(C)

        # implement Bayes rule here
        # compute log-numerators first and then normalize using logsumexp
        # there are more compact ways to do the normalization
        # feel free to rearrange the code, as long as you return correct
        # probabilities

        # for i in range(C):
        for k, i in enumerate(label_set):
            edges = class_edges[i]

            thetas = class_thetas[i]
            if len(edges) == 0:
                tree_prob = -100 + self.compute_lp_x_given_Theta(x, thetas, edges)
            else:
                tree_prob = self.compute_lp_x_given_Theta(x, thetas, edges)
            lognumerator[k] = tree_prob + log_p_h[k]

        # print(lognumerator)
        maxlognumerator = np.max(lognumerator)
        lognumerator = lognumerator - maxlognumerator
        # use logsumexp to compute denominator
        logdenominator = self.logsumexp(lognumerator)
        # print(logdenominator)

        numerator = np.exp(lognumerator)
        # print(numerator)
        denominator = np.exp(logdenominator)
        # print(denominator)
        # denominator = np.sum(numerator)
        # print(denominator)
        probs = numerator / denominator

        assert (np.all(probs >= 0))
        assert (np.abs(np.sum(probs) - 1.0) < 1e-5)
        return probs

    # def evaluate_predictions(test_features, test_labels, class_edges, class_thetas, log_p_h):
    #     label_set = np.unique(train_labels)
    #     pred_lab = np.zeros((test_features.shape[0], 2))
    #     # the first column shoud be the predicted label, and the second column should be the probability of that label.
    #     for i in range(len(test_labels)):
    #         x = test_features[i, :]
    #         res = p_h_given_x_theta(x, class_edges, class_thetas, log_p_h, label_set)
    #         # print(type(res))
    #         max_value = np.max(res)
    #         max_index = np.argmax(res)
    #         # predicted label
    #         pred_lab[i, 0] = label_set[max_index]
    #         # probability of that label
    #         pred_lab[i, 1] = max_value
    #     return pred_lab


    # print("Prediction Accuracy: {}".format(np.mean(pred_lab[:,0]==test_labels)))


    # def cal_accuracy(pred_lab, test_labels):
    #     res = []
    #     for i, pred in enumerate(pred_lab[:, 0]):
    #         d = test_labels[i]
    #         d_minus = d - 1
    #         d_plus = d + 1
    #         if pred == d or pred == d_minus or pred == d_plus:
    #             res.append(1)
    #         else:
    #             res.append(0)
    #     return res
    #
    # accuracy = np.mean(cal_accuracy(pred_lab, test_labels))
    #
    # print(accuracy)

# bow_clf = Pipeline([('vect', CountVectorizer(stop_words='english', min_df=1, max_df=0.9)),
#                     ('tfidf', TfidfTransformer(use_idf=False)),
#                     ('clf', SGDClassifier(loss='log', penalty='l2', alpha=1e-3, n_iter=5)),
#                    ])
# custom_clf = CustomClassifier(word_to_tag_dict)
#
# ovr_clf = OneVsRestClassifier(VotingClassifier(estimators=[('bow', bow_clf), ('custom', custom_clf)],
#                                                voting='soft'))
#
# params = { 'estimator_weights': ([1, 1], [1, 2], [2, 1]) }
# gs_clf = GridSearchCV(ovr_clf, params, n_jobs=-1, verbose=1, scoring='precision_samples')
#
# binarizer = MultiLabelBinarizer()
#
# gs_clf.fit(X, binarizer.fit_transform(y))


