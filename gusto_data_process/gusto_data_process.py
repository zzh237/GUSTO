import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import sys
import collections
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import os
class GustoDataProcess():

    def __init__(self, config, dir):
        # self.df = df
        self.config = config
        self.dir_ = dir

    def preprocess_df(self, df):
        # temp = df[['id','original_mql_at','joined_at','converted']]
        if self.config['metric'] == 'conversion time':
            df = df.loc[df['converted']==True]

        df['join_days'] = df.apply(lambda x: abs(pd.Timedelta(x['joined_at']-x['original_mql_at']).days) if x['converted'] == True else np.NaN, axis=1)
        days1 = df.loc[df['converted']==True, 'join_days'].values
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax = sns.kdeplot(days1)
        fig.savefig(os.path.join(self.dir_['result'], 'hist_conversion_days.png'))

        days1 = np.log(days1+1)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax = sns.kdeplot(days1)
        fig.savefig(os.path.join(self.dir_['result'], 'log_hist_conversion_days.png'))

        employees = df['number_of_employees'].values
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax = sns.kdeplot(employees)
        fig.savefig(os.path.join(self.dir_['result'], 'log_hist_employees.png'))

        df['plan cost'] = 39.0 + df['lead_signup_size'] * 6.0
        plancost = df['plan cost'].values
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax = sns.kdeplot(plancost)
        fig.savefig(os.path.join(self.dir_['result'], 'log_hist_plancost.png'))
        # to_log = ['join_days','number_of_employees', 'plan cost', 'lead_signup_size']
        # to_log = ['join_days']

        # df[to_log] = df[to_log].applymap(lambda x: np.log(x + 1))



        #log transform the data,

        # cats = pd.qcut(days1, 4)
        # print pd.value_counts(cats)
        #
        # cats = pd.cut(days1, 4)
        # print pd.value_counts(cats)
        # sys.exit(0)

        days1 = days1[:, np.newaxis]
        days1_idx = df.loc[df['converted'] == True, 'id'].values

        days1 = MinMaxScaler().fit_transform(days1)
        days1 = days1.flatten()


        days1 = pd.qcut(days1,4, labels = False)
        # print np.max(days1), np.min(days1)
        days1 = 4 - days1
        days0_idx = df.loc[df['converted']==False, 'id'].values
        y0 = np.zeros(days0_idx.shape[0])
        y = np.concatenate([days1, y0])
        idx = np.concatenate([days1_idx, days0_idx])
        temp2  = pd.DataFrame({'id':idx, 'Class':y})
        df = pd.merge(left=df,
                      right=temp2.set_index('id', drop=True),
                      left_on=['id'],
                       how='left', right_index=True)

        df['Class'] = df['Class'].astype(int)
        df = self.process_x(df)

        return df

    def process_x(self,df):
        col_name = df.columns
        col_idx = [df.columns.get_loc(c) for c in df.columns]
        col = zip(col_name,col_idx)
        encode_cols = range(7,23)
        encode_cols.append(5)
        encode_cols.append(4)
        encode_cols.sort()
        encode_cols_names = [col_name[i] for i in encode_cols]
        # df['area_code'] = df['area_code'].replace(np.NaN, 'unknown')
        # temp = df['area_code']
        # print temp.head(1000)
        # print temp.value_counts()
        # #
        # sys.exit(0)
        le = LabelEncoder()
        threshold = 30
        for col in encode_cols_names:
            # print "___", col
            df[col] = df[col].replace(np.NaN, 'unknown')

            vc = df[col].value_counts()
            # print vc
            vals_to_remove = vc[vc <= threshold].index.values
            df[col].loc[df[col].isin(vals_to_remove)] = 'other'
            vc = df[col].value_counts()
            print len(vc.index)
            if len(vc.index) > 2:
                df = pd.get_dummies(df, prefix=[col], columns=[col])
            else:
                df[col] = le.fit_transform(df[col])
            # print df[col].value_counts()
            # df[col] = le.fit_transform(df[col])
            # one_hot = pd.get_dummies(df[col])
            # df = df.drop(col, axis=1)
            # df = df.join(one_hot)
            # print df.columns
            # print df[col].value_counts()

        # df = pd.get_dummies(df, prefix=encode_cols_names, columns=encode_cols_names)

        # encode_X = df.iloc[:, encode_cols].as_matrix()
        # encode_X = le.fit_transform(encode_X)
        # enc = OneHotEncoder()
        # encode_X = enc.fit_transform(encode_X).A
        # non_encode_cols = [3, 6]
        # non_encode_X = df.iloc[:, non_encode_cols]
        # X = np.hstack((encode_X, non_encode_X))
        # print encode_X
        # X = MinMaxScaler().fit_transform(X)

        return df

    def trin_test_split(self, X, y):
        CVClass = StratifiedShuffleSplit
        cv = CVClass(test_size=0.2, random_state=np.random.seed(42))

        train, test = next(cv.split(X, y))
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]
        return X_train, y_train, X_test, y_test

    def train_test_regressor_split(self, X, y):

        msk = np.random.rand(len(X)) < 0.8

        X_train, y_train = X[msk], y[msk]

        X_test, y_test = X[~msk], y[~msk]

        return X_train, y_train, X_test, y_test




    def get_quality_X_y(self, df):
        columns_X = [i for i in df.columns if
                     i not in ['id', 'plan cost', 'number_of_employees', 'original_mql_at', 'joined_at', 'Class', 'join_days', 'converted']]
        columns_y = ['converted']

        miss_columns = df.columns[df.isnull().any()].tolist()
        miss_columns_X = [i for i in miss_columns if i in columns_X]
        for col in miss_columns_X:
            null_index = df[col].isnull()
            df.loc[~null_index, [col]] = MinMaxScaler().fit_transform(df.loc[~null_index, [col]])
            # https: // stackoverflow.com / questions / 9365982 / missing - values - in -scikits - machine - learning
            df.loc[null_index, [col]] = -1.0
        print df[miss_columns_X].describe()
        X, y = df[columns_X].values, df[columns_y].values.flatten()

        return X, y, columns_X

    def get_quality_cost_X_y(self, df):
        columns_X = [i for i in df.columns if
                     i not in ['id', 'number_of_employees', 'original_mql_at', 'joined_at', 'Class',
                               'join_days', 'converted']]
        columns_y = ['converted']

        miss_columns = df.columns[df.isnull().any()].tolist()
        miss_columns_X = [i for i in miss_columns if i in columns_X]
        for col in miss_columns_X:
            null_index = df[col].isnull()
            df.loc[~null_index, [col]] = MinMaxScaler().fit_transform(df.loc[~null_index, [col]])
            # https: // stackoverflow.com / questions / 9365982 / missing - values - in -scikits - machine - learning
            df.loc[null_index, [col]] = -1.0
        print df[miss_columns_X].describe()
        X, y = df[columns_X].values, df[columns_y].values.flatten()

        return X, y, columns_X


    def get_conversion_time_X_y(self, df):
        columns_X = [i for i in df.columns if
                     i not in ['id', 'plan cost', 'original_mql_at', 'joined_at', 'Class', 'join_days', 'converted']]
        columns_y = ['join_days']

        miss_columns = df.columns[df.isnull().any()].tolist()
        miss_columns_X = [i for i in miss_columns if i in columns_X]
        for col in miss_columns_X:
            null_index = df[col].isnull()
            df.loc[~null_index, [col]] = MinMaxScaler().fit_transform(df.loc[~null_index, [col]])
            # https: // stackoverflow.com / questions / 9365982 / missing - values - in -scikits - machine - learning
            df.loc[null_index, [col]] = -1.0
        print df[miss_columns_X].describe()
        X= df[columns_X].values
        y = df[columns_y].values.flatten()

        return X, y, columns_X


    def get_X_y(self,df):
        if self.config['metric'] == 'quality':
            X, y, columns_X = self.get_quality_X_y(df)
            X_train, y_train, X_test, y_test = self.trin_test_split(X, y)
        if self.config['metric'] == 'fraud':
            pass

        if self.config['metric'] == 'quality_cost':
            X, y, columns_X = self.get_quality_cost_X_y(df)
            X_train, y_train, X_test, y_test = self.trin_test_split(X, y)

        if self.config['metric'] == 'conversion time':
            X, y, columns_X = self.get_conversion_time_X_y(df)
            X_train, y_train, X_test, y_test = self.train_test_regressor_split(X, y)


        return X_train, y_train, X_test, y_test, columns_X

