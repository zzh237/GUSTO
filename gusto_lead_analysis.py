from gusto_train_algo.classification.gusto_train_nn import *
from gusto_train_algo.classification.gusto_train_bn import *
from gusto_train_algo.classification.gusto_train_svm import *

from gusto_data_process.gusto_data_process import *
from gusto_train_algo.regression.gusto_train_gb import *
from gusto_train_algo.regression.gusto_train_lr import *
from gusto_train_algo.regression.gusto_train_uf import *
from plotting import *
from gusto_train_algo.classification.gusto_train_xgb import *
import sys
# metric = sys.argv[1]
metric = 'quality'
head_dir = ""
abspath = os.path.abspath(head_dir)

if abspath not in sys.path:
    sys.path.append(abspath)


df_DIR = os.path.abspath(os.path.join(head_dir, 'gusto_data_process\\data'))
model_DIR = os.path.abspath(os.path.join(head_dir, 'gusto_result_store\\model'))
result_DIR = os.path.abspath(os.path.join(head_dir, 'gusto_result_store\\result'))

dir_place = dict()
dir_place['df']=df_DIR
dir_place['model']=model_DIR
dir_place['result']=result_DIR

#change metric for each run
# metric: "quality" ___for classification of converted/non converted
# metric: "quality_quality_cost" ___for classification of converted/non converted, add plan cost
# metric: "conversion time"___for regression predict the conversion time, quick or slow
config = {'train':True,'metric': metric}
# config_file = r'..\tests\json_for_tests.json'
# config = GustoJasonConig(config_file)
# config_file = r'..\tests\json_for_tests.json'
# json_from_file = ''
# with open(config_file, 'r') as my_json:
#     json_from_file = my_json.read()


# df = pd.read_excel(os.path.join(df_DIR, 'Zhi Zhang_ gusto_leads_sample.csv.xlsx'), sheetname=0)
# df  = GustoDataProcess(config, dir_place).preprocess_df(df)
# df.to_hdf(os.path.join(df_DIR, 'datasets.hdf'), 'data', complib='blosc', complevel=9)

df = pd.read_hdf(os.path.join(df_DIR,'datasets.hdf'),'data')
train_X, train_y, test_X, test_y, columns_X = GustoDataProcess(config, dir_place).get_X_y(df)

title = config['metric']
if config['train'] == True:
    if title == 'quality' or title == 'quality_cost':
        trainer = GustoTrainNN(train_X,train_y,columns_X,dir_place, title)
        trainer.train()
        trainer = GustoTrainSVM(train_X,train_y,columns_X,dir_place, title)
        trainer.train()
        trainer = GustoTrainXGB(train_X,train_y,columns_X,dir_place, title)
        trainer.train()
    if title == 'conversion time':
        trainer = GustoTrainUF(train_X, train_y, columns_X, dir_place, title)
        trainer.train()
        trainer = GustoTrainGB(train_X, train_y, test_X, test_y, columns_X, dir_place, title)
        trainer.train()
        trainer = GustoTrainLR(train_X, train_y, columns_X, dir_place, title)
        trainer.train()
        # trainer = GustoTrainMLPR(train_X, train_y, columns_X, dir_place, title)
        # trainer.train()


Plotter(dir_place, test_X, test_y, title).plot()


sys.exit(0)


