#coding:utf-8
import pandas as pd
import numpy as np
import fasttext
import gensim
import sys
import tensorflow as tf
import keras
import importlib
import lightgbm as lgb
import xgboost as xgb
importlib.reload(sys)
from keras import backend as K
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec

from sklearn.linear_model import Ridge
from sklearn.cross_validation import StratifiedKFold,KFold
from scipy.sparse import csr_matrix, hstack,coo_matrix
from sklearn.metrics import roc_auc_score, mean_squared_error

from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import text, sequence
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,HashingVectorizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical,plot_model
from keras.models import Sequential,Model
from keras.layers import Dense, Input, Flatten, Dropout,Conv1D, MaxPooling1D,Embedding,Activation,LSTM
from keras.layers.recurrent import LSTM, GRU 
from keras.layers import SpatialDropout1D, concatenate,Concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D,GlobalMaxPool1D
from keras.callbacks import Callback
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.models import load_model

def get_data():

    traina = pd.read_csv('../data/train_pre.csv')
    trainb = pd.read_csv('../data/train_pre_B.csv')
    testa = pd.read_csv('../data/test_pre.csv')
    testb = pd.read_csv('../data/test_pre_B.csv')
    train_data = pd.concat([traina,trainb],axis=0)
    test_data = testb.copy()
    data = pd.concat([traina,trainb,testa,testb],axis=0)
    train_data.columns = ['Id','cutted_Dis','Score']
    test_data.columns = ['Id','cutted_Dis']
    data.columns = ['cutted_Dis','Id','Score']


    traina0 = pd.read_csv('../data/train_first.csv')
    trainb0 = pd.read_csv('../data/train_second.csv')
    testa0 = pd.read_csv('../data/predict_first.csv')
    testb0 = pd.read_csv('../data/predict_second.csv')
    train_data0 = pd.concat([traina0,trainb0],axis=0)
    test_data0 = testb0.copy()
    data0 = pd.concat([traina0,trainb0,testa0,testb0],axis=0)
    print('train_data len: ',len(train_data))
    train_data = pd.merge(train_data,train_data0[['Id','Discuss']],on='Id')
    print('train_data len: ',len(train_data))

    print('test_data len: ',len(test_data))
    test_data = pd.merge(test_data,test_data0[['Id','Discuss']],on='Id')
    print('test_data len: ',len(test_data))

    print('data len: ',len(data))
    data = pd.merge(data,data0[['Id','Discuss']],on='Id')
    print('data len: ',len(data))

    a = pd.read_csv('../data/full_same_dis_filled_180316.csv',encoding='GBK')
    c = pd.read_csv('../data/full_unknown_dis_filled_180316.csv',encoding='GBK')

    return data,train_data,test_data,test_data['Id'],a,c

def get_label(pred):
    score = np.array([1,2,3,4,5])
    pred2 = []
    for p in pred:
        pr = np.sum(p * score)
        pred2.append(pr)
    return np.array(pred2)

def xx_mse_s(y_true,y_pre):
    y_true = y_true
    y_pre = pd.DataFrame({'res':list(y_pre)})

    y_pre['res'] = y_pre['res'].astype(int)
    return 1 / ( 1 + mean_squared_error(y_true,y_pre['res'].values)**0.5)

data,train_data,test_data,test_id,same,unkown = get_data()
lgb_ci_feat_init = pd.read_csv('../stack_cache/5_fold_lgb_ci_feat_init.csv')
lgb_zi_feat_init = pd.read_csv('../stack_cache/5_fold_lgb_zi_feat_init.csv')
ridge_ci_feat_init = pd.read_csv('../stack_cache/5_fold_ridge_ci_feat_init.csv')
ridge_zi_feat_init = pd.read_csv('../stack_cache/5_fold_ridge_zi_feat_init.csv')
mlp_ci_feat_init = pd.read_csv('../stack_cache/5_fold_mlp_ci_feat_init.csv')
mlp_zi_feat_init = pd.read_csv('../stack_cache/5_fold_mlp_zi_feat_init.csv')
bayes_ci_feat_init = pd.read_csv('../stack_cache/5_fold_bayes_ci_feat_init.csv')
keras_mlp_ci_feat_init = pd.read_csv('../stack_cache/5_fold_keras_mlp_ci_feat_init.csv')
keras_mlp_zi_feat_init = pd.read_csv('../stack_cache/5_fold_keras_mlp_zi_feat_init.csv')
nn_model1_ci_feat_init = pd.read_csv('../stack_cache/5_fold_nn_model1_ci_feat_init.csv')
nn_model1_zi_feat_init = pd.read_csv('../stack_cache/5_fold_nn_model1_zi_feat_init.csv')
nn_model_ince_ci_feat_init = pd.read_csv('../stack_cache/5_fold_nn_model_ince_ci_feat_init.csv')

lgb_ci_prob_init = pd.read_csv('../stack_cache/5_fold_lgb_ci_prob_init.csv')
lgb_zi_prob_init = pd.read_csv('../stack_cache/5_fold_lgb_zi_prob_init.csv')
ridge_ci_prob_init = pd.read_csv('../stack_cache/5_fold_ridge_ci_prob_init.csv')
ridge_zi_prob_init = pd.read_csv('../stack_cache/5_fold_ridge_zi_prob_init.csv')
mlp_ci_prob_init = pd.read_csv('../stack_cache/5_fold_mlp_ci_prob_init.csv')
mlp_zi_prob_init = pd.read_csv('../stack_cache/5_fold_mlp_zi_prob_init.csv')
bayes_ci_prob_init = pd.read_csv('../stack_cache/5_fold_bayes_ci_prob_init.csv')
keras_mlp_ci_prob_init = pd.read_csv('../stack_cache/5_fold_keras_mlp_ci_prob_init.csv')
keras_mlp_zi_prob_init = pd.read_csv('../stack_cache/5_fold_keras_mlp_zi_prob_init.csv')
nn_model1_ci_prob_init = pd.read_csv('../stack_cache/5_fold_nn_model1_ci_prob_init.csv')
nn_model1_zi_prob_init = pd.read_csv('../stack_cache/5_fold_nn_model1_zi_prob_init.csv')
nn_model_ince_ci_prob_init = pd.read_csv('../stack_cache/5_fold_nn_model_ince_ci_prob_init.csv')

train_feat = lgb_ci_feat_init.copy()
train_feat = pd.merge(train_feat,lgb_zi_feat_init,on='Id')
train_feat = pd.merge(train_feat,ridge_ci_feat_init,on='Id')
train_feat = pd.merge(train_feat,ridge_zi_feat_init,on='Id')
train_feat = pd.merge(train_feat,mlp_ci_feat_init,on='Id')
train_feat = pd.merge(train_feat,mlp_zi_feat_init,on='Id')
train_feat = pd.merge(train_feat,bayes_ci_feat_init,on='Id')
train_feat = pd.merge(train_feat,keras_mlp_ci_feat_init,on='Id')
train_feat = pd.merge(train_feat,keras_mlp_zi_feat_init,on='Id')
train_feat = pd.merge(train_feat,nn_model1_ci_feat_init,on='Id')
train_feat = pd.merge(train_feat,nn_model1_zi_feat_init,on='Id')
train_feat = pd.merge(train_feat,nn_model_ince_ci_feat_init,on='Id')

train_feat['label'] = train_data['Score']
test_prob = lgb_ci_prob_init.copy()
test_prob = pd.merge(test_prob,lgb_zi_prob_init,on='Id')
test_prob = pd.merge(test_prob,ridge_ci_prob_init,on='Id')
test_prob = pd.merge(test_prob,ridge_zi_prob_init,on='Id')
test_prob = pd.merge(test_prob,mlp_ci_prob_init,on='Id')
test_prob = pd.merge(test_prob,mlp_zi_prob_init,on='Id')
test_prob = pd.merge(test_prob,bayes_ci_prob_init,on='Id')
test_prob = pd.merge(test_prob,keras_mlp_ci_prob_init,on='Id')
test_prob = pd.merge(test_prob,keras_mlp_zi_prob_init,on='Id')
test_prob = pd.merge(test_prob,nn_model1_ci_prob_init,on='Id')
test_prob = pd.merge(test_prob,nn_model1_zi_prob_init,on='Id')
test_prob = pd.merge(test_prob,nn_model_ince_ci_prob_init,on='Id')

stack_train = train_feat.iloc[:,1:-1]
stack_test = test_prob.iloc[:,1:]
stack_test.columns = stack_train.columns

def n_folds_train_xgb(n_folds,y,X,test_hh):
    skf = KFold(X.shape[0], n_folds,random_state=2018,shuffle=True)
    # skf = list(StratifiedKFold(y, n_folds,random_state=2018,shuffle=True))

#     params = {
#     'learning_rate': 0.01,
#     'objective': 'regression_l2',
#     'metric': 'mse',
#     'num_leaves': 128,
#     'bagging_fraction': 0.7,
#     'feature_fraction': 0.7,
#     # 'colsample_bylevel': 0.7,
#     'nthread': -1
#     }
    params = {
        'objective': 'reg:linear',
        'eta': 0.01,
        'colsample_bytree': 0.8,
        #'min_child_weight': 2,
       # 'max_depth': 5,
        'subsample': 0.8,
        #'alpha': 10,
        #'gamma': 30,
       # 'lambda': 5,
        'silent': 1,

        'verbose_eval': True,
       # 'nthread': 8,
        'eval_metric': 'rmse',
       # 'scale_pos_weight': 10,
        'seed': 201804,
        'missing': -1
    }
    plst = list(params.items())
    num_round = 10000
    
    daset_blend_train = np.zeros((X.shape[0], 1))
    daset_blend_test = np.zeros((test_hh.shape[0], 1))
    daset_blend_test_0 = np.zeros((test_hh.shape[0], len(skf)))

    for i ,(train,test) in enumerate(skf):
        X_train, y_train, X_test, y_test = X[train], y.values[train], X[test], y.values[test]
#         X_train = pd.DataFrame(X_train)
#         X_train.columns = stack_train.columns
#         print(X_train.head())
#         X_test = pd.DataFrame(X_test)
#         X_test.columns = stack_train.columns       
#         print(X_test.head()) 
        
        train1 = xgb.DMatrix(X_train, label=y_train)
        valid1 = xgb.DMatrix(X_test, label=y_test) 
        evallist = [(train1, 'train'), (valid1, 'eval')]
        model = xgb.train(plst, train1, num_round, evallist,verbose_eval=100,early_stopping_rounds=30)
        y_sub = model.predict(valid1,ntree_limit=model.best_ntree_limit)
        print(1 /(1 + mean_squared_error(y_test,y_sub)**0.5))
        print(xx_mse_s(y_test, y_sub))
        daset_blend_train[test,0]=y_sub
        test_1 = xgb.DMatrix(test_hh)
        daset_blend_test_0[:,i]=model.predict(test_1,ntree_limit=model.best_ntree_limit)
    daset_blend_test[:, 0] = daset_blend_test_0.mean(1)

    return daset_blend_train,daset_blend_test


a, b = n_folds_train_xgb(5,train_feat['label'],stack_train.values,stack_test.values)

def n_folds_train_lgb(n_folds,y,X,test_hh):
    skf = KFold(X.shape[0], n_folds,random_state=2018,shuffle=True)
    # skf = list(StratifiedKFold(y, n_folds,random_state=2018,shuffle=True))

    params = {
    'learning_rate': 0.01,
    'objective': 'regression_l2',
    'metric': 'mse',
    'num_leaves': 128,
    'bagging_fraction': 0.7,
    'feature_fraction': 0.7,
    # 'colsample_bylevel': 0.7,
    'nthread': -1
    }

    daset_blend_train = np.zeros((X.shape[0], 1))
    daset_blend_test = np.zeros((test_hh.shape[0], 1))
    daset_blend_test_0 = np.zeros((test_hh.shape[0], len(skf)))

    for i ,(train,test) in enumerate(skf):
        X_train, y_train, X_test, y_test = X[train], y.values[train], X[test], y.values[test]
        train1 = lgb.Dataset(X_train, label=y_train)
        valid1 = lgb.Dataset(X_test, label=y_test) 
        model = lgb.train(params, train1, num_boost_round=10000, valid_sets=[train1, valid1],verbose_eval=100,early_stopping_rounds=10)
        y_sub = model.predict(X_test, num_iteration=model.best_iteration)
        print(1 /(1 + mean_squared_error(y_test,y_sub)**0.5))
        print(xx_mse_s(y_test, y_sub))
        daset_blend_train[test,0]=y_sub
        daset_blend_test_0[:,i]=model.predict(test_hh, num_iteration=model.best_iteration)
    daset_blend_test[:, 0] = daset_blend_test_0.mean(1)

    return daset_blend_train,daset_blend_test

c, d = n_folds_train_lgb(5,train_feat['label'],stack_train.values,stack_test.values)

res = pd.DataFrame(test_data['Id'])
res['xgb_prob'] = b
res['lgb_prob'] = d
res['stack_all_init'] = (res['xgb_prob'] + res['lgb_prob'])/2
res[['Id','stack_all_init']].to_csv('../stack_cache/stack_all_init.csv',index=False)