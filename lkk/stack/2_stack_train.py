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
    a = pd.read_csv('../data/full_same_dis_filled_180316.csv',encoding='GBK')
    b = pd.read_csv('../data/full_tobe_classify_180316.csv',encoding='GBK')
    b['cutted_Dis'] = b['cutted_Dis'].fillna("空白")
    c = pd.read_csv('../data/full_unknown_dis_filled_180316.csv',encoding='GBK')
    test_init = pd.read_csv('../data/predict_second.csv')
    train_data = b[~b.Score.isnull()]
    test_data = b[b.Score.isnull()]
    b = b.reset_index()
    train_data = train_data.reset_index()
    test_data = test_data.reset_index()
    return b,train_data,test_data,test_init['Id'],a,c

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

model1_train_init = pd.read_csv('../stack_cache/stcking_ince___sfceshi_diejia_train_model1_no_shuffle.csv')
model1_test_init = pd.read_csv('../stack_cache/stcking_ince___sfceshi_diejia_test_model1_no_shuffle.csv')
model1_train_shuffle = pd.read_csv('../stack_cache/stcking_ince___sfceshi_diejia_train_model1_shuffle.csv')
model1_test_shuffle = pd.read_csv('../stack_cache/stcking_ince___sfceshi_diejia_test_model1_shuffle.csv')
model1_train_zi = pd.read_csv('../stack_cache/stcking_ince___sfceshi_diejia_train_model1_no_shuffle_zi.csv')
model1_test_zi = pd.read_csv('../stack_cache/stcking_ince___sfceshi_diejia_test_model1_no_shuffle_zi.csv')
model_ince_train_init = pd.read_csv('../stack_cache/stcking_ince___sfceshi_diejia_train_model_ince_no_shuffle.csv')
model_ince_test_init = pd.read_csv('../stack_cache/stcking_ince___sfceshi_diejia_test_model_ince_no_shuffle.csv')

duo_train = pd.read_csv('../stack_cache/st_duo_train.csv')
duo_test = pd.read_csv('../stack_cache/st_duo_test.csv')
lgb_ci_train = pd.read_csv('../stack_cache/5_fold_lgb_ci_feat.csv')
lgb_ci_test = pd.read_csv('../stack_cache/5_fold_lgb_ci_prob.csv')
lgb_zi_train = pd.read_csv('../stack_cache/5_fold_lgb_zi_feat.csv')
lgb_zi_test = pd.read_csv('../stack_cache/5_fold_lgb_zi_prob.csv')
ridge_ci_train = pd.read_csv('../stack_cache/5_fold_ridge_ci_feat.csv')
ridge_ci_test = pd.read_csv('../stack_cache/5_fold_ridge_ci_prob.csv')
ridge_zi_train = pd.read_csv('../stack_cache/5_fold_ridge_zi_feat.csv')
ridge_zi_test = pd.read_csv('../stack_cache/5_fold_ridge_zi_prob.csv')
mlp_ci_train = pd.read_csv('../stack_cache/5_fold_mlp_ci_feat.csv')
mlp_ci_test = pd.read_csv('../stack_cache/5_fold_mlp_ci_prob.csv')
mlp_zi_train = pd.read_csv('../stack_cache/5_fold_mlp_zi_feat.csv')
mlp_zi_test = pd.read_csv('../stack_cache/5_fold_mlp_zi_prob.csv')
bayes_ci_train = pd.read_csv('../stack_cache/5_fold_bayes_ci_feat.csv')
bayes_ci_test = pd.read_csv('../stack_cache/5_fold_bayes_ci_prob.csv')
keras_mlp_ci_train = pd.read_csv('../stack_cache/5_fold_keras_mlp_ci_feat.csv')
keras_mlp_ci_test = pd.read_csv('../stack_cache/5_fold_keras_mlp_ci_prob.csv')
keras_mlp_zi_train = pd.read_csv('../stack_cache/5_fold_keras_mlp_zi_feat.csv')
keras_mlp_zi_test = pd.read_csv('../stack_cache/5_fold_keras_mlp_zi_prob.csv')
fasttext_ci_train = pd.read_csv('../stack_cache/5_fold_fasttext_ci_feat.csv')
fasttext_ci_test = pd.read_csv('../stack_cache/5_fold_fasttext_ci_prob.csv')

train_feat = pd.DataFrame(train_data['Id'])
train_feat['model1_train_init'] = model1_train_init['train1']
train_feat['model1_train_shuffle'] = model1_train_shuffle['train1']
train_feat['model1_train_zi'] = model1_train_zi['train1']
train_feat['model_ince_train_init'] = model_ince_train_init['train1']
train_feat['duo_train'] = duo_train['train']
train_feat = pd.merge(train_feat,lgb_ci_train,on='Id')
train_feat = pd.merge(train_feat,lgb_zi_train,on='Id')
train_feat = pd.merge(train_feat,ridge_ci_train,on='Id')
train_feat = pd.merge(train_feat,ridge_zi_train,on='Id')
train_feat = pd.merge(train_feat,mlp_ci_train,on='Id')
train_feat = pd.merge(train_feat,mlp_zi_train,on='Id')
train_feat = pd.merge(train_feat,bayes_ci_train,on='Id')
train_feat = pd.merge(train_feat,keras_mlp_ci_train,on='Id')
train_feat = pd.merge(train_feat,keras_mlp_zi_train,on='Id')
train_feat = pd.merge(train_feat,fasttext_ci_train,on='Id')

test_feat = pd.DataFrame(test_data['Id'])
test_feat['model1_test_init'] = model1_test_init['Score']
test_feat['model1_test_shuffle'] = model1_test_shuffle['Score']
test_feat['model1_test_zi'] = model1_test_zi['Score']
test_feat['model_ince_test_init'] = model_ince_test_init['Score']
test_feat['duo_test'] = duo_test['test']
test_feat = pd.merge(test_feat,lgb_ci_test,on='Id')
test_feat = pd.merge(test_feat,lgb_zi_test,on='Id')
test_feat = pd.merge(test_feat,ridge_ci_test,on='Id')
test_feat = pd.merge(test_feat,ridge_zi_test,on='Id')
test_feat = pd.merge(test_feat,mlp_ci_test,on='Id')
test_feat = pd.merge(test_feat,mlp_zi_test,on='Id')
test_feat = pd.merge(test_feat,bayes_ci_test,on='Id')
test_feat = pd.merge(test_feat,keras_mlp_ci_test,on='Id')
test_feat = pd.merge(test_feat,keras_mlp_zi_test,on='Id')
test_feat = pd.merge(test_feat,fasttext_ci_test,on='Id')

test_feat.columns = train_feat.columns
train_feat['label'] = train_data['Score']

feat = ['model1_train_init', 'model1_train_shuffle', 'model1_train_zi',
       'model_ince_train_init', 'duo_train', 'lgb_ci_feat', 'lgb_zi_feat',
       'ridge_ci_feat', 'ridge_zi_feat',
       'bayes_ci_feat',
       'fasttext_ci_feat']


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
        model = lgb.train(params, train1, num_boost_round=10000, valid_sets=[train1, valid1],verbose_eval=100,early_stopping_rounds=30)
        y_sub = model.predict(X_test)
        print(1 /(1 + mean_squared_error(y_test,y_sub)**0.5))
        print(xx_mse_s(y_test, y_sub))
        daset_blend_train[test,0]=y_sub
        daset_blend_test_0[:,i]=model.predict(test_hh)
    daset_blend_test[:, 0] = daset_blend_test_0.mean(1)

    return daset_blend_train,daset_blend_test

a, b = n_folds_train_lgb(5,train_feat['label'],train_feat[feat].values,test_feat[feat].values)

res = pd.DataFrame(test_data['Id'])
res['stack_prob'] = b  
res[['Id','stack_prob']].to_csv('../stack_cache/stack_prob.csv',index=False)