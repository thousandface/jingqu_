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


VECTOR_DIR = './word2vec/word2vec_wx'
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 256


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

def train_w2v():
    data = pd.read_csv('../data/full_tobe_classify_180316.csv',encoding='GBK')
    data['cutted_Dis'].to_csv('../data/lg_all_data.txt',index=False,encoding='utf-8')
    sentences=word2vec.Text8Corpus('../data/lg_all_data.txt')
    model=word2vec.Word2Vec(sentences,min_count=2,size=256)
    model.save('./word2vec/lg_data_model_comment_256dim')

def w2v_mean(train_data,test_data,VECTOR_DIR,EMBEDDING_DIM):
    if((VECTOR_DIR[-4:]=='.txt')|(VECTOR_DIR[-4:]=='.vec')):
        model1 = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_DIR)
    elif(VECTOR_DIR[-4:]=='.bin'):
        model1 = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_DIR, binary=True)
    else:
        model1=Word2Vec.load(VECTOR_DIR)
    train_texts = train_data['cutted_Dis'].values
    test_texts = test_data['cutted_Dis'].values
    x_train = []
    x_test = []
    for train_doc in train_texts:
        words = str(train_doc).split(' ')
        vector = np.zeros(EMBEDDING_DIM)
        word_num = 0
        for word in words:
            if word in model1:
                vector += model1[word]
                word_num += 1
        if word_num > 0:
            vector = vector/word_num
        x_train.append(vector)
    for test_doc in test_texts:
        words = str(test_doc).split(' ')
        vector = np.zeros(EMBEDDING_DIM)
        word_num = 0
        for word in words:
            if word in model1:
                vector += model1[word]
                word_num += 1
        if word_num > 0:
            vector = vector/word_num
        x_test.append(vector)
    return x_train,x_test


def tfidf_process_ci(data,train_data,test_data):

    y = train_data['Score']  
    tf1 = TfidfVectorizer(ngram_range=(1,6),token_pattern='\w+',analyzer='word')
    tf1.fit(data['cutted_Dis'])
    data1=tf1.transform(train_data['cutted_Dis'])
    test1 = tf1.transform(test_data['cutted_Dis'])
    print(data1.shape)  
    tf2 = HashingVectorizer(ngram_range=(1,2),lowercase=False)
    tf2.fit(data['cutted_Dis'])
    data2 = tf2.transform(train_data['cutted_Dis'])
    test2 = tf2.transform(test_data['cutted_Dis'])
    print(data2.shape)   
    train = hstack((data1,data2)).tocsr() 
    test = hstack((test1,test2)).tocsr()
    return train,test,y

def cut_zi(data):
    import re
    r = []
    for i in data['Discuss']:
        r1 = u"a-zA-Z0-9“”‘’[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？?、~@#￥%……&*（）]+"
        tem = re.sub(r1,'',i)
        r.append(tem)
    data['zi'] = r
    p1 = []
    for i in data['zi']:
        i=str(i)
        s=' '
        for j in range(len(i)):
            s = s + ' ' + i[j]
        p1.append(s)
    data['cut_zi'] = p1
    return data
def tfidf_process_zi(data,train_data,test_data):
    data = cut_zi(data)
    train_data = cut_zi(train_data)
    test_data = cut_zi(test_data)

    y = train_data['Score']  
    tf1 = TfidfVectorizer(ngram_range=(1,6),analyzer='char')
    tf1.fit(data['cut_zi'])
    data1=tf1.transform(train_data['cut_zi'])
    test1 = tf1.transform(test_data['cut_zi'])
    print(data1.shape)  
    tf2 = HashingVectorizer(ngram_range=(1,2),lowercase=False)
    tf2.fit(data['cut_zi'])
    data2 = tf2.transform(train_data['cut_zi'])
    test2 = tf2.transform(test_data['cut_zi'])
    print(data2.shape)   
    train = hstack((data1,data2)).tocsr() 
    test = hstack((test1,test2)).tocsr()
    return train,test,y

def train_w2v_zi():
    data = pd.read_csv('../data/full_tobe_classify_180316.csv',encoding='GBK')
    data = cut_zi(data)
    data['cut_zi'].to_csv('../data/lg_all_data_zi.txt',index=False,encoding='utf-8')
    sentences=word2vec.Text8Corpus('../data/lg_all_data_zi.txt')
    model=word2vec.Word2Vec(sentences,min_count=2,size=256)
    model.save('./word2vec/lg_data_model_comment_256dim_zi')

def n_folds_train_lgb(n_folds,y,X,test_hh):
    skf = KFold(X.shape[0], n_folds,random_state=2018,shuffle=True)
    # skf = list(StratifiedKFold(y, n_folds,random_state=2018,shuffle=True))

    params = {
    'learning_rate': 0.05,
    'objective': 'regression_l2',
    'metric': 'mse',
    'num_leaves': 256,
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
        model = lgb.train(params, train1, num_boost_round=10000, valid_sets=[train1, valid1],verbose_eval=10,early_stopping_rounds=50)
        y_sub = model.predict(X_test)
        print(1 /(1 + mean_squared_error(y_test,y_sub)**0.5))
        print(xx_mse_s(y_test, y_sub))
        daset_blend_train[test,0]=y_sub
        daset_blend_test_0[:,i]=model.predict(test_hh)
    daset_blend_test[:, 0] = daset_blend_test_0.mean(1)

    return daset_blend_train,daset_blend_test

def train_lgb(y,X,test_hh):
    params = {
    'learning_rate': 0.05,
    'objective': 'regression_l2',
    'metric': 'mse',
    'num_leaves': 256,
    'bagging_fraction': 0.7,
    'feature_fraction': 0.7,
#     'colsample_bylevel': 0.7,
    'nthread': -1
    }
    train_all = lgb.Dataset(X, label=y)
    model = lgb.train(params,train_all,num_boost_round=450,valid_sets=[train_all],verbose_eval=10)
    result = model.predict(test_hh)
    return result



if __name__ == "__main__":
    # 训练自己文本的词向量
    # train_w2v()
    # 训练自己文本的字向量
    # train_w2v_zi()

    data,train_data,test_data,test_id,same,unkown = get_data()


    # wv2向量平均
    print("微信词向量......")
    VECTOR_DIR_1 = './word2vec/word2vec_wx'
    x_train_1,x_test_1 = w2v_mean(train_data,test_data,VECTOR_DIR_1,256)
    x_train_1 = coo_matrix(np.asarray(x_train_1))
    x_test_1 = coo_matrix(np.asarray(x_test_1))
    print("百度百科向量1......")
    VECTOR_DIR_2 = './word2vec/news12g_bdbk20g_nov90g_dim128.bin'
    x_train_2,x_test_2 = w2v_mean(train_data,test_data,VECTOR_DIR_2,64)
    x_train_2 = coo_matrix(np.asarray(x_train_2))
    x_test_2 = coo_matrix(np.asarray(x_test_2))
    print("百度百科向量2......")
    VECTOR_DIR_3 = './word2vec/news_12g_baidubaike_20g_novel_90g_embedding_64.bin'
    x_train_3,x_test_3 = w2v_mean(train_data,test_data,VECTOR_DIR_3,64)
    x_train_3 = coo_matrix(np.asarray(x_train_3))
    x_test_3 = coo_matrix(np.asarray(x_test_3))
    print("比赛数据训练向量......")
    VECTOR_DIR_4 = './word2vec/model_comment_256dim'
    x_train_4,x_test_4 = w2v_mean(train_data,test_data,VECTOR_DIR_4,256)
    x_train_4 = coo_matrix(np.asarray(x_train_4))
    x_test_4 = coo_matrix(np.asarray(x_test_4))
    print("老哥训练向量......")
    VECTOR_DIR_5 = './word2vec/full_dis_256.model'
    x_train_5,x_test_5 = w2v_mean(train_data,test_data,VECTOR_DIR_5,256)
    x_train_5 = coo_matrix(np.asarray(x_train_5))
    x_test_5 = coo_matrix(np.asarray(x_test_5))

    print("fasttext向量......")
    VECTOR_DIR_6 = './word2vec/model.vec'
    x_train_6,x_test_6 = w2v_mean(train_data,test_data,VECTOR_DIR_6,100)
    x_train_6 = coo_matrix(np.asarray(x_train_6))
    x_test_6 = coo_matrix(np.asarray(x_test_6))

    print("glove向量......")
    VECTOR_DIR_7 = './word2vec/glove_model_200dim.txt'
    x_train_7,x_test_7 = w2v_mean(train_data,test_data,VECTOR_DIR_7,200)
    x_train_7 = coo_matrix(np.asarray(x_train_7))
    x_test_7 = coo_matrix(np.asarray(x_test_7))

    print("TF-IDF词......")
    X,test_hh,y = tfidf_process_ci(data,train_data,test_data)
    X = hstack((X,x_train_1,x_train_2,x_train_3,x_train_4,x_train_5,x_train_6,x_train_7)).tocsr()
    test_hh = hstack((test_hh,x_test_1,x_test_2,x_test_3,x_test_4,x_test_5,x_test_6,x_test_7)).tocsr()
    print('X.shape:',X.shape)
    print('test_hh.shape:',test_hh.shape)

    print("lgb_5fold training......")
    lgb_ci_feat,lgb_ci_prob = n_folds_train_lgb(5,y,X,test_hh)

    train_data['lgb_ci_feat'] = lgb_ci_feat
    test_data['lgb_ci_prob'] = lgb_ci_prob
    train_data[['Id','lgb_ci_feat']].to_csv('../stack_cache/5_fold_lgb_ci_feat_init.csv',index=False)
    test_data[['Id','lgb_ci_prob']].to_csv('../stack_cache/5_fold_lgb_ci_prob_init.csv',index=False)

    print("TF-IDF字......")
    X,test_hh,y = tfidf_process_zi(data,train_data,test_data)
    X = hstack((X,x_train_1,x_train_2,x_train_3,x_train_4,x_train_5,x_train_6,x_train_7)).tocsr()
    test_hh = hstack((test_hh,x_test_1,x_test_2,x_test_3,x_test_4,x_test_5,x_test_6,x_test_7)).tocsr()
    print('X.shape:',X.shape)
    print('test_hh.shape:',test_hh.shape)

    print("lgb_zi_5fold training......")
    lgb_zi_feat,lgb_zi_prob = n_folds_train_lgb(5,y,X,test_hh)

    train_data['lgb_zi_feat'] = lgb_zi_feat
    test_data['lgb_zi_prob'] = lgb_zi_prob
    train_data[['Id','lgb_zi_feat']].to_csv('../stack_cache/5_fold_lgb_zi_feat_init.csv',index=False)
    test_data[['Id','lgb_zi_prob']].to_csv('../stack_cache/5_fold_lgb_zi_prob_init.csv',index=False)


    print("lgb_all training......")
    result = train_lgb(y,X,test_hh)
    res = pd.DataFrame(test_data['Id'])
    res['prob'] = result    
    merge = pd.DataFrame(test_id)
    merge = pd.merge(merge,res,how='left',on='Id')
    merge = pd.merge(merge,same[['Id','Score']],how='left',on='Id')
    merge = pd.merge(merge,unkown[['Id','Score']],how='left',on='Id',suffixes=['_same','_unkown'])
    merge['final_score'] =  merge.prob.where(merge.prob.notnull(),merge.Score_same.where(merge.Score_same.notnull(),merge.Score_unkown))
    merge['prob1'] = merge['final_score'].apply(lambda x:5 if x>=4.7 else x)
    merge['prob1'] = merge['prob1'].apply(lambda x:1 if x<=1 else x)  
    merge[['Id', 'final_score']].to_csv('../result/lg_data_all_mergetfidf_5w2v_notchange_5516.csv',index=False,header=False)
    merge[['Id', 'prob1']].to_csv('../result/lg_data_all_mergetfidf_5w2v.csv',index=False,header=False) # 5516



# train_data len:  220000
# train_data len:  220000
# test_data len:  50000
# test_data len:  50000
# data len:  300000
# data len:  300000
# 微信词向量......
# 百度百科向量1......
# 百度百科向量2......
# 比赛数据训练向量......
# 老哥训练向量......
# fasttext向量......
# glove向量......
# TF-IDF词......
# C:\Users\lizhi\Anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:1089: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
#   if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
# (220000, 17654487)
# (220000, 1048576)
# X.shape: (220000, 18704259)
# test_hh.shape: (50000, 18704259)
# lgb_5fold training......
# [LightGBM] [Info] Total Bins 1283494
# [LightGBM] [Info] Number of data: 176000, number of used features: 56420
# Training until validation scores don't improve for 50 rounds.
# [10]    training's l2: 0.48074  valid_1's l2: 0.499777
# [20]    training's l2: 0.431507 valid_1's l2: 0.465022
# [30]    training's l2: 0.399236 valid_1's l2: 0.446432
# [40]    training's l2: 0.375961 valid_1's l2: 0.434765
# [50]    training's l2: 0.357407 valid_1's l2: 0.427242
# [60]    training's l2: 0.341631 valid_1's l2: 0.421351
# [70]    training's l2: 0.327772 valid_1's l2: 0.416371
# [80]    training's l2: 0.31538  valid_1's l2: 0.412245
# [90]    training's l2: 0.304166 valid_1's l2: 0.408835
# [100]   training's l2: 0.293859 valid_1's l2: 0.405711
# [110]   training's l2: 0.284376 valid_1's l2: 0.402914
# [120]   training's l2: 0.275506 valid_1's l2: 0.400345
# [130]   training's l2: 0.267178 valid_1's l2: 0.398131
# [140]   training's l2: 0.259416 valid_1's l2: 0.396277
# [150]   training's l2: 0.252267 valid_1's l2: 0.394676
# [160]   training's l2: 0.245438 valid_1's l2: 0.393301
# [170]   training's l2: 0.239168 valid_1's l2: 0.391846
# [180]   training's l2: 0.23321  valid_1's l2: 0.390606
# [190]   training's l2: 0.227457 valid_1's l2: 0.389547
# [200]   training's l2: 0.222259 valid_1's l2: 0.388572
# [210]   training's l2: 0.217412 valid_1's l2: 0.38747
# [220]   training's l2: 0.21271  valid_1's l2: 0.386682
# [230]   training's l2: 0.207936 valid_1's l2: 0.385738
# [240]   training's l2: 0.203591 valid_1's l2: 0.384794
# [250]   training's l2: 0.199154 valid_1's l2: 0.383901
# [260]   training's l2: 0.194727 valid_1's l2: 0.383197
# [270]   training's l2: 0.190874 valid_1's l2: 0.38256
# [280]   training's l2: 0.187223 valid_1's l2: 0.381816
# [290]   training's l2: 0.183558 valid_1's l2: 0.38105
# [300]   training's l2: 0.180145 valid_1's l2: 0.380415
# [310]   training's l2: 0.176704 valid_1's l2: 0.379917
# [320]   training's l2: 0.173285 valid_1's l2: 0.379471
# [330]   training's l2: 0.170172 valid_1's l2: 0.37888
# [340]   training's l2: 0.167089 valid_1's l2: 0.378199
# [350]   training's l2: 0.164053 valid_1's l2: 0.377714
# [360]   training's l2: 0.161058 valid_1's l2: 0.377252
# [370]   training's l2: 0.158333 valid_1's l2: 0.376697
# [380]   training's l2: 0.155783 valid_1's l2: 0.376183
# [390]   training's l2: 0.153134 valid_1's l2: 0.375813
# [400]   training's l2: 0.150608 valid_1's l2: 0.375472
# [410]   training's l2: 0.147988 valid_1's l2: 0.375005
# [420]   training's l2: 0.145668 valid_1's l2: 0.374614
# [430]   training's l2: 0.143211 valid_1's l2: 0.374248
# [440]   training's l2: 0.140731 valid_1's l2: 0.373822
# [450]   training's l2: 0.138511 valid_1's l2: 0.373479
# [460]   training's l2: 0.136284 valid_1's l2: 0.373215
# [470]   training's l2: 0.134092 valid_1's l2: 0.372935
# [480]   training's l2: 0.132118 valid_1's l2: 0.372626
# [490]   training's l2: 0.13012  valid_1's l2: 0.372244
# [500]   training's l2: 0.128266 valid_1's l2: 0.371904
# [510]   training's l2: 0.126251 valid_1's l2: 0.371555
# [520]   training's l2: 0.124418 valid_1's l2: 0.371252
# [530]   training's l2: 0.122521 valid_1's l2: 0.370902
# [540]   training's l2: 0.120899 valid_1's l2: 0.37068
# [550]   training's l2: 0.119093 valid_1's l2: 0.37042
# [560]   training's l2: 0.117508 valid_1's l2: 0.370232
# [570]   training's l2: 0.115923 valid_1's l2: 0.370069
# [580]   training's l2: 0.114342 valid_1's l2: 0.369867
# [590]   training's l2: 0.112708 valid_1's l2: 0.369635
# [600]   training's l2: 0.111094 valid_1's l2: 0.369414
# [610]   training's l2: 0.109524 valid_1's l2: 0.369171
# [620]   training's l2: 0.108017 valid_1's l2: 0.368967
# [630]   training's l2: 0.106567 valid_1's l2: 0.368842
# [640]   training's l2: 0.105122 valid_1's l2: 0.368601
# [650]   training's l2: 0.103845 valid_1's l2: 0.368376
# [660]   training's l2: 0.102541 valid_1's l2: 0.368196
# [670]   training's l2: 0.101336 valid_1's l2: 0.367987
# [680]   training's l2: 0.100089 valid_1's l2: 0.367907
# [690]   training's l2: 0.0988128    valid_1's l2: 0.367738
# [700]   training's l2: 0.0975918    valid_1's l2: 0.367653
# [710]   training's l2: 0.0964619    valid_1's l2: 0.367543
# [720]   training's l2: 0.095132 valid_1's l2: 0.367404
# [730]   training's l2: 0.0940626    valid_1's l2: 0.367225
# [740]   training's l2: 0.0930169    valid_1's l2: 0.367122
# [750]   training's l2: 0.0918997    valid_1's l2: 0.367038
# [760]   training's l2: 0.0907548    valid_1's l2: 0.366864
# [770]   training's l2: 0.0896839    valid_1's l2: 0.366769
# [780]   training's l2: 0.0886686    valid_1's l2: 0.366638
# [790]   training's l2: 0.0876273    valid_1's l2: 0.366537
# [800]   training's l2: 0.0866702    valid_1's l2: 0.366381
# [810]   training's l2: 0.0856678    valid_1's l2: 0.366202
# [820]   training's l2: 0.0847098    valid_1's l2: 0.366091
# [830]   training's l2: 0.0838072    valid_1's l2: 0.365975
# [840]   training's l2: 0.0828832    valid_1's l2: 0.365856
# [850]   training's l2: 0.0820619    valid_1's l2: 0.365766
# [860]   training's l2: 0.0811942    valid_1's l2: 0.365707
# [870]   training's l2: 0.0803609    valid_1's l2: 0.365614
# [880]   training's l2: 0.0795113    valid_1's l2: 0.365494
# [890]   training's l2: 0.0786714    valid_1's l2: 0.365404
# [900]   training's l2: 0.0778362    valid_1's l2: 0.365264
# [910]   training's l2: 0.0770775    valid_1's l2: 0.365178
# [920]   training's l2: 0.0763448    valid_1's l2: 0.365086
# [930]   training's l2: 0.0756376    valid_1's l2: 0.365014
# [940]   training's l2: 0.074802 valid_1's l2: 0.364924
# [950]   training's l2: 0.0739709    valid_1's l2: 0.364838
# [960]   training's l2: 0.0732923    valid_1's l2: 0.36478
# [970]   training's l2: 0.0726161    valid_1's l2: 0.364702
# [980]   training's l2: 0.0719781    valid_1's l2: 0.364616
# [990]   training's l2: 0.0713662    valid_1's l2: 0.364496
# [1000]  training's l2: 0.0707319    valid_1's l2: 0.364472
# [1010]  training's l2: 0.0701499    valid_1's l2: 0.36442
# [1020]  training's l2: 0.0694206    valid_1's l2: 0.364353
# [1030]  training's l2: 0.0688983    valid_1's l2: 0.364321
# [1040]  training's l2: 0.0683395    valid_1's l2: 0.364278
# [1050]  training's l2: 0.0677659    valid_1's l2: 0.364259
# [1060]  training's l2: 0.0672139    valid_1's l2: 0.364227
# [1070]  training's l2: 0.0666592    valid_1's l2: 0.364145
# [1080]  training's l2: 0.0661247    valid_1's l2: 0.364125
# [1090]  training's l2: 0.065578 valid_1's l2: 0.364076
# [1100]  training's l2: 0.0650385    valid_1's l2: 0.363999
# [1110]  training's l2: 0.0645432    valid_1's l2: 0.363948
# [1120]  training's l2: 0.0640222    valid_1's l2: 0.363899
# [1130]  training's l2: 0.0635343    valid_1's l2: 0.363842
# [1140]  training's l2: 0.0630929    valid_1's l2: 0.363818
# [1150]  training's l2: 0.0625942    valid_1's l2: 0.363741
# [1160]  training's l2: 0.0621434    valid_1's l2: 0.363703
# [1170]  training's l2: 0.0616556    valid_1's l2: 0.363695
# [1180]  training's l2: 0.0612547    valid_1's l2: 0.363658
# [1190]  training's l2: 0.0607701    valid_1's l2: 0.363622
# [1200]  training's l2: 0.0603365    valid_1's l2: 0.363577
# [1210]  training's l2: 0.0599464    valid_1's l2: 0.363575
# [1220]  training's l2: 0.0595124    valid_1's l2: 0.363575
# [1230]  training's l2: 0.0590817    valid_1's l2: 0.363561
# [1240]  training's l2: 0.0586715    valid_1's l2: 0.36357
# [1250]  training's l2: 0.0582565    valid_1's l2: 0.363536
# [1260]  training's l2: 0.0578599    valid_1's l2: 0.363525
# [1270]  training's l2: 0.0574629    valid_1's l2: 0.363558
# [1280]  training's l2: 0.0570952    valid_1's l2: 0.363523
# [1290]  training's l2: 0.0567585    valid_1's l2: 0.363506
# [1300]  training's l2: 0.0563729    valid_1's l2: 0.363452
# [1310]  training's l2: 0.0560298    valid_1's l2: 0.363422
# [1320]  training's l2: 0.055641 valid_1's l2: 0.363374
# [1330]  training's l2: 0.0553084    valid_1's l2: 0.363346
# [1340]  training's l2: 0.05497  valid_1's l2: 0.363376
# [1350]  training's l2: 0.0546257    valid_1's l2: 0.363375
# [1360]  training's l2: 0.0543523    valid_1's l2: 0.363359
# [1370]  training's l2: 0.0540457    valid_1's l2: 0.36334
# [1380]  training's l2: 0.053739 valid_1's l2: 0.363332
# [1390]  training's l2: 0.0534188    valid_1's l2: 0.363304
# [1400]  training's l2: 0.0530961    valid_1's l2: 0.363293
# [1410]  training's l2: 0.0528176    valid_1's l2: 0.363267
# [1420]  training's l2: 0.0525454    valid_1's l2: 0.363204
# [1430]  training's l2: 0.0522561    valid_1's l2: 0.363204
# [1440]  training's l2: 0.051951 valid_1's l2: 0.36316
# [1450]  training's l2: 0.0516715    valid_1's l2: 0.363151
# [1460]  training's l2: 0.0514239    valid_1's l2: 0.363159
# [1470]  training's l2: 0.0511519    valid_1's l2: 0.363135
# [1480]  training's l2: 0.0509034    valid_1's l2: 0.363131
# [1490]  training's l2: 0.0506346    valid_1's l2: 0.363101
# [1500]  training's l2: 0.0504017    valid_1's l2: 0.363104
# [1510]  training's l2: 0.0501523    valid_1's l2: 0.363123
# [1520]  training's l2: 0.0499384    valid_1's l2: 0.363133
# [1530]  training's l2: 0.0497109    valid_1's l2: 0.363091
# [1540]  training's l2: 0.0494788    valid_1's l2: 0.363042
# [1550]  training's l2: 0.0492537    valid_1's l2: 0.363054
# [1560]  training's l2: 0.0490296    valid_1's l2: 0.363045
# [1570]  training's l2: 0.048816 valid_1's l2: 0.363058
# [1580]  training's l2: 0.0486168    valid_1's l2: 0.363057
# Early stopping, best iteration is:
# [1539]  training's l2: 0.0495054    valid_1's l2: 0.363038
# [LightGBM] [Info] Finished loading 1539 models
# 0.6240145926808336
# 0.5301607754399793
# [LightGBM] [Info] Total Bins 1280109
# [LightGBM] [Info] Number of data: 176000, number of used features: 55960
# Training until validation scores don't improve for 50 rounds.
# [10]    training's l2: 0.480177 valid_1's l2: 0.503412
# [20]    training's l2: 0.431246 valid_1's l2: 0.468231
# [30]    training's l2: 0.39927  valid_1's l2: 0.448383
# [40]    training's l2: 0.376069 valid_1's l2: 0.436481
# [50]    training's l2: 0.357354 valid_1's l2: 0.428511
# [60]    training's l2: 0.341687 valid_1's l2: 0.42237
# [70]    training's l2: 0.327788 valid_1's l2: 0.41731
# [80]    training's l2: 0.315331 valid_1's l2: 0.41297
# [90]    training's l2: 0.304068 valid_1's l2: 0.409621
# [100]   training's l2: 0.293839 valid_1's l2: 0.406659
# [110]   training's l2: 0.2843   valid_1's l2: 0.403921
# [120]   training's l2: 0.275434 valid_1's l2: 0.401696
# [130]   training's l2: 0.267139 valid_1's l2: 0.399938
# [140]   training's l2: 0.259386 valid_1's l2: 0.397963
# [150]   training's l2: 0.252259 valid_1's l2: 0.396258
# [160]   training's l2: 0.245636 valid_1's l2: 0.394946
# [170]   training's l2: 0.239419 valid_1's l2: 0.393581
# [180]   training's l2: 0.233637 valid_1's l2: 0.392479
# [190]   training's l2: 0.228332 valid_1's l2: 0.39149
# [200]   training's l2: 0.22317  valid_1's l2: 0.390593
# [210]   training's l2: 0.218235 valid_1's l2: 0.389704
# [220]   training's l2: 0.21326  valid_1's l2: 0.388897
# [230]   training's l2: 0.208786 valid_1's l2: 0.388023
# [240]   training's l2: 0.204444 valid_1's l2: 0.387386
# [250]   training's l2: 0.200276 valid_1's l2: 0.386513
# [260]   training's l2: 0.196364 valid_1's l2: 0.385862
# [270]   training's l2: 0.192292 valid_1's l2: 0.385199
# [280]   training's l2: 0.188545 valid_1's l2: 0.384541
# [290]   training's l2: 0.184632 valid_1's l2: 0.383845
# [300]   training's l2: 0.181259 valid_1's l2: 0.383262
# [310]   training's l2: 0.177859 valid_1's l2: 0.382693
# [320]   training's l2: 0.174262 valid_1's l2: 0.382074
# [330]   training's l2: 0.17106  valid_1's l2: 0.381573
# [340]   training's l2: 0.167811 valid_1's l2: 0.380961
# [350]   training's l2: 0.165057 valid_1's l2: 0.380407
# [360]   training's l2: 0.162096 valid_1's l2: 0.379921
# [370]   training's l2: 0.159276 valid_1's l2: 0.379502
# [380]   training's l2: 0.156642 valid_1's l2: 0.379023
# [390]   training's l2: 0.153722 valid_1's l2: 0.378656
# [400]   training's l2: 0.15111  valid_1's l2: 0.378247
# [410]   training's l2: 0.148505 valid_1's l2: 0.3778
# [420]   training's l2: 0.145942 valid_1's l2: 0.377432
# [430]   training's l2: 0.143594 valid_1's l2: 0.377104
# [440]   training's l2: 0.141249 valid_1's l2: 0.376701
# [450]   training's l2: 0.138951 valid_1's l2: 0.376299
# [460]   training's l2: 0.136867 valid_1's l2: 0.375975
# [470]   training's l2: 0.13467  valid_1's l2: 0.375698
# [480]   training's l2: 0.132659 valid_1's l2: 0.375566
# [490]   training's l2: 0.130723 valid_1's l2: 0.375252
# [500]   training's l2: 0.128736 valid_1's l2: 0.375001
# [510]   training's l2: 0.126889 valid_1's l2: 0.374724
# [520]   training's l2: 0.12496  valid_1's l2: 0.374435
# [530]   training's l2: 0.123196 valid_1's l2: 0.374135
# [540]   training's l2: 0.121591 valid_1's l2: 0.373911
# [550]   training's l2: 0.119988 valid_1's l2: 0.373658
# [560]   training's l2: 0.118178 valid_1's l2: 0.373473
# [570]   training's l2: 0.116384 valid_1's l2: 0.373275
# [580]   training's l2: 0.114838 valid_1's l2: 0.373072
# [590]   training's l2: 0.11318  valid_1's l2: 0.372835
# [600]   training's l2: 0.111611 valid_1's l2: 0.37266
# [610]   training's l2: 0.110134 valid_1's l2: 0.372478
# [620]   training's l2: 0.108591 valid_1's l2: 0.37224
# [630]   training's l2: 0.107075 valid_1's l2: 0.372015
# [640]   training's l2: 0.105607 valid_1's l2: 0.3718
# [650]   training's l2: 0.104241 valid_1's l2: 0.37159
# [660]   training's l2: 0.102972 valid_1's l2: 0.371433
# [670]   training's l2: 0.101658 valid_1's l2: 0.371267
# [680]   training's l2: 0.100379 valid_1's l2: 0.371124
# [690]   training's l2: 0.0990207    valid_1's l2: 0.370922
# [700]   training's l2: 0.0977771    valid_1's l2: 0.370738
# [710]   training's l2: 0.0965942    valid_1's l2: 0.370506
# [720]   training's l2: 0.0953586    valid_1's l2: 0.370432
# [730]   training's l2: 0.0940839    valid_1's l2: 0.370212
# [740]   training's l2: 0.0929906    valid_1's l2: 0.370077
# [750]   training's l2: 0.0917918    valid_1's l2: 0.36987
# [760]   training's l2: 0.0906912    valid_1's l2: 0.369646
# [770]   training's l2: 0.0896543    valid_1's l2: 0.369396
# [780]   training's l2: 0.0886189    valid_1's l2: 0.369262
# [790]   training's l2: 0.087672 valid_1's l2: 0.369113
# [800]   training's l2: 0.0866594    valid_1's l2: 0.369051
# [810]   training's l2: 0.0856628    valid_1's l2: 0.368957
# [820]   training's l2: 0.0847387    valid_1's l2: 0.368857
# [830]   training's l2: 0.0838255    valid_1's l2: 0.368756
# [840]   training's l2: 0.0828709    valid_1's l2: 0.368696
# [850]   training's l2: 0.0819499    valid_1's l2: 0.368549
# [860]   training's l2: 0.0810159    valid_1's l2: 0.368422
# [870]   training's l2: 0.0801392    valid_1's l2: 0.368279
# [880]   training's l2: 0.0793242    valid_1's l2: 0.368208
# [890]   training's l2: 0.0785263    valid_1's l2: 0.368095
# [900]   training's l2: 0.0777903    valid_1's l2: 0.368031
# [910]   training's l2: 0.0770158    valid_1's l2: 0.367911
# [920]   training's l2: 0.0762857    valid_1's l2: 0.367805
# [930]   training's l2: 0.0755279    valid_1's l2: 0.367724
# [940]   training's l2: 0.0748275    valid_1's l2: 0.367633
# [950]   training's l2: 0.0741154    valid_1's l2: 0.367595
# [960]   training's l2: 0.073388 valid_1's l2: 0.367524
# [970]   training's l2: 0.07276  valid_1's l2: 0.367434
# [980]   training's l2: 0.0720623    valid_1's l2: 0.367316
# [990]   training's l2: 0.0713598    valid_1's l2: 0.367287
# [1000]  training's l2: 0.0706962    valid_1's l2: 0.367251
# [1010]  training's l2: 0.0700588    valid_1's l2: 0.367142
# [1020]  training's l2: 0.0694754    valid_1's l2: 0.367085
# [1030]  training's l2: 0.0688476    valid_1's l2: 0.366995
# [1040]  training's l2: 0.0683037    valid_1's l2: 0.366894
# [1050]  training's l2: 0.0677823    valid_1's l2: 0.366834
# [1060]  training's l2: 0.067236 valid_1's l2: 0.366704
# [1070]  training's l2: 0.0666415    valid_1's l2: 0.366628
# [1080]  training's l2: 0.0660533    valid_1's l2: 0.366556
# [1090]  training's l2: 0.06547  valid_1's l2: 0.366482
# [1100]  training's l2: 0.0649273    valid_1's l2: 0.366434
# [1110]  training's l2: 0.06438  valid_1's l2: 0.366417
# [1120]  training's l2: 0.0638386    valid_1's l2: 0.366376
# [1130]  training's l2: 0.0633585    valid_1's l2: 0.366318
# [1140]  training's l2: 0.0628624    valid_1's l2: 0.366313
# [1150]  training's l2: 0.0623749    valid_1's l2: 0.366257
# [1160]  training's l2: 0.0618401    valid_1's l2: 0.366236
# [1170]  training's l2: 0.0613542    valid_1's l2: 0.36624
# [1180]  training's l2: 0.0608267    valid_1's l2: 0.366198
# [1190]  training's l2: 0.0603992    valid_1's l2: 0.366154
# [1200]  training's l2: 0.0600077    valid_1's l2: 0.366119
# [1210]  training's l2: 0.0596004    valid_1's l2: 0.366059
# [1220]  training's l2: 0.0591257    valid_1's l2: 0.366009
# [1230]  training's l2: 0.0587193    valid_1's l2: 0.365951
# [1240]  training's l2: 0.0583155    valid_1's l2: 0.365939
# [1250]  training's l2: 0.0579378    valid_1's l2: 0.365925
# [1260]  training's l2: 0.0575472    valid_1's l2: 0.365902
# [1270]  training's l2: 0.0571453    valid_1's l2: 0.365857
# [1280]  training's l2: 0.0567878    valid_1's l2: 0.365838
# [1290]  training's l2: 0.0564204    valid_1's l2: 0.365843
# [1300]  training's l2: 0.0560688    valid_1's l2: 0.365808
# [1310]  training's l2: 0.0557297    valid_1's l2: 0.365746
# [1320]  training's l2: 0.0553638    valid_1's l2: 0.365724
# [1330]  training's l2: 0.0550387    valid_1's l2: 0.365685
# [1340]  training's l2: 0.0546988    valid_1's l2: 0.36565
# [1350]  training's l2: 0.0543399    valid_1's l2: 0.36562
# [1360]  training's l2: 0.054022 valid_1's l2: 0.365628
# [1370]  training's l2: 0.0537369    valid_1's l2: 0.365601
# [1380]  training's l2: 0.0534206    valid_1's l2: 0.365577
# [1390]  training's l2: 0.0531359    valid_1's l2: 0.365509
# [1400]  training's l2: 0.0528282    valid_1's l2: 0.36549
# [1410]  training's l2: 0.0525508    valid_1's l2: 0.365454
# [1420]  training's l2: 0.0522277    valid_1's l2: 0.365412
# [1430]  training's l2: 0.0519661    valid_1's l2: 0.365392
# [1440]  training's l2: 0.0517165    valid_1's l2: 0.365376
# [1450]  training's l2: 0.0514354    valid_1's l2: 0.365373
# [1460]  training's l2: 0.0511969    valid_1's l2: 0.365392
# [1470]  training's l2: 0.0509609    valid_1's l2: 0.365385
# [1480]  training's l2: 0.050707 valid_1's l2: 0.365352
# [1490]  training's l2: 0.0504818    valid_1's l2: 0.365341
# [1500]  training's l2: 0.0502242    valid_1's l2: 0.365313
# [1510]  training's l2: 0.0499614    valid_1's l2: 0.365295
# [1520]  training's l2: 0.0497068    valid_1's l2: 0.365264
# [1530]  training's l2: 0.0494916    valid_1's l2: 0.365243
# [1540]  training's l2: 0.0492727    valid_1's l2: 0.365224
# [1550]  training's l2: 0.0490809    valid_1's l2: 0.365209
# [1560]  training's l2: 0.0488467    valid_1's l2: 0.365189
# [1570]  training's l2: 0.0486417    valid_1's l2: 0.365175
# [1580]  training's l2: 0.0484398    valid_1's l2: 0.36516
# [1590]  training's l2: 0.0482464    valid_1's l2: 0.365154
# [1600]  training's l2: 0.0480441    valid_1's l2: 0.365117
# [1610]  training's l2: 0.0478615    valid_1's l2: 0.365125
# [1620]  training's l2: 0.0476696    valid_1's l2: 0.365112
# [1630]  training's l2: 0.0474667    valid_1's l2: 0.365089
# [1640]  training's l2: 0.0472772    valid_1's l2: 0.365099
# [1650]  training's l2: 0.0470899    valid_1's l2: 0.365065
# [1660]  training's l2: 0.0469214    valid_1's l2: 0.365065
# [1670]  training's l2: 0.0467376    valid_1's l2: 0.365056
# [1680]  training's l2: 0.0465646    valid_1's l2: 0.365028
# [1690]  training's l2: 0.0463739    valid_1's l2: 0.365033
# [1700]  training's l2: 0.0461939    valid_1's l2: 0.365005
# [1710]  training's l2: 0.0460222    valid_1's l2: 0.364981
# [1720]  training's l2: 0.0458698    valid_1's l2: 0.364988
# [1730]  training's l2: 0.0457105    valid_1's l2: 0.364992
# [1740]  training's l2: 0.0455657    valid_1's l2: 0.364971
# [1750]  training's l2: 0.0454037    valid_1's l2: 0.364972
# [1760]  training's l2: 0.0452444    valid_1's l2: 0.364953
# [1770]  training's l2: 0.0450907    valid_1's l2: 0.364932
# [1780]  training's l2: 0.0449409    valid_1's l2: 0.364921
# [1790]  training's l2: 0.0448016    valid_1's l2: 0.364904
# [1800]  training's l2: 0.0446608    valid_1's l2: 0.364916
# [1810]  training's l2: 0.0445255    valid_1's l2: 0.364912
# [1820]  training's l2: 0.0443803    valid_1's l2: 0.364904
# [1830]  training's l2: 0.0442542    valid_1's l2: 0.364926
# [1840]  training's l2: 0.0441163    valid_1's l2: 0.364922
# [1850]  training's l2: 0.0439914    valid_1's l2: 0.364916
# [1860]  training's l2: 0.0438665    valid_1's l2: 0.364909
# [1870]  training's l2: 0.0437468    valid_1's l2: 0.364922
# Early stopping, best iteration is:
# [1820]  training's l2: 0.0443803    valid_1's l2: 0.364904
# [LightGBM] [Info] Finished loading 1820 models
# 0.6234131112784351
# 0.5300995211974439
# [LightGBM] [Info] Total Bins 1277038
# [LightGBM] [Info] Number of data: 176000, number of used features: 55452
# Training until validation scores don't improve for 50 rounds.
# [10]    training's l2: 0.481042 valid_1's l2: 0.497578
# [20]    training's l2: 0.43182  valid_1's l2: 0.463965
# [30]    training's l2: 0.39953  valid_1's l2: 0.445304
# [40]    training's l2: 0.37634  valid_1's l2: 0.434194
# [50]    training's l2: 0.357764 valid_1's l2: 0.426524
# [60]    training's l2: 0.341947 valid_1's l2: 0.420491
# [70]    training's l2: 0.327902 valid_1's l2: 0.415426
# [80]    training's l2: 0.315424 valid_1's l2: 0.411134
# [90]    training's l2: 0.304175 valid_1's l2: 0.407146
# [100]   training's l2: 0.293894 valid_1's l2: 0.404098
# [110]   training's l2: 0.284416 valid_1's l2: 0.40134
# [120]   training's l2: 0.275497 valid_1's l2: 0.398798
# [130]   training's l2: 0.267341 valid_1's l2: 0.396626
# [140]   training's l2: 0.259749 valid_1's l2: 0.394615
# [150]   training's l2: 0.252645 valid_1's l2: 0.393074
# [160]   training's l2: 0.245965 valid_1's l2: 0.391509
# [170]   training's l2: 0.239816 valid_1's l2: 0.389902
# [180]   training's l2: 0.234281 valid_1's l2: 0.388743
# [190]   training's l2: 0.228833 valid_1's l2: 0.387538
# [200]   training's l2: 0.223491 valid_1's l2: 0.386381
# [210]   training's l2: 0.218615 valid_1's l2: 0.385364
# [220]   training's l2: 0.213745 valid_1's l2: 0.38436
# [230]   training's l2: 0.20922  valid_1's l2: 0.383455
# [240]   training's l2: 0.204991 valid_1's l2: 0.382434
# [250]   training's l2: 0.200668 valid_1's l2: 0.381474
# [260]   training's l2: 0.196359 valid_1's l2: 0.380684
# [270]   training's l2: 0.192369 valid_1's l2: 0.379938
# [280]   training's l2: 0.188644 valid_1's l2: 0.37925
# [290]   training's l2: 0.185029 valid_1's l2: 0.378546
# [300]   training's l2: 0.181491 valid_1's l2: 0.377818
# [310]   training's l2: 0.177849 valid_1's l2: 0.377221
# [320]   training's l2: 0.174506 valid_1's l2: 0.37663
# [330]   training's l2: 0.171467 valid_1's l2: 0.376048
# [340]   training's l2: 0.168342 valid_1's l2: 0.375496
# [350]   training's l2: 0.165427 valid_1's l2: 0.375078
# [360]   training's l2: 0.16238  valid_1's l2: 0.374651
# [370]   training's l2: 0.159509 valid_1's l2: 0.374107
# [380]   training's l2: 0.156754 valid_1's l2: 0.373584
# [390]   training's l2: 0.154067 valid_1's l2: 0.373213
# [400]   training's l2: 0.151544 valid_1's l2: 0.372802
# [410]   training's l2: 0.1489   valid_1's l2: 0.372427
# [420]   training's l2: 0.146395 valid_1's l2: 0.37195
# [430]   training's l2: 0.143946 valid_1's l2: 0.371576
# [440]   training's l2: 0.14154  valid_1's l2: 0.371136
# [450]   training's l2: 0.139088 valid_1's l2: 0.370831
# [460]   training's l2: 0.136817 valid_1's l2: 0.370361
# [470]   training's l2: 0.134496 valid_1's l2: 0.370048
# [480]   training's l2: 0.132386 valid_1's l2: 0.369711
# [490]   training's l2: 0.130288 valid_1's l2: 0.369426
# [500]   training's l2: 0.128315 valid_1's l2: 0.369113
# [510]   training's l2: 0.126494 valid_1's l2: 0.368831
# [520]   training's l2: 0.124582 valid_1's l2: 0.368601
# [530]   training's l2: 0.122571 valid_1's l2: 0.368268
# [540]   training's l2: 0.120942 valid_1's l2: 0.367959
# [550]   training's l2: 0.119077 valid_1's l2: 0.367721
# [560]   training's l2: 0.117068 valid_1's l2: 0.367427
# [570]   training's l2: 0.115431 valid_1's l2: 0.367205
# [580]   training's l2: 0.113686 valid_1's l2: 0.366961
# [590]   training's l2: 0.112081 valid_1's l2: 0.366641
# [600]   training's l2: 0.110544 valid_1's l2: 0.366293
# [610]   training's l2: 0.109049 valid_1's l2: 0.366067
# [620]   training's l2: 0.107619 valid_1's l2: 0.36578
# [630]   training's l2: 0.106242 valid_1's l2: 0.365541
# [640]   training's l2: 0.104718 valid_1's l2: 0.365254
# [650]   training's l2: 0.103215 valid_1's l2: 0.365013
# [660]   training's l2: 0.1018   valid_1's l2: 0.364817
# [670]   training's l2: 0.10036  valid_1's l2: 0.364596
# [680]   training's l2: 0.0989467    valid_1's l2: 0.364429
# [690]   training's l2: 0.0977124    valid_1's l2: 0.364243
# [700]   training's l2: 0.0963497    valid_1's l2: 0.363975
# [710]   training's l2: 0.0951999    valid_1's l2: 0.363754
# [720]   training's l2: 0.0940406    valid_1's l2: 0.363576
# [730]   training's l2: 0.0928107    valid_1's l2: 0.363411
# [740]   training's l2: 0.0915614    valid_1's l2: 0.363175
# [750]   training's l2: 0.0904254    valid_1's l2: 0.363059
# [760]   training's l2: 0.0893566    valid_1's l2: 0.362898
# [770]   training's l2: 0.0883514    valid_1's l2: 0.362706
# [780]   training's l2: 0.0873331    valid_1's l2: 0.362528
# [790]   training's l2: 0.0862074    valid_1's l2: 0.362327
# [800]   training's l2: 0.0851567    valid_1's l2: 0.36217
# [810]   training's l2: 0.0842592    valid_1's l2: 0.362082
# [820]   training's l2: 0.0832741    valid_1's l2: 0.361972
# [830]   training's l2: 0.0822791    valid_1's l2: 0.361848
# [840]   training's l2: 0.0813102    valid_1's l2: 0.361731
# [850]   training's l2: 0.0804124    valid_1's l2: 0.361663
# [860]   training's l2: 0.0795119    valid_1's l2: 0.361525
# [870]   training's l2: 0.0786548    valid_1's l2: 0.361447
# [880]   training's l2: 0.077748 valid_1's l2: 0.361293
# [890]   training's l2: 0.0769132    valid_1's l2: 0.361249
# [900]   training's l2: 0.0761604    valid_1's l2: 0.361162
# [910]   training's l2: 0.075371 valid_1's l2: 0.361028
# [920]   training's l2: 0.0745311    valid_1's l2: 0.360886
# [930]   training's l2: 0.0738019    valid_1's l2: 0.360802
# [940]   training's l2: 0.073096 valid_1's l2: 0.360731
# [950]   training's l2: 0.0723924    valid_1's l2: 0.360619
# [960]   training's l2: 0.0717598    valid_1's l2: 0.360545
# [970]   training's l2: 0.0710525    valid_1's l2: 0.360502
# [980]   training's l2: 0.070408 valid_1's l2: 0.36044
# [990]   training's l2: 0.0697905    valid_1's l2: 0.360395
# [1000]  training's l2: 0.0691253    valid_1's l2: 0.360305
# [1010]  training's l2: 0.0685556    valid_1's l2: 0.360251
# [1020]  training's l2: 0.0679078    valid_1's l2: 0.360151
# [1030]  training's l2: 0.067321 valid_1's l2: 0.360112
# [1040]  training's l2: 0.0667383    valid_1's l2: 0.360005
# [1050]  training's l2: 0.066192 valid_1's l2: 0.359925
# [1060]  training's l2: 0.0656404    valid_1's l2: 0.35987
# [1070]  training's l2: 0.0650593    valid_1's l2: 0.359837
# [1080]  training's l2: 0.0645278    valid_1's l2: 0.35977
# [1090]  training's l2: 0.0639857    valid_1's l2: 0.359703
# [1100]  training's l2: 0.0634927    valid_1's l2: 0.359652
# [1110]  training's l2: 0.0630133    valid_1's l2: 0.359638
# [1120]  training's l2: 0.0625134    valid_1's l2: 0.359591
# [1130]  training's l2: 0.0620346    valid_1's l2: 0.359514
# [1140]  training's l2: 0.0615655    valid_1's l2: 0.359487
# [1150]  training's l2: 0.0610815    valid_1's l2: 0.359455
# [1160]  training's l2: 0.0606587    valid_1's l2: 0.359434
# [1170]  training's l2: 0.0602466    valid_1's l2: 0.359421
# [1180]  training's l2: 0.059802 valid_1's l2: 0.359378
# [1190]  training's l2: 0.0593764    valid_1's l2: 0.359341
# [1200]  training's l2: 0.0589773    valid_1's l2: 0.359307
# [1210]  training's l2: 0.0585226    valid_1's l2: 0.359287
# [1220]  training's l2: 0.058084 valid_1's l2: 0.359229
# [1230]  training's l2: 0.0576571    valid_1's l2: 0.359189
# [1240]  training's l2: 0.0573028    valid_1's l2: 0.359154
# [1250]  training's l2: 0.0569283    valid_1's l2: 0.359105
# [1260]  training's l2: 0.0565646    valid_1's l2: 0.359062
# [1270]  training's l2: 0.0561975    valid_1's l2: 0.359013
# [1280]  training's l2: 0.0558307    valid_1's l2: 0.358965
# [1290]  training's l2: 0.055487 valid_1's l2: 0.358958
# [1300]  training's l2: 0.0551362    valid_1's l2: 0.358931
# [1310]  training's l2: 0.0547572    valid_1's l2: 0.358865
# [1320]  training's l2: 0.0544252    valid_1's l2: 0.358812
# [1330]  training's l2: 0.0541231    valid_1's l2: 0.358804
# [1340]  training's l2: 0.0538103    valid_1's l2: 0.358784
# [1350]  training's l2: 0.0535176    valid_1's l2: 0.358759
# [1360]  training's l2: 0.0532098    valid_1's l2: 0.358725
# [1370]  training's l2: 0.0528892    valid_1's l2: 0.358706
# [1380]  training's l2: 0.0526316    valid_1's l2: 0.35871
# [1390]  training's l2: 0.0523396    valid_1's l2: 0.358684
# [1400]  training's l2: 0.0520629    valid_1's l2: 0.358641
# [1410]  training's l2: 0.0517959    valid_1's l2: 0.358609
# [1420]  training's l2: 0.0515205    valid_1's l2: 0.358578
# [1430]  training's l2: 0.0512448    valid_1's l2: 0.358579
# [1440]  training's l2: 0.0509904    valid_1's l2: 0.358531
# [1450]  training's l2: 0.0507242    valid_1's l2: 0.35852
# [1460]  training's l2: 0.0504792    valid_1's l2: 0.358518
# [1470]  training's l2: 0.0502101    valid_1's l2: 0.358524
# [1480]  training's l2: 0.0499545    valid_1's l2: 0.358504
# [1490]  training's l2: 0.0497044    valid_1's l2: 0.358478
# [1500]  training's l2: 0.0494814    valid_1's l2: 0.358452
# [1510]  training's l2: 0.0492632    valid_1's l2: 0.358434
# [1520]  training's l2: 0.0490369    valid_1's l2: 0.358432
# [1530]  training's l2: 0.0487979    valid_1's l2: 0.358387
# [1540]  training's l2: 0.048573 valid_1's l2: 0.358386
# [1550]  training's l2: 0.048348 valid_1's l2: 0.358366
# [1560]  training's l2: 0.048143 valid_1's l2: 0.358379
# [1570]  training's l2: 0.0479464    valid_1's l2: 0.358381
# [1580]  training's l2: 0.0477354    valid_1's l2: 0.358382
# [1590]  training's l2: 0.0475369    valid_1's l2: 0.358361
# [1600]  training's l2: 0.0473317    valid_1's l2: 0.358362
# [1610]  training's l2: 0.0471286    valid_1's l2: 0.358334
# [1620]  training's l2: 0.0469544    valid_1's l2: 0.358316
# [1630]  training's l2: 0.0467905    valid_1's l2: 0.35832
# [1640]  training's l2: 0.0466176    valid_1's l2: 0.358298
# [1650]  training's l2: 0.0464516    valid_1's l2: 0.358276
# [1660]  training's l2: 0.0462815    valid_1's l2: 0.358266
# [1670]  training's l2: 0.0461121    valid_1's l2: 0.35825
# [1680]  training's l2: 0.0459588    valid_1's l2: 0.358231
# [1690]  training's l2: 0.0457907    valid_1's l2: 0.358215
# [1700]  training's l2: 0.0456059    valid_1's l2: 0.358218
# [1710]  training's l2: 0.0454434    valid_1's l2: 0.3582
# [1720]  training's l2: 0.0452935    valid_1's l2: 0.358208
# [1730]  training's l2: 0.0451347    valid_1's l2: 0.358196
# [1740]  training's l2: 0.0449855    valid_1's l2: 0.358203
# [1750]  training's l2: 0.0448311    valid_1's l2: 0.358199
# [1760]  training's l2: 0.0446897    valid_1's l2: 0.358168
# [1770]  training's l2: 0.04455  valid_1's l2: 0.358163
# [1780]  training's l2: 0.0444086    valid_1's l2: 0.358154
# [1790]  training's l2: 0.0442755    valid_1's l2: 0.358154
# [1800]  training's l2: 0.044137 valid_1's l2: 0.358156
# [1810]  training's l2: 0.0440062    valid_1's l2: 0.358142
# [1820]  training's l2: 0.0438788    valid_1's l2: 0.358136
# [1830]  training's l2: 0.0437461    valid_1's l2: 0.358127
# [1840]  training's l2: 0.0436202    valid_1's l2: 0.358127
# [1850]  training's l2: 0.0435005    valid_1's l2: 0.358147
# [1860]  training's l2: 0.0433669    valid_1's l2: 0.358141
# [1870]  training's l2: 0.0432557    valid_1's l2: 0.358142
# Early stopping, best iteration is:
# [1826]  training's l2: 0.0437958    valid_1's l2: 0.358124
# [LightGBM] [Info] Finished loading 1826 models
# 0.6256121957467803
# 0.5306554818551293
# [LightGBM] [Info] Total Bins 1283453
# [LightGBM] [Info] Number of data: 176000, number of used features: 56174
# Training until validation scores don't improve for 50 rounds.
# [10]    training's l2: 0.480016 valid_1's l2: 0.504535
# [20]    training's l2: 0.430731 valid_1's l2: 0.469956
# [30]    training's l2: 0.398779 valid_1's l2: 0.450497
# [40]    training's l2: 0.375534 valid_1's l2: 0.438359
# [50]    training's l2: 0.357054 valid_1's l2: 0.43002
# [60]    training's l2: 0.341227 valid_1's l2: 0.423596
# [70]    training's l2: 0.327184 valid_1's l2: 0.418199
# [80]    training's l2: 0.314881 valid_1's l2: 0.413911
# [90]    training's l2: 0.303669 valid_1's l2: 0.410197
# [100]   training's l2: 0.293362 valid_1's l2: 0.407248
# [110]   training's l2: 0.283826 valid_1's l2: 0.404725
# [120]   training's l2: 0.275003 valid_1's l2: 0.402637
# [130]   training's l2: 0.266758 valid_1's l2: 0.400526
# [140]   training's l2: 0.2592   valid_1's l2: 0.398714
# [150]   training's l2: 0.251948 valid_1's l2: 0.397091
# [160]   training's l2: 0.245291 valid_1's l2: 0.395536
# [170]   training's l2: 0.239204 valid_1's l2: 0.394177
# [180]   training's l2: 0.233656 valid_1's l2: 0.392997
# [190]   training's l2: 0.22819  valid_1's l2: 0.391939
# [200]   training's l2: 0.22307  valid_1's l2: 0.390796
# [210]   training's l2: 0.218087 valid_1's l2: 0.389971
# [220]   training's l2: 0.213402 valid_1's l2: 0.389014
# [230]   training's l2: 0.208798 valid_1's l2: 0.388192
# [240]   training's l2: 0.204381 valid_1's l2: 0.387429
# [250]   training's l2: 0.200109 valid_1's l2: 0.386626
# [260]   training's l2: 0.196126 valid_1's l2: 0.385998
# [270]   training's l2: 0.192235 valid_1's l2: 0.385274
# [280]   training's l2: 0.188336 valid_1's l2: 0.384598
# [290]   training's l2: 0.184903 valid_1's l2: 0.383911
# [300]   training's l2: 0.181102 valid_1's l2: 0.383217
# [310]   training's l2: 0.177662 valid_1's l2: 0.38269
# [320]   training's l2: 0.174239 valid_1's l2: 0.382132
# [330]   training's l2: 0.170942 valid_1's l2: 0.381525
# [340]   training's l2: 0.167739 valid_1's l2: 0.381074
# [350]   training's l2: 0.164973 valid_1's l2: 0.380519
# [360]   training's l2: 0.162085 valid_1's l2: 0.380146
# [370]   training's l2: 0.159479 valid_1's l2: 0.379776
# [380]   training's l2: 0.156576 valid_1's l2: 0.379325
# [390]   training's l2: 0.153936 valid_1's l2: 0.378914
# [400]   training's l2: 0.151345 valid_1's l2: 0.378559
# [410]   training's l2: 0.148913 valid_1's l2: 0.378271
# [420]   training's l2: 0.146508 valid_1's l2: 0.377932
# [430]   training's l2: 0.144166 valid_1's l2: 0.377661
# [440]   training's l2: 0.141948 valid_1's l2: 0.377361
# [450]   training's l2: 0.139715 valid_1's l2: 0.377028
# [460]   training's l2: 0.137336 valid_1's l2: 0.376715
# [470]   training's l2: 0.135191 valid_1's l2: 0.376467
# [480]   training's l2: 0.133006 valid_1's l2: 0.376132
# [490]   training's l2: 0.131054 valid_1's l2: 0.375947
# [500]   training's l2: 0.128856 valid_1's l2: 0.37551
# [510]   training's l2: 0.126932 valid_1's l2: 0.375198
# [520]   training's l2: 0.125157 valid_1's l2: 0.374949
# [530]   training's l2: 0.123201 valid_1's l2: 0.374726
# [540]   training's l2: 0.121444 valid_1's l2: 0.374498
# [550]   training's l2: 0.119633 valid_1's l2: 0.374345
# [560]   training's l2: 0.118    valid_1's l2: 0.37404
# [570]   training's l2: 0.116318 valid_1's l2: 0.373851
# [580]   training's l2: 0.114903 valid_1's l2: 0.373665
# [590]   training's l2: 0.113298 valid_1's l2: 0.373419
# [600]   training's l2: 0.111648 valid_1's l2: 0.373232
# [610]   training's l2: 0.11009  valid_1's l2: 0.372999
# [620]   training's l2: 0.108741 valid_1's l2: 0.372833
# [630]   training's l2: 0.107244 valid_1's l2: 0.3727
# [640]   training's l2: 0.105864 valid_1's l2: 0.372498
# [650]   training's l2: 0.104499 valid_1's l2: 0.372302
# [660]   training's l2: 0.103151 valid_1's l2: 0.372098
# [670]   training's l2: 0.101788 valid_1's l2: 0.371967
# [680]   training's l2: 0.100482 valid_1's l2: 0.371748
# [690]   training's l2: 0.0992183    valid_1's l2: 0.371525
# [700]   training's l2: 0.0979846    valid_1's l2: 0.371394
# [710]   training's l2: 0.0967652    valid_1's l2: 0.37129
# [720]   training's l2: 0.0955716    valid_1's l2: 0.371115
# [730]   training's l2: 0.094357 valid_1's l2: 0.371042
# [740]   training's l2: 0.0932122    valid_1's l2: 0.370956
# [750]   training's l2: 0.0920652    valid_1's l2: 0.370801
# [760]   training's l2: 0.0909937    valid_1's l2: 0.370703
# [770]   training's l2: 0.0898766    valid_1's l2: 0.370557
# [780]   training's l2: 0.0888882    valid_1's l2: 0.370455
# [790]   training's l2: 0.0878641    valid_1's l2: 0.370268
# [800]   training's l2: 0.0868563    valid_1's l2: 0.370118
# [810]   training's l2: 0.085929 valid_1's l2: 0.370008
# [820]   training's l2: 0.0849328    valid_1's l2: 0.369895
# [830]   training's l2: 0.0839838    valid_1's l2: 0.369717
# [840]   training's l2: 0.082976 valid_1's l2: 0.369571
# [850]   training's l2: 0.0819506    valid_1's l2: 0.369455
# [860]   training's l2: 0.0809874    valid_1's l2: 0.369276
# [870]   training's l2: 0.0801294    valid_1's l2: 0.369196
# [880]   training's l2: 0.0793099    valid_1's l2: 0.369136
# [890]   training's l2: 0.0784748    valid_1's l2: 0.369042
# [900]   training's l2: 0.0776167    valid_1's l2: 0.368924
# [910]   training's l2: 0.0769208    valid_1's l2: 0.368814
# [920]   training's l2: 0.0761853    valid_1's l2: 0.368689
# [930]   training's l2: 0.0754492    valid_1's l2: 0.368602
# [940]   training's l2: 0.0746586    valid_1's l2: 0.368504
# [950]   training's l2: 0.0739758    valid_1's l2: 0.368424
# [960]   training's l2: 0.0733119    valid_1's l2: 0.368369
# [970]   training's l2: 0.0726227    valid_1's l2: 0.368263
# [980]   training's l2: 0.0719282    valid_1's l2: 0.368192
# [990]   training's l2: 0.0712699    valid_1's l2: 0.368179
# [1000]  training's l2: 0.0706055    valid_1's l2: 0.368083
# [1010]  training's l2: 0.0699471    valid_1's l2: 0.368046
# [1020]  training's l2: 0.0693799    valid_1's l2: 0.368032
# [1030]  training's l2: 0.0687558    valid_1's l2: 0.367886
# [1040]  training's l2: 0.0681285    valid_1's l2: 0.367792
# [1050]  training's l2: 0.0674897    valid_1's l2: 0.367663
# [1060]  training's l2: 0.0669098    valid_1's l2: 0.367603
# [1070]  training's l2: 0.0664066    valid_1's l2: 0.367564
# [1080]  training's l2: 0.065877 valid_1's l2: 0.367557
# [1090]  training's l2: 0.0653128    valid_1's l2: 0.367484
# [1100]  training's l2: 0.0648096    valid_1's l2: 0.367414
# [1110]  training's l2: 0.064245 valid_1's l2: 0.367382
# [1120]  training's l2: 0.0637238    valid_1's l2: 0.367278
# [1130]  training's l2: 0.0632209    valid_1's l2: 0.367259
# [1140]  training's l2: 0.0627234    valid_1's l2: 0.367204
# [1150]  training's l2: 0.0622134    valid_1's l2: 0.367145
# [1160]  training's l2: 0.0617265    valid_1's l2: 0.367104
# [1170]  training's l2: 0.0612802    valid_1's l2: 0.367063
# [1180]  training's l2: 0.0608308    valid_1's l2: 0.367008
# [1190]  training's l2: 0.0603568    valid_1's l2: 0.366954
# [1200]  training's l2: 0.0599304    valid_1's l2: 0.366909
# [1210]  training's l2: 0.0594869    valid_1's l2: 0.366853
# [1220]  training's l2: 0.0590871    valid_1's l2: 0.366829
# [1230]  training's l2: 0.0586421    valid_1's l2: 0.366778
# [1240]  training's l2: 0.0582348    valid_1's l2: 0.366737
# [1250]  training's l2: 0.0578201    valid_1's l2: 0.366693
# [1260]  training's l2: 0.0574011    valid_1's l2: 0.366697
# [1270]  training's l2: 0.057035 valid_1's l2: 0.366721
# [1280]  training's l2: 0.0566182    valid_1's l2: 0.366639
# [1290]  training's l2: 0.056228 valid_1's l2: 0.366606
# [1300]  training's l2: 0.0558833    valid_1's l2: 0.366589
# [1310]  training's l2: 0.0555393    valid_1's l2: 0.366579
# [1320]  training's l2: 0.0552161    valid_1's l2: 0.366582
# [1330]  training's l2: 0.0548948    valid_1's l2: 0.36651
# [1340]  training's l2: 0.0545478    valid_1's l2: 0.366504
# [1350]  training's l2: 0.0542243    valid_1's l2: 0.366482
# [1360]  training's l2: 0.0539278    valid_1's l2: 0.366448
# [1370]  training's l2: 0.0536043    valid_1's l2: 0.366441
# [1380]  training's l2: 0.0533049    valid_1's l2: 0.366394
# [1390]  training's l2: 0.0529945    valid_1's l2: 0.366373
# [1400]  training's l2: 0.0527004    valid_1's l2: 0.366307
# [1410]  training's l2: 0.0523928    valid_1's l2: 0.366287
# [1420]  training's l2: 0.0521117    valid_1's l2: 0.366261
# [1430]  training's l2: 0.0518512    valid_1's l2: 0.366252
# [1440]  training's l2: 0.0515937    valid_1's l2: 0.366233
# [1450]  training's l2: 0.0513201    valid_1's l2: 0.366244
# [1460]  training's l2: 0.0510352    valid_1's l2: 0.366208
# [1470]  training's l2: 0.050779 valid_1's l2: 0.366211
# [1480]  training's l2: 0.0505267    valid_1's l2: 0.366202
# [1490]  training's l2: 0.050278 valid_1's l2: 0.366189
# [1500]  training's l2: 0.0500385    valid_1's l2: 0.36617
# [1510]  training's l2: 0.0498006    valid_1's l2: 0.366143
# [1520]  training's l2: 0.0495754    valid_1's l2: 0.36611
# [1530]  training's l2: 0.0493514    valid_1's l2: 0.366114
# [1540]  training's l2: 0.0491191    valid_1's l2: 0.366078
# [1550]  training's l2: 0.0489032    valid_1's l2: 0.366071
# [1560]  training's l2: 0.0486903    valid_1's l2: 0.366059
# [1570]  training's l2: 0.0484609    valid_1's l2: 0.366042
# [1580]  training's l2: 0.0482638    valid_1's l2: 0.36603
# [1590]  training's l2: 0.0480563    valid_1's l2: 0.366011
# [1600]  training's l2: 0.0478625    valid_1's l2: 0.36602
# [1610]  training's l2: 0.0476705    valid_1's l2: 0.365994
# [1620]  training's l2: 0.047475 valid_1's l2: 0.365985
# [1630]  training's l2: 0.0472872    valid_1's l2: 0.365969
# [1640]  training's l2: 0.0471087    valid_1's l2: 0.365971
# [1650]  training's l2: 0.0469315    valid_1's l2: 0.365962
# [1660]  training's l2: 0.0467581    valid_1's l2: 0.365986
# [1670]  training's l2: 0.0465681    valid_1's l2: 0.365968
# [1680]  training's l2: 0.0464   valid_1's l2: 0.36597
# [1690]  training's l2: 0.0462177    valid_1's l2: 0.365954
# [1700]  training's l2: 0.0460502    valid_1's l2: 0.365945
# [1710]  training's l2: 0.0458891    valid_1's l2: 0.365933
# [1720]  training's l2: 0.0457347    valid_1's l2: 0.365921
# [1730]  training's l2: 0.0455896    valid_1's l2: 0.365942
# [1740]  training's l2: 0.0454223    valid_1's l2: 0.365947
# [1750]  training's l2: 0.0452557    valid_1's l2: 0.365934
# [1760]  training's l2: 0.0451108    valid_1's l2: 0.365928
# Early stopping, best iteration is:
# [1719]  training's l2: 0.0457506    valid_1's l2: 0.36592
# [LightGBM] [Info] Finished loading 1719 models
# 0.6230867833277934
# 0.5313629276423931
# [LightGBM] [Info] Total Bins 1280274
# [LightGBM] [Info] Number of data: 176000, number of used features: 55811
# Training until validation scores don't improve for 50 rounds.
# [10]    training's l2: 0.483639 valid_1's l2: 0.488516
# [20]    training's l2: 0.433592 valid_1's l2: 0.455014
# [30]    training's l2: 0.400879 valid_1's l2: 0.436453
# [40]    training's l2: 0.377295 valid_1's l2: 0.425439
# [50]    training's l2: 0.358604 valid_1's l2: 0.417936
# [60]    training's l2: 0.342706 valid_1's l2: 0.41221
# [70]    training's l2: 0.328645 valid_1's l2: 0.407405
# [80]    training's l2: 0.316091 valid_1's l2: 0.403243
# [90]    training's l2: 0.304787 valid_1's l2: 0.399806
# [100]   training's l2: 0.294435 valid_1's l2: 0.396833
# [110]   training's l2: 0.284891 valid_1's l2: 0.394131
# [120]   training's l2: 0.276092 valid_1's l2: 0.391879
# [130]   training's l2: 0.267686 valid_1's l2: 0.390042
# [140]   training's l2: 0.259908 valid_1's l2: 0.388287
# [150]   training's l2: 0.252697 valid_1's l2: 0.386521
# [160]   training's l2: 0.246225 valid_1's l2: 0.38512
# [170]   training's l2: 0.239662 valid_1's l2: 0.383762
# [180]   training's l2: 0.233963 valid_1's l2: 0.382494
# [190]   training's l2: 0.228352 valid_1's l2: 0.381315
# [200]   training's l2: 0.223225 valid_1's l2: 0.38015
# [210]   training's l2: 0.21814  valid_1's l2: 0.379128
# [220]   training's l2: 0.213453 valid_1's l2: 0.378498
# [230]   training's l2: 0.208997 valid_1's l2: 0.377655
# [240]   training's l2: 0.204718 valid_1's l2: 0.376955
# [250]   training's l2: 0.200391 valid_1's l2: 0.376101
# [260]   training's l2: 0.19633  valid_1's l2: 0.375406
# [270]   training's l2: 0.192322 valid_1's l2: 0.374713
# [280]   training's l2: 0.188534 valid_1's l2: 0.374086
# [290]   training's l2: 0.18494  valid_1's l2: 0.373451
# [300]   training's l2: 0.181228 valid_1's l2: 0.372671
# [310]   training's l2: 0.177999 valid_1's l2: 0.372143
# [320]   training's l2: 0.17503  valid_1's l2: 0.371703
# [330]   training's l2: 0.171828 valid_1's l2: 0.371114
# [340]   training's l2: 0.168644 valid_1's l2: 0.370627
# [350]   training's l2: 0.165493 valid_1's l2: 0.370221
# [360]   training's l2: 0.162584 valid_1's l2: 0.369723
# [370]   training's l2: 0.159625 valid_1's l2: 0.369259
# [380]   training's l2: 0.157059 valid_1's l2: 0.36896
# [390]   training's l2: 0.154419 valid_1's l2: 0.368611
# [400]   training's l2: 0.15181  valid_1's l2: 0.368233
# [410]   training's l2: 0.149465 valid_1's l2: 0.367892
# [420]   training's l2: 0.147007 valid_1's l2: 0.367485
# [430]   training's l2: 0.144483 valid_1's l2: 0.367126
# [440]   training's l2: 0.14216  valid_1's l2: 0.366813
# [450]   training's l2: 0.139713 valid_1's l2: 0.366494
# [460]   training's l2: 0.13744  valid_1's l2: 0.366039
# [470]   training's l2: 0.135229 valid_1's l2: 0.365708
# [480]   training's l2: 0.133105 valid_1's l2: 0.365483
# [490]   training's l2: 0.13098  valid_1's l2: 0.365175
# [500]   training's l2: 0.128993 valid_1's l2: 0.364907
# [510]   training's l2: 0.127064 valid_1's l2: 0.3646
# [520]   training's l2: 0.125217 valid_1's l2: 0.364432
# [530]   training's l2: 0.123393 valid_1's l2: 0.364221
# [540]   training's l2: 0.12144  valid_1's l2: 0.363972
# [550]   training's l2: 0.119488 valid_1's l2: 0.363741
# [560]   training's l2: 0.117622 valid_1's l2: 0.363506
# [570]   training's l2: 0.115947 valid_1's l2: 0.363263
# [580]   training's l2: 0.11438  valid_1's l2: 0.362987
# [590]   training's l2: 0.112713 valid_1's l2: 0.362769
# [600]   training's l2: 0.111127 valid_1's l2: 0.362514
# [610]   training's l2: 0.109646 valid_1's l2: 0.362338
# [620]   training's l2: 0.108051 valid_1's l2: 0.362096
# [630]   training's l2: 0.106687 valid_1's l2: 0.361945
# [640]   training's l2: 0.105179 valid_1's l2: 0.361738
# [650]   training's l2: 0.103792 valid_1's l2: 0.361499
# [660]   training's l2: 0.102496 valid_1's l2: 0.361341
# [670]   training's l2: 0.10117  valid_1's l2: 0.361155
# [680]   training's l2: 0.0998769    valid_1's l2: 0.361009
# [690]   training's l2: 0.0985823    valid_1's l2: 0.360767
# [700]   training's l2: 0.0972911    valid_1's l2: 0.360643
# [710]   training's l2: 0.0960508    valid_1's l2: 0.360454
# [720]   training's l2: 0.0948488    valid_1's l2: 0.36026
# [730]   training's l2: 0.0936177    valid_1's l2: 0.360108
# [740]   training's l2: 0.0923843    valid_1's l2: 0.35991
# [750]   training's l2: 0.0912425    valid_1's l2: 0.359766
# [760]   training's l2: 0.0900654    valid_1's l2: 0.359597
# [770]   training's l2: 0.088972 valid_1's l2: 0.359454
# [780]   training's l2: 0.0879844    valid_1's l2: 0.35937
# [790]   training's l2: 0.0869093    valid_1's l2: 0.359265
# [800]   training's l2: 0.0858205    valid_1's l2: 0.359126
# [810]   training's l2: 0.0848045    valid_1's l2: 0.358995
# [820]   training's l2: 0.0838983    valid_1's l2: 0.358833
# [830]   training's l2: 0.0828972    valid_1's l2: 0.358681
# [840]   training's l2: 0.0819985    valid_1's l2: 0.358522
# [850]   training's l2: 0.0810557    valid_1's l2: 0.358403
# [860]   training's l2: 0.0801015    valid_1's l2: 0.358294
# [870]   training's l2: 0.0792214    valid_1's l2: 0.358171
# [880]   training's l2: 0.0782904    valid_1's l2: 0.358035
# [890]   training's l2: 0.0774742    valid_1's l2: 0.357919
# [900]   training's l2: 0.0766686    valid_1's l2: 0.357775
# [910]   training's l2: 0.0758693    valid_1's l2: 0.357703
# [920]   training's l2: 0.075115 valid_1's l2: 0.357639
# [930]   training's l2: 0.0743434    valid_1's l2: 0.357546
# [940]   training's l2: 0.0735995    valid_1's l2: 0.357456
# [950]   training's l2: 0.0728754    valid_1's l2: 0.357347
# [960]   training's l2: 0.0721666    valid_1's l2: 0.357292
# [970]   training's l2: 0.0714941    valid_1's l2: 0.357238
# [980]   training's l2: 0.0708985    valid_1's l2: 0.357216
# [990]   training's l2: 0.0703058    valid_1's l2: 0.357165
# [1000]  training's l2: 0.0696568    valid_1's l2: 0.357112
# [1010]  training's l2: 0.0689401    valid_1's l2: 0.357041
# [1020]  training's l2: 0.0682432    valid_1's l2: 0.356959
# [1030]  training's l2: 0.0675745    valid_1's l2: 0.356914
# [1040]  training's l2: 0.0669853    valid_1's l2: 0.3568
# [1050]  training's l2: 0.0663816    valid_1's l2: 0.356634
# [1060]  training's l2: 0.0658097    valid_1's l2: 0.356563
# [1070]  training's l2: 0.0652474    valid_1's l2: 0.356441
# [1080]  training's l2: 0.0646928    valid_1's l2: 0.356381
# [1090]  training's l2: 0.0641626    valid_1's l2: 0.356336
# [1100]  training's l2: 0.0636453    valid_1's l2: 0.356306
# [1110]  training's l2: 0.0631011    valid_1's l2: 0.356194
# [1120]  training's l2: 0.062599 valid_1's l2: 0.356128
# [1130]  training's l2: 0.0621409    valid_1's l2: 0.356065
# [1140]  training's l2: 0.0616028    valid_1's l2: 0.355966
# [1150]  training's l2: 0.0610801    valid_1's l2: 0.355892
# [1160]  training's l2: 0.060634 valid_1's l2: 0.355837
# [1170]  training's l2: 0.0601629    valid_1's l2: 0.355789
# [1180]  training's l2: 0.0597185    valid_1's l2: 0.355765
# [1190]  training's l2: 0.0592497    valid_1's l2: 0.355723
# [1200]  training's l2: 0.0587554    valid_1's l2: 0.355671
# [1210]  training's l2: 0.0582772    valid_1's l2: 0.355625
# [1220]  training's l2: 0.0578585    valid_1's l2: 0.35554
# [1230]  training's l2: 0.0574687    valid_1's l2: 0.355514
# [1240]  training's l2: 0.0570561    valid_1's l2: 0.355478
# [1250]  training's l2: 0.0566688    valid_1's l2: 0.355426
# [1260]  training's l2: 0.0562926    valid_1's l2: 0.355416
# [1270]  training's l2: 0.0558614    valid_1's l2: 0.355386
# [1280]  training's l2: 0.0554918    valid_1's l2: 0.355367
# [1290]  training's l2: 0.0551707    valid_1's l2: 0.355338
# [1300]  training's l2: 0.0548251    valid_1's l2: 0.355316
# [1310]  training's l2: 0.0544911    valid_1's l2: 0.355284
# [1320]  training's l2: 0.0541525    valid_1's l2: 0.355211
# [1330]  training's l2: 0.0538424    valid_1's l2: 0.355206
# [1340]  training's l2: 0.0535209    valid_1's l2: 0.355187
# [1350]  training's l2: 0.053203 valid_1's l2: 0.355137
# [1360]  training's l2: 0.0529113    valid_1's l2: 0.355111
# [1370]  training's l2: 0.0525994    valid_1's l2: 0.35509
# [1380]  training's l2: 0.0522696    valid_1's l2: 0.355049
# [1390]  training's l2: 0.0519516    valid_1's l2: 0.355021
# [1400]  training's l2: 0.0516501    valid_1's l2: 0.35497
# [1410]  training's l2: 0.0513485    valid_1's l2: 0.354938
# [1420]  training's l2: 0.0510769    valid_1's l2: 0.354908
# [1430]  training's l2: 0.0507997    valid_1's l2: 0.354907
# [1440]  training's l2: 0.0505442    valid_1's l2: 0.354908
# [1450]  training's l2: 0.0502629    valid_1's l2: 0.354905
# [1460]  training's l2: 0.0499903    valid_1's l2: 0.354924
# [1470]  training's l2: 0.0497212    valid_1's l2: 0.354895
# [1480]  training's l2: 0.0494843    valid_1's l2: 0.35487
# [1490]  training's l2: 0.0492104    valid_1's l2: 0.354846
# [1500]  training's l2: 0.0489754    valid_1's l2: 0.354791
# [1510]  training's l2: 0.0487471    valid_1's l2: 0.354773
# [1520]  training's l2: 0.0485072    valid_1's l2: 0.354804
# [1530]  training's l2: 0.0482873    valid_1's l2: 0.354787
# [1540]  training's l2: 0.0480569    valid_1's l2: 0.354762
# [1550]  training's l2: 0.0478673    valid_1's l2: 0.354742
# [1560]  training's l2: 0.0476491    valid_1's l2: 0.354709
# [1570]  training's l2: 0.0474519    valid_1's l2: 0.354695
# [1580]  training's l2: 0.0472475    valid_1's l2: 0.354685
# [1590]  training's l2: 0.0470496    valid_1's l2: 0.354671
# [1600]  training's l2: 0.046839 valid_1's l2: 0.354681
# [1610]  training's l2: 0.0466335    valid_1's l2: 0.354655
# [1620]  training's l2: 0.0464356    valid_1's l2: 0.354655
# [1630]  training's l2: 0.0462538    valid_1's l2: 0.354652
# [1640]  training's l2: 0.0460649    valid_1's l2: 0.354616
# [1650]  training's l2: 0.0458915    valid_1's l2: 0.35459
# [1660]  training's l2: 0.0456951    valid_1's l2: 0.354609
# [1670]  training's l2: 0.0455263    valid_1's l2: 0.354596
# [1680]  training's l2: 0.0453646    valid_1's l2: 0.354585
# [1690]  training's l2: 0.0451984    valid_1's l2: 0.354595
# [1700]  training's l2: 0.0450404    valid_1's l2: 0.354573
# [1710]  training's l2: 0.0448608    valid_1's l2: 0.35458
# [1720]  training's l2: 0.0447101    valid_1's l2: 0.354582
# [1730]  training's l2: 0.0445522    valid_1's l2: 0.354586
# [1740]  training's l2: 0.0443886    valid_1's l2: 0.354602
# [1750]  training's l2: 0.0442356    valid_1's l2: 0.354616
# Early stopping, best iteration is:
# [1703]  training's l2: 0.0449904    valid_1's l2: 0.354566
# [LightGBM] [Info] Finished loading 1703 models
# 0.6267807355409513
# 0.5314357068589701
# TF-IDF字......
# (220000, 11772915)
# (220000, 1048576)
# X.shape: (220000, 12822687)
# test_hh.shape: (50000, 12822687)
# lgb_zi_5fold training......
# [LightGBM] [Info] Total Bins 6674788
# [LightGBM] [Info] Number of data: 176000, number of used features: 299512
# Training until validation scores don't improve for 50 rounds.
# [10]    training's l2: 0.478893 valid_1's l2: 0.497827
# [20]    training's l2: 0.428652 valid_1's l2: 0.462521
# [30]    training's l2: 0.396288 valid_1's l2: 0.442607
# [40]    training's l2: 0.372644 valid_1's l2: 0.430511
# [50]    training's l2: 0.35406  valid_1's l2: 0.422337
# [60]    training's l2: 0.338411 valid_1's l2: 0.416387
# [70]    training's l2: 0.324471 valid_1's l2: 0.411224
# [80]    training's l2: 0.312058 valid_1's l2: 0.407235
# [90]    training's l2: 0.300773 valid_1's l2: 0.403602
# [100]   training's l2: 0.290441 valid_1's l2: 0.400537
# [110]   training's l2: 0.280974 valid_1's l2: 0.397899
# [120]   training's l2: 0.271971 valid_1's l2: 0.395264
# [130]   training's l2: 0.263621 valid_1's l2: 0.393049
# [140]   training's l2: 0.255983 valid_1's l2: 0.391269
# [150]   training's l2: 0.248968 valid_1's l2: 0.389519
# [160]   training's l2: 0.242155 valid_1's l2: 0.388098
# [170]   training's l2: 0.235921 valid_1's l2: 0.386621
# [180]   training's l2: 0.229993 valid_1's l2: 0.385224
# [190]   training's l2: 0.224302 valid_1's l2: 0.384064
# [200]   training's l2: 0.218935 valid_1's l2: 0.383045
# [210]   training's l2: 0.214006 valid_1's l2: 0.381999
# [220]   training's l2: 0.209388 valid_1's l2: 0.380968
# [230]   training's l2: 0.204712 valid_1's l2: 0.380197
# [240]   training's l2: 0.200536 valid_1's l2: 0.379328
# [250]   training's l2: 0.196277 valid_1's l2: 0.378499
# [260]   training's l2: 0.192432 valid_1's l2: 0.377813
# [270]   training's l2: 0.188492 valid_1's l2: 0.377159
# [280]   training's l2: 0.184785 valid_1's l2: 0.376422
# [290]   training's l2: 0.181256 valid_1's l2: 0.375763
# [300]   training's l2: 0.177675 valid_1's l2: 0.375116
# [310]   training's l2: 0.17454  valid_1's l2: 0.374567
# [320]   training's l2: 0.17133  valid_1's l2: 0.374036
# [330]   training's l2: 0.168219 valid_1's l2: 0.373365
# [340]   training's l2: 0.16545  valid_1's l2: 0.372754
# [350]   training's l2: 0.16249  valid_1's l2: 0.372286
# [360]   training's l2: 0.159526 valid_1's l2: 0.37179
# [370]   training's l2: 0.15682  valid_1's l2: 0.371384
# [380]   training's l2: 0.154085 valid_1's l2: 0.370925
# [390]   training's l2: 0.151671 valid_1's l2: 0.370704
# [400]   training's l2: 0.149325 valid_1's l2: 0.370262
# [410]   training's l2: 0.146885 valid_1's l2: 0.36974
# [420]   training's l2: 0.144418 valid_1's l2: 0.369333
# [430]   training's l2: 0.142227 valid_1's l2: 0.369008
# [440]   training's l2: 0.139979 valid_1's l2: 0.368588
# [450]   training's l2: 0.137704 valid_1's l2: 0.368222
# [460]   training's l2: 0.135658 valid_1's l2: 0.367898
# [470]   training's l2: 0.133426 valid_1's l2: 0.367552
# [480]   training's l2: 0.131198 valid_1's l2: 0.36721
# [490]   training's l2: 0.129139 valid_1's l2: 0.366885
# [500]   training's l2: 0.127147 valid_1's l2: 0.366638
# [510]   training's l2: 0.12519  valid_1's l2: 0.366335
# [520]   training's l2: 0.123362 valid_1's l2: 0.366066
# [530]   training's l2: 0.121506 valid_1's l2: 0.365843
# [540]   training's l2: 0.119978 valid_1's l2: 0.365595
# [550]   training's l2: 0.118327 valid_1's l2: 0.365298
# [560]   training's l2: 0.116588 valid_1's l2: 0.365076
# [570]   training's l2: 0.114883 valid_1's l2: 0.364899
# [580]   training's l2: 0.113303 valid_1's l2: 0.364689
# [590]   training's l2: 0.111815 valid_1's l2: 0.364506
# [600]   training's l2: 0.110267 valid_1's l2: 0.364296
# [610]   training's l2: 0.108701 valid_1's l2: 0.364057
# [620]   training's l2: 0.107332 valid_1's l2: 0.363879
# [630]   training's l2: 0.105905 valid_1's l2: 0.363671
# [640]   training's l2: 0.104494 valid_1's l2: 0.363464
# [650]   training's l2: 0.103203 valid_1's l2: 0.363332
# [660]   training's l2: 0.101802 valid_1's l2: 0.36317
# [670]   training's l2: 0.100539 valid_1's l2: 0.362954
# [680]   training's l2: 0.0992748    valid_1's l2: 0.362809
# [690]   training's l2: 0.0980733    valid_1's l2: 0.362623
# [700]   training's l2: 0.0968474    valid_1's l2: 0.362457
# [710]   training's l2: 0.0956591    valid_1's l2: 0.362324
# [720]   training's l2: 0.0944365    valid_1's l2: 0.362183
# [730]   training's l2: 0.0934123    valid_1's l2: 0.362016
# [740]   training's l2: 0.0922355    valid_1's l2: 0.361866
# [750]   training's l2: 0.0911575    valid_1's l2: 0.361724
# [760]   training's l2: 0.0900728    valid_1's l2: 0.361631
# [770]   training's l2: 0.0890639    valid_1's l2: 0.361542
# [780]   training's l2: 0.0880761    valid_1's l2: 0.361406
# [790]   training's l2: 0.0871123    valid_1's l2: 0.361345
# [800]   training's l2: 0.0860828    valid_1's l2: 0.361257
# [810]   training's l2: 0.0851013    valid_1's l2: 0.361143
# [820]   training's l2: 0.0841616    valid_1's l2: 0.360993
# [830]   training's l2: 0.0831972    valid_1's l2: 0.36095
# [840]   training's l2: 0.0823213    valid_1's l2: 0.360859
# [850]   training's l2: 0.0813833    valid_1's l2: 0.360747
# [860]   training's l2: 0.0806349    valid_1's l2: 0.360656
# [870]   training's l2: 0.0798192    valid_1's l2: 0.360611
# [880]   training's l2: 0.0790282    valid_1's l2: 0.360537
# [890]   training's l2: 0.0781936    valid_1's l2: 0.360328
# [900]   training's l2: 0.0774011    valid_1's l2: 0.360285
# [910]   training's l2: 0.0765488    valid_1's l2: 0.360185
# [920]   training's l2: 0.0757132    valid_1's l2: 0.360175
# [930]   training's l2: 0.0749953    valid_1's l2: 0.360023
# [940]   training's l2: 0.0742753    valid_1's l2: 0.359941
# [950]   training's l2: 0.073548 valid_1's l2: 0.359847
# [960]   training's l2: 0.0728412    valid_1's l2: 0.359796
# [970]   training's l2: 0.0721198    valid_1's l2: 0.359721
# [980]   training's l2: 0.0714549    valid_1's l2: 0.359672
# [990]   training's l2: 0.0708248    valid_1's l2: 0.359611
# [1000]  training's l2: 0.0701341    valid_1's l2: 0.35963
# [1010]  training's l2: 0.0695188    valid_1's l2: 0.359544
# [1020]  training's l2: 0.0688826    valid_1's l2: 0.359457
# [1030]  training's l2: 0.0682584    valid_1's l2: 0.359357
# [1040]  training's l2: 0.0676252    valid_1's l2: 0.359314
# [1050]  training's l2: 0.0670725    valid_1's l2: 0.359304
# [1060]  training's l2: 0.066522 valid_1's l2: 0.3593
# [1070]  training's l2: 0.0660196    valid_1's l2: 0.359224
# [1080]  training's l2: 0.0654179    valid_1's l2: 0.35914
# [1090]  training's l2: 0.0649035    valid_1's l2: 0.359107
# [1100]  training's l2: 0.0644085    valid_1's l2: 0.359084
# [1110]  training's l2: 0.0638626    valid_1's l2: 0.359028
# [1120]  training's l2: 0.0633554    valid_1's l2: 0.359005
# [1130]  training's l2: 0.0628539    valid_1's l2: 0.359
# [1140]  training's l2: 0.0623704    valid_1's l2: 0.358955
# [1150]  training's l2: 0.0619332    valid_1's l2: 0.358911
# [1160]  training's l2: 0.0614981    valid_1's l2: 0.358943
# [1170]  training's l2: 0.0610259    valid_1's l2: 0.358921
# [1180]  training's l2: 0.0605835    valid_1's l2: 0.358871
# [1190]  training's l2: 0.0601412    valid_1's l2: 0.35886
# [1200]  training's l2: 0.0596924    valid_1's l2: 0.358882
# [1210]  training's l2: 0.0592658    valid_1's l2: 0.358879
# [1220]  training's l2: 0.0588196    valid_1's l2: 0.358808
# [1230]  training's l2: 0.0584123    valid_1's l2: 0.358739
# [1240]  training's l2: 0.0580366    valid_1's l2: 0.358722
# [1250]  training's l2: 0.0576557    valid_1's l2: 0.358673
# [1260]  training's l2: 0.0572955    valid_1's l2: 0.358625
# [1270]  training's l2: 0.0568579    valid_1's l2: 0.358638
# [1280]  training's l2: 0.0564456    valid_1's l2: 0.358619
# [1290]  training's l2: 0.0560964    valid_1's l2: 0.358608
# [1300]  training's l2: 0.0557371    valid_1's l2: 0.3586
# [1310]  training's l2: 0.0553776    valid_1's l2: 0.358567
# [1320]  training's l2: 0.0550456    valid_1's l2: 0.358579
# [1330]  training's l2: 0.0547247    valid_1's l2: 0.358516
# [1340]  training's l2: 0.0543655    valid_1's l2: 0.358494
# [1350]  training's l2: 0.0540491    valid_1's l2: 0.358482
# [1360]  training's l2: 0.0537194    valid_1's l2: 0.358437
# [1370]  training's l2: 0.053383 valid_1's l2: 0.35839
# [1380]  training's l2: 0.0530141    valid_1's l2: 0.358357
# [1390]  training's l2: 0.0527118    valid_1's l2: 0.358353
# [1400]  training's l2: 0.0524084    valid_1's l2: 0.358318
# [1410]  training's l2: 0.0520996    valid_1's l2: 0.35832
# [1420]  training's l2: 0.0518017    valid_1's l2: 0.358293
# [1430]  training's l2: 0.0514943    valid_1's l2: 0.3583
# [1440]  training's l2: 0.051212 valid_1's l2: 0.358303
# [1450]  training's l2: 0.0509147    valid_1's l2: 0.358266
# [1460]  training's l2: 0.0506252    valid_1's l2: 0.358237
# [1470]  training's l2: 0.0503504    valid_1's l2: 0.358236
# [1480]  training's l2: 0.0500592    valid_1's l2: 0.358216
# [1490]  training's l2: 0.0497998    valid_1's l2: 0.358185
# [1500]  training's l2: 0.0495523    valid_1's l2: 0.35819
# [1510]  training's l2: 0.0492934    valid_1's l2: 0.3582
# [1520]  training's l2: 0.0490342    valid_1's l2: 0.358171
# [1530]  training's l2: 0.0488288    valid_1's l2: 0.358173
# [1540]  training's l2: 0.0486003    valid_1's l2: 0.358158
# [1550]  training's l2: 0.0483894    valid_1's l2: 0.358151
# [1560]  training's l2: 0.0481567    valid_1's l2: 0.358125
# [1570]  training's l2: 0.0479204    valid_1's l2: 0.35813
# [1580]  training's l2: 0.0476994    valid_1's l2: 0.358153
# [1590]  training's l2: 0.0474823    valid_1's l2: 0.358143
# [1600]  training's l2: 0.0472898    valid_1's l2: 0.358147
# [1610]  training's l2: 0.0470925    valid_1's l2: 0.358108
# [1620]  training's l2: 0.0468751    valid_1's l2: 0.358103
# [1630]  training's l2: 0.0466689    valid_1's l2: 0.358102
# [1640]  training's l2: 0.0464813    valid_1's l2: 0.358099
# [1650]  training's l2: 0.046283 valid_1's l2: 0.358096
# [1660]  training's l2: 0.0460744    valid_1's l2: 0.358097
# [1670]  training's l2: 0.0458841    valid_1's l2: 0.358095
# [1680]  training's l2: 0.0456768    valid_1's l2: 0.358095
# Early stopping, best iteration is:
# [1634]  training's l2: 0.0465852    valid_1's l2: 0.358084
# [LightGBM] [Info] Finished loading 1634 models
# 0.625625216036441
# 0.5303338864722368
# [LightGBM] [Info] Total Bins 6662314
# [LightGBM] [Info] Number of data: 176000, number of used features: 298886
# Training until validation scores don't improve for 50 rounds.
# [10]    training's l2: 0.478393 valid_1's l2: 0.501504
# [20]    training's l2: 0.428224 valid_1's l2: 0.464867
# [30]    training's l2: 0.395866 valid_1's l2: 0.444978
# [40]    training's l2: 0.372484 valid_1's l2: 0.433026
# [50]    training's l2: 0.353627 valid_1's l2: 0.424785
# [60]    training's l2: 0.337796 valid_1's l2: 0.41867
# [70]    training's l2: 0.323827 valid_1's l2: 0.413752
# [80]    training's l2: 0.311271 valid_1's l2: 0.409826
# [90]    training's l2: 0.300006 valid_1's l2: 0.406359
# [100]   training's l2: 0.289733 valid_1's l2: 0.40341
# [110]   training's l2: 0.280249 valid_1's l2: 0.400736
# [120]   training's l2: 0.271328 valid_1's l2: 0.398363
# [130]   training's l2: 0.263135 valid_1's l2: 0.396446
# [140]   training's l2: 0.2555   valid_1's l2: 0.394644
# [150]   training's l2: 0.248349 valid_1's l2: 0.39298
# [160]   training's l2: 0.24201  valid_1's l2: 0.39146
# [170]   training's l2: 0.23583  valid_1's l2: 0.390215
# [180]   training's l2: 0.230105 valid_1's l2: 0.389118
# [190]   training's l2: 0.224621 valid_1's l2: 0.38812
# [200]   training's l2: 0.219317 valid_1's l2: 0.387208
# [210]   training's l2: 0.214347 valid_1's l2: 0.386266
# [220]   training's l2: 0.209672 valid_1's l2: 0.385255
# [230]   training's l2: 0.205192 valid_1's l2: 0.384344
# [240]   training's l2: 0.20069  valid_1's l2: 0.383522
# [250]   training's l2: 0.196516 valid_1's l2: 0.382777
# [260]   training's l2: 0.192522 valid_1's l2: 0.382128
# [270]   training's l2: 0.18864  valid_1's l2: 0.381455
# [280]   training's l2: 0.185075 valid_1's l2: 0.38087
# [290]   training's l2: 0.181511 valid_1's l2: 0.380378
# [300]   training's l2: 0.178086 valid_1's l2: 0.379818
# [310]   training's l2: 0.174654 valid_1's l2: 0.379309
# [320]   training's l2: 0.171427 valid_1's l2: 0.37878
# [330]   training's l2: 0.168418 valid_1's l2: 0.378252
# [340]   training's l2: 0.165336 valid_1's l2: 0.377782
# [350]   training's l2: 0.16262  valid_1's l2: 0.377326
# [360]   training's l2: 0.159768 valid_1's l2: 0.376829
# [370]   training's l2: 0.15697  valid_1's l2: 0.376351
# [380]   training's l2: 0.154201 valid_1's l2: 0.375957
# [390]   training's l2: 0.151406 valid_1's l2: 0.37553
# [400]   training's l2: 0.148922 valid_1's l2: 0.37515
# [410]   training's l2: 0.146582 valid_1's l2: 0.374786
# [420]   training's l2: 0.144144 valid_1's l2: 0.3744
# [430]   training's l2: 0.141728 valid_1's l2: 0.37411
# [440]   training's l2: 0.139523 valid_1's l2: 0.373793
# [450]   training's l2: 0.137353 valid_1's l2: 0.373476
# [460]   training's l2: 0.13532  valid_1's l2: 0.37316
# [470]   training's l2: 0.133017 valid_1's l2: 0.372922
# [480]   training's l2: 0.131078 valid_1's l2: 0.37263
# [490]   training's l2: 0.129128 valid_1's l2: 0.372316
# [500]   training's l2: 0.127317 valid_1's l2: 0.372099
# [510]   training's l2: 0.125619 valid_1's l2: 0.37189
# [520]   training's l2: 0.123696 valid_1's l2: 0.371555
# [530]   training's l2: 0.121898 valid_1's l2: 0.371296
# [540]   training's l2: 0.120222 valid_1's l2: 0.371059
# [550]   training's l2: 0.118508 valid_1's l2: 0.370812
# [560]   training's l2: 0.116913 valid_1's l2: 0.370656
# [570]   training's l2: 0.115238 valid_1's l2: 0.370419
# [580]   training's l2: 0.113626 valid_1's l2: 0.370294
# [590]   training's l2: 0.112022 valid_1's l2: 0.370108
# [600]   training's l2: 0.110408 valid_1's l2: 0.370001
# [610]   training's l2: 0.108925 valid_1's l2: 0.369889
# [620]   training's l2: 0.107422 valid_1's l2: 0.369715
# [630]   training's l2: 0.10602  valid_1's l2: 0.369595
# [640]   training's l2: 0.104633 valid_1's l2: 0.369405
# [650]   training's l2: 0.103297 valid_1's l2: 0.369227
# [660]   training's l2: 0.102    valid_1's l2: 0.36905
# [670]   training's l2: 0.100686 valid_1's l2: 0.368907
# [680]   training's l2: 0.0995156    valid_1's l2: 0.368776
# [690]   training's l2: 0.0983   valid_1's l2: 0.368627
# [700]   training's l2: 0.0971798    valid_1's l2: 0.368421
# [710]   training's l2: 0.095969 valid_1's l2: 0.368253
# [720]   training's l2: 0.094771 valid_1's l2: 0.368147
# [730]   training's l2: 0.0936954    valid_1's l2: 0.367937
# [740]   training's l2: 0.0926493    valid_1's l2: 0.367826
# [750]   training's l2: 0.0915773    valid_1's l2: 0.367718
# [760]   training's l2: 0.0904795    valid_1's l2: 0.367647
# [770]   training's l2: 0.0894065    valid_1's l2: 0.367517
# [780]   training's l2: 0.0882787    valid_1's l2: 0.367426
# [790]   training's l2: 0.0873234    valid_1's l2: 0.367277
# [800]   training's l2: 0.0864104    valid_1's l2: 0.367221
# [810]   training's l2: 0.0854167    valid_1's l2: 0.36719
# [820]   training's l2: 0.084425 valid_1's l2: 0.367122
# [830]   training's l2: 0.0835826    valid_1's l2: 0.367106
# [840]   training's l2: 0.0826294    valid_1's l2: 0.367064
# [850]   training's l2: 0.081747 valid_1's l2: 0.366951
# [860]   training's l2: 0.0808201    valid_1's l2: 0.366896
# [870]   training's l2: 0.0800129    valid_1's l2: 0.366815
# [880]   training's l2: 0.0791191    valid_1's l2: 0.366733
# [890]   training's l2: 0.0783733    valid_1's l2: 0.3667
# [900]   training's l2: 0.0775829    valid_1's l2: 0.366578
# [910]   training's l2: 0.0767394    valid_1's l2: 0.366449
# [920]   training's l2: 0.0759778    valid_1's l2: 0.366408
# [930]   training's l2: 0.0753183    valid_1's l2: 0.366287
# [940]   training's l2: 0.0745402    valid_1's l2: 0.366267
# [950]   training's l2: 0.0738212    valid_1's l2: 0.366213
# [960]   training's l2: 0.073153 valid_1's l2: 0.366157
# [970]   training's l2: 0.0725072    valid_1's l2: 0.366118
# [980]   training's l2: 0.0718355    valid_1's l2: 0.366062
# [990]   training's l2: 0.0711461    valid_1's l2: 0.366063
# [1000]  training's l2: 0.0705111    valid_1's l2: 0.365988
# [1010]  training's l2: 0.0698761    valid_1's l2: 0.365899
# [1020]  training's l2: 0.0692642    valid_1's l2: 0.365833
# [1030]  training's l2: 0.0686062    valid_1's l2: 0.365768
# [1040]  training's l2: 0.0679741    valid_1's l2: 0.365732
# [1050]  training's l2: 0.0673545    valid_1's l2: 0.365699
# [1060]  training's l2: 0.0668197    valid_1's l2: 0.365679
# [1070]  training's l2: 0.0662435    valid_1's l2: 0.36567
# [1080]  training's l2: 0.0656378    valid_1's l2: 0.365624
# [1090]  training's l2: 0.065086 valid_1's l2: 0.365562
# [1100]  training's l2: 0.0644887    valid_1's l2: 0.365531
# [1110]  training's l2: 0.0639242    valid_1's l2: 0.365463
# [1120]  training's l2: 0.0634148    valid_1's l2: 0.365443
# [1130]  training's l2: 0.0628306    valid_1's l2: 0.365455
# [1140]  training's l2: 0.0623549    valid_1's l2: 0.365393
# [1150]  training's l2: 0.0619018    valid_1's l2: 0.36537
# [1160]  training's l2: 0.0614322    valid_1's l2: 0.365298
# [1170]  training's l2: 0.0609454    valid_1's l2: 0.365258
# [1180]  training's l2: 0.0604553    valid_1's l2: 0.365227
# [1190]  training's l2: 0.0600609    valid_1's l2: 0.365201
# [1200]  training's l2: 0.0596441    valid_1's l2: 0.365183
# [1210]  training's l2: 0.0592107    valid_1's l2: 0.365153
# [1220]  training's l2: 0.0587478    valid_1's l2: 0.365082
# [1230]  training's l2: 0.0583735    valid_1's l2: 0.365046
# [1240]  training's l2: 0.0579549    valid_1's l2: 0.364996
# [1250]  training's l2: 0.0575224    valid_1's l2: 0.364977
# [1260]  training's l2: 0.0571198    valid_1's l2: 0.36494
# [1270]  training's l2: 0.0567592    valid_1's l2: 0.364964
# [1280]  training's l2: 0.0563908    valid_1's l2: 0.364978
# [1290]  training's l2: 0.0559907    valid_1's l2: 0.364949
# [1300]  training's l2: 0.0556865    valid_1's l2: 0.364937
# [1310]  training's l2: 0.0553098    valid_1's l2: 0.364927
# [1320]  training's l2: 0.0549554    valid_1's l2: 0.36491
# [1330]  training's l2: 0.0545873    valid_1's l2: 0.364872
# [1340]  training's l2: 0.0542464    valid_1's l2: 0.364867
# [1350]  training's l2: 0.0539429    valid_1's l2: 0.364846
# [1360]  training's l2: 0.0536339    valid_1's l2: 0.364838
# [1370]  training's l2: 0.0533111    valid_1's l2: 0.36483
# [1380]  training's l2: 0.0529963    valid_1's l2: 0.36482
# [1390]  training's l2: 0.0527198    valid_1's l2: 0.364831
# [1400]  training's l2: 0.0524212    valid_1's l2: 0.364811
# [1410]  training's l2: 0.0521321    valid_1's l2: 0.364803
# [1420]  training's l2: 0.0518585    valid_1's l2: 0.364788
# [1430]  training's l2: 0.0515722    valid_1's l2: 0.364767
# [1440]  training's l2: 0.0512738    valid_1's l2: 0.364754
# [1450]  training's l2: 0.0510192    valid_1's l2: 0.36475
# [1460]  training's l2: 0.0507515    valid_1's l2: 0.36474
# [1470]  training's l2: 0.0504543    valid_1's l2: 0.364732
# [1480]  training's l2: 0.0502196    valid_1's l2: 0.364737
# [1490]  training's l2: 0.0499478    valid_1's l2: 0.364749
# [1500]  training's l2: 0.0497103    valid_1's l2: 0.364717
# [1510]  training's l2: 0.0494836    valid_1's l2: 0.364709
# [1520]  training's l2: 0.0492529    valid_1's l2: 0.364688
# [1530]  training's l2: 0.0490139    valid_1's l2: 0.364667
# [1540]  training's l2: 0.0487767    valid_1's l2: 0.364666
# [1550]  training's l2: 0.0485601    valid_1's l2: 0.364618
# [1560]  training's l2: 0.0483342    valid_1's l2: 0.364605
# [1570]  training's l2: 0.0481107    valid_1's l2: 0.364622
# [1580]  training's l2: 0.0478933    valid_1's l2: 0.364632
# [1590]  training's l2: 0.047684 valid_1's l2: 0.364624
# [1600]  training's l2: 0.0474853    valid_1's l2: 0.364598
# [1610]  training's l2: 0.0472917    valid_1's l2: 0.364598
# [1620]  training's l2: 0.047078 valid_1's l2: 0.364614
# [1630]  training's l2: 0.0468649    valid_1's l2: 0.364609
# [1640]  training's l2: 0.0466668    valid_1's l2: 0.364621
# [1650]  training's l2: 0.046478 valid_1's l2: 0.364641
# Early stopping, best iteration is:
# [1602]  training's l2: 0.0474554    valid_1's l2: 0.36459
# [LightGBM] [Info] Finished loading 1602 models
# 0.6235140598005153
# 0.5297002837305967
# [LightGBM] [Info] Total Bins 6647785
# [LightGBM] [Info] Number of data: 176000, number of used features: 297535
# Training until validation scores don't improve for 50 rounds.
# [10]    training's l2: 0.479421 valid_1's l2: 0.496316
# [20]    training's l2: 0.42901  valid_1's l2: 0.460972
# [30]    training's l2: 0.396478 valid_1's l2: 0.441617
# [40]    training's l2: 0.372893 valid_1's l2: 0.429945
# [50]    training's l2: 0.35422  valid_1's l2: 0.42167
# [60]    training's l2: 0.338384 valid_1's l2: 0.415334
# [70]    training's l2: 0.324608 valid_1's l2: 0.410269
# [80]    training's l2: 0.312193 valid_1's l2: 0.405934
# [90]    training's l2: 0.300906 valid_1's l2: 0.40196
# [100]   training's l2: 0.290582 valid_1's l2: 0.398827
# [110]   training's l2: 0.280902 valid_1's l2: 0.395937
# [120]   training's l2: 0.272168 valid_1's l2: 0.393267
# [130]   training's l2: 0.263975 valid_1's l2: 0.391121
# [140]   training's l2: 0.25642  valid_1's l2: 0.389092
# [150]   training's l2: 0.249596 valid_1's l2: 0.387289
# [160]   training's l2: 0.242817 valid_1's l2: 0.385637
# [170]   training's l2: 0.236845 valid_1's l2: 0.384264
# [180]   training's l2: 0.231372 valid_1's l2: 0.382976
# [190]   training's l2: 0.225899 valid_1's l2: 0.381708
# [200]   training's l2: 0.220482 valid_1's l2: 0.380643
# [210]   training's l2: 0.215557 valid_1's l2: 0.379506
# [220]   training's l2: 0.210864 valid_1's l2: 0.3785
# [230]   training's l2: 0.20612  valid_1's l2: 0.377551
# [240]   training's l2: 0.201631 valid_1's l2: 0.376677
# [250]   training's l2: 0.197393 valid_1's l2: 0.375865
# [260]   training's l2: 0.193392 valid_1's l2: 0.375189
# [270]   training's l2: 0.189548 valid_1's l2: 0.374475
# [280]   training's l2: 0.185591 valid_1's l2: 0.373644
# [290]   training's l2: 0.182257 valid_1's l2: 0.37295
# [300]   training's l2: 0.178732 valid_1's l2: 0.37218
# [310]   training's l2: 0.175344 valid_1's l2: 0.371552
# [320]   training's l2: 0.172232 valid_1's l2: 0.370958
# [330]   training's l2: 0.169236 valid_1's l2: 0.370291
# [340]   training's l2: 0.166032 valid_1's l2: 0.369874
# [350]   training's l2: 0.163238 valid_1's l2: 0.369354
# [360]   training's l2: 0.160497 valid_1's l2: 0.368922
# [370]   training's l2: 0.157724 valid_1's l2: 0.368437
# [380]   training's l2: 0.15492  valid_1's l2: 0.368026
# [390]   training's l2: 0.152399 valid_1's l2: 0.36754
# [400]   training's l2: 0.149871 valid_1's l2: 0.367124
# [410]   training's l2: 0.147334 valid_1's l2: 0.366745
# [420]   training's l2: 0.144991 valid_1's l2: 0.366302
# [430]   training's l2: 0.142589 valid_1's l2: 0.365974
# [440]   training's l2: 0.14042  valid_1's l2: 0.365652
# [450]   training's l2: 0.138145 valid_1's l2: 0.365276
# [460]   training's l2: 0.136141 valid_1's l2: 0.365056
# [470]   training's l2: 0.134031 valid_1's l2: 0.364781
# [480]   training's l2: 0.131937 valid_1's l2: 0.364554
# [490]   training's l2: 0.129991 valid_1's l2: 0.364293
# [500]   training's l2: 0.128034 valid_1's l2: 0.364003
# [510]   training's l2: 0.125993 valid_1's l2: 0.36357
# [520]   training's l2: 0.124118 valid_1's l2: 0.363403
# [530]   training's l2: 0.122241 valid_1's l2: 0.363073
# [540]   training's l2: 0.120595 valid_1's l2: 0.362798
# [550]   training's l2: 0.118714 valid_1's l2: 0.362455
# [560]   training's l2: 0.116954 valid_1's l2: 0.362253
# [570]   training's l2: 0.11545  valid_1's l2: 0.362044
# [580]   training's l2: 0.113863 valid_1's l2: 0.361759
# [590]   training's l2: 0.112331 valid_1's l2: 0.361487
# [600]   training's l2: 0.110803 valid_1's l2: 0.361294
# [610]   training's l2: 0.109314 valid_1's l2: 0.36104
# [620]   training's l2: 0.107802 valid_1's l2: 0.360826
# [630]   training's l2: 0.106245 valid_1's l2: 0.360516
# [640]   training's l2: 0.10488  valid_1's l2: 0.36035
# [650]   training's l2: 0.103521 valid_1's l2: 0.360108
# [660]   training's l2: 0.102157 valid_1's l2: 0.359949
# [670]   training's l2: 0.100853 valid_1's l2: 0.359786
# [680]   training's l2: 0.0995885    valid_1's l2: 0.359662
# [690]   training's l2: 0.0983022    valid_1's l2: 0.359484
# [700]   training's l2: 0.0970612    valid_1's l2: 0.35929
# [710]   training's l2: 0.0956898    valid_1's l2: 0.359069
# [720]   training's l2: 0.0945033    valid_1's l2: 0.358915
# [730]   training's l2: 0.0933374    valid_1's l2: 0.358668
# [740]   training's l2: 0.0922471    valid_1's l2: 0.358515
# [750]   training's l2: 0.0911128    valid_1's l2: 0.358322
# [760]   training's l2: 0.0900697    valid_1's l2: 0.358277
# [770]   training's l2: 0.0889817    valid_1's l2: 0.358159
# [780]   training's l2: 0.0878912    valid_1's l2: 0.358106
# [790]   training's l2: 0.0869304    valid_1's l2: 0.357923
# [800]   training's l2: 0.0858931    valid_1's l2: 0.357759
# [810]   training's l2: 0.0849286    valid_1's l2: 0.357583
# [820]   training's l2: 0.0839632    valid_1's l2: 0.357437
# [830]   training's l2: 0.0830681    valid_1's l2: 0.357364
# [840]   training's l2: 0.0820613    valid_1's l2: 0.357244
# [850]   training's l2: 0.0811791    valid_1's l2: 0.357148
# [860]   training's l2: 0.0803411    valid_1's l2: 0.35707
# [870]   training's l2: 0.0795059    valid_1's l2: 0.356953
# [880]   training's l2: 0.0786291    valid_1's l2: 0.356929
# [890]   training's l2: 0.0776915    valid_1's l2: 0.356826
# [900]   training's l2: 0.0767928    valid_1's l2: 0.356671
# [910]   training's l2: 0.0759947    valid_1's l2: 0.356555
# [920]   training's l2: 0.0752982    valid_1's l2: 0.356433
# [930]   training's l2: 0.0745332    valid_1's l2: 0.356324
# [940]   training's l2: 0.0738182    valid_1's l2: 0.356253
# [950]   training's l2: 0.0731099    valid_1's l2: 0.356212
# [960]   training's l2: 0.0723904    valid_1's l2: 0.356129
# [970]   training's l2: 0.0717412    valid_1's l2: 0.356082
# [980]   training's l2: 0.0710843    valid_1's l2: 0.355994
# [990]   training's l2: 0.0704669    valid_1's l2: 0.35591
# [1000]  training's l2: 0.0698972    valid_1's l2: 0.355853
# [1010]  training's l2: 0.0692685    valid_1's l2: 0.355812
# [1020]  training's l2: 0.0686436    valid_1's l2: 0.355731
# [1030]  training's l2: 0.0679925    valid_1's l2: 0.355697
# [1040]  training's l2: 0.0674284    valid_1's l2: 0.355609
# [1050]  training's l2: 0.0667755    valid_1's l2: 0.355521
# [1060]  training's l2: 0.0661636    valid_1's l2: 0.355475
# [1070]  training's l2: 0.0656233    valid_1's l2: 0.355414
# [1080]  training's l2: 0.065064 valid_1's l2: 0.355349
# [1090]  training's l2: 0.0645699    valid_1's l2: 0.355289
# [1100]  training's l2: 0.06409  valid_1's l2: 0.355249
# [1110]  training's l2: 0.0635573    valid_1's l2: 0.355224
# [1120]  training's l2: 0.0630462    valid_1's l2: 0.355179
# [1130]  training's l2: 0.0624886    valid_1's l2: 0.355139
# [1140]  training's l2: 0.0620296    valid_1's l2: 0.355063
# [1150]  training's l2: 0.0615269    valid_1's l2: 0.354989
# [1160]  training's l2: 0.0610603    valid_1's l2: 0.354955
# [1170]  training's l2: 0.0605984    valid_1's l2: 0.354907
# [1180]  training's l2: 0.0601644    valid_1's l2: 0.354862
# [1190]  training's l2: 0.0597052    valid_1's l2: 0.35488
# [1200]  training's l2: 0.0592399    valid_1's l2: 0.35479
# [1210]  training's l2: 0.0587471    valid_1's l2: 0.35472
# [1220]  training's l2: 0.0582945    valid_1's l2: 0.354695
# [1230]  training's l2: 0.0579237    valid_1's l2: 0.354673
# [1240]  training's l2: 0.0575351    valid_1's l2: 0.354641
# [1250]  training's l2: 0.0571736    valid_1's l2: 0.354599
# [1260]  training's l2: 0.0568048    valid_1's l2: 0.354547
# [1270]  training's l2: 0.0564383    valid_1's l2: 0.354508
# [1280]  training's l2: 0.056057 valid_1's l2: 0.354529
# [1290]  training's l2: 0.0556526    valid_1's l2: 0.354542
# [1300]  training's l2: 0.0552984    valid_1's l2: 0.354526
# [1310]  training's l2: 0.0549642    valid_1's l2: 0.3545
# [1320]  training's l2: 0.0546028    valid_1's l2: 0.354467
# [1330]  training's l2: 0.0542269    valid_1's l2: 0.35444
# [1340]  training's l2: 0.0538775    valid_1's l2: 0.354387
# [1350]  training's l2: 0.0535397    valid_1's l2: 0.354373
# [1360]  training's l2: 0.0531811    valid_1's l2: 0.354332
# [1370]  training's l2: 0.0528751    valid_1's l2: 0.354321
# [1380]  training's l2: 0.0525469    valid_1's l2: 0.354298
# [1390]  training's l2: 0.0522321    valid_1's l2: 0.354277
# [1400]  training's l2: 0.0519022    valid_1's l2: 0.354299
# [1410]  training's l2: 0.0516021    valid_1's l2: 0.354276
# [1420]  training's l2: 0.0513471    valid_1's l2: 0.354264
# [1430]  training's l2: 0.0510511    valid_1's l2: 0.354234
# [1440]  training's l2: 0.0507794    valid_1's l2: 0.354232
# [1450]  training's l2: 0.0504892    valid_1's l2: 0.354228
# [1460]  training's l2: 0.0502251    valid_1's l2: 0.354212
# [1470]  training's l2: 0.0499664    valid_1's l2: 0.354188
# [1480]  training's l2: 0.0497146    valid_1's l2: 0.354175
# [1490]  training's l2: 0.049447 valid_1's l2: 0.354158
# [1500]  training's l2: 0.049178 valid_1's l2: 0.354129
# [1510]  training's l2: 0.0489688    valid_1's l2: 0.354087
# [1520]  training's l2: 0.0487459    valid_1's l2: 0.354103
# [1530]  training's l2: 0.0485014    valid_1's l2: 0.354099
# [1540]  training's l2: 0.0482597    valid_1's l2: 0.354088
# [1550]  training's l2: 0.0480361    valid_1's l2: 0.354069
# [1560]  training's l2: 0.0477961    valid_1's l2: 0.354054
# [1570]  training's l2: 0.047572 valid_1's l2: 0.354053
# [1580]  training's l2: 0.0473488    valid_1's l2: 0.35404
# [1590]  training's l2: 0.0471281    valid_1's l2: 0.35402
# [1600]  training's l2: 0.0468939    valid_1's l2: 0.354018
# [1610]  training's l2: 0.0466649    valid_1's l2: 0.354022
# [1620]  training's l2: 0.0464747    valid_1's l2: 0.354001
# [1630]  training's l2: 0.0462788    valid_1's l2: 0.353986
# [1640]  training's l2: 0.0460911    valid_1's l2: 0.353984
# [1650]  training's l2: 0.0459267    valid_1's l2: 0.35396
# [1660]  training's l2: 0.0457241    valid_1's l2: 0.35396
# [1670]  training's l2: 0.0455287    valid_1's l2: 0.35395
# [1680]  training's l2: 0.0453532    valid_1's l2: 0.353961
# [1690]  training's l2: 0.0451768    valid_1's l2: 0.353985
# [1700]  training's l2: 0.044993 valid_1's l2: 0.354
# [1710]  training's l2: 0.0448314    valid_1's l2: 0.354007
# [1720]  training's l2: 0.0446673    valid_1's l2: 0.353985
# Early stopping, best iteration is:
# [1675]  training's l2: 0.0454392    valid_1's l2: 0.353942
# [LightGBM] [Info] Finished loading 1675 models
# 0.6269866464792492
# 0.5310395589237853
# [LightGBM] [Info] Total Bins 6678911
# [LightGBM] [Info] Number of data: 176000, number of used features: 300111
# Training until validation scores don't improve for 50 rounds.
# [10]    training's l2: 0.478063 valid_1's l2: 0.502349
# [20]    training's l2: 0.427869 valid_1's l2: 0.466005
# [30]    training's l2: 0.395505 valid_1's l2: 0.445915
# [40]    training's l2: 0.372217 valid_1's l2: 0.433675
# [50]    training's l2: 0.353622 valid_1's l2: 0.425335
# [60]    training's l2: 0.337919 valid_1's l2: 0.418933
# [70]    training's l2: 0.32394  valid_1's l2: 0.414005
# [80]    training's l2: 0.311409 valid_1's l2: 0.409527
# [90]    training's l2: 0.300109 valid_1's l2: 0.405811
# [100]   training's l2: 0.289777 valid_1's l2: 0.402767
# [110]   training's l2: 0.280299 valid_1's l2: 0.399949
# [120]   training's l2: 0.271529 valid_1's l2: 0.397504
# [130]   training's l2: 0.26352  valid_1's l2: 0.395238
# [140]   training's l2: 0.255929 valid_1's l2: 0.393345
# [150]   training's l2: 0.248919 valid_1's l2: 0.391828
# [160]   training's l2: 0.24228  valid_1's l2: 0.390258
# [170]   training's l2: 0.236196 valid_1's l2: 0.388916
# [180]   training's l2: 0.230636 valid_1's l2: 0.387797
# [190]   training's l2: 0.224993 valid_1's l2: 0.386568
# [200]   training's l2: 0.219714 valid_1's l2: 0.385522
# [210]   training's l2: 0.21468  valid_1's l2: 0.384531
# [220]   training's l2: 0.209893 valid_1's l2: 0.383465
# [230]   training's l2: 0.205298 valid_1's l2: 0.382495
# [240]   training's l2: 0.201116 valid_1's l2: 0.381757
# [250]   training's l2: 0.197164 valid_1's l2: 0.380964
# [260]   training's l2: 0.192892 valid_1's l2: 0.38029
# [270]   training's l2: 0.188918 valid_1's l2: 0.379562
# [280]   training's l2: 0.18523  valid_1's l2: 0.378868
# [290]   training's l2: 0.181685 valid_1's l2: 0.378197
# [300]   training's l2: 0.178268 valid_1's l2: 0.377723
# [310]   training's l2: 0.174859 valid_1's l2: 0.377202
# [320]   training's l2: 0.17168  valid_1's l2: 0.376713
# [330]   training's l2: 0.168653 valid_1's l2: 0.376174
# [340]   training's l2: 0.165721 valid_1's l2: 0.375593
# [350]   training's l2: 0.162726 valid_1's l2: 0.375137
# [360]   training's l2: 0.159789 valid_1's l2: 0.374708
# [370]   training's l2: 0.15705  valid_1's l2: 0.374311
# [380]   training's l2: 0.154364 valid_1's l2: 0.373895
# [390]   training's l2: 0.15183  valid_1's l2: 0.373513
# [400]   training's l2: 0.149442 valid_1's l2: 0.37307
# [410]   training's l2: 0.146865 valid_1's l2: 0.372754
# [420]   training's l2: 0.144564 valid_1's l2: 0.372345
# [430]   training's l2: 0.142276 valid_1's l2: 0.372001
# [440]   training's l2: 0.140196 valid_1's l2: 0.371648
# [450]   training's l2: 0.13791  valid_1's l2: 0.371358
# [460]   training's l2: 0.135909 valid_1's l2: 0.371065
# [470]   training's l2: 0.133909 valid_1's l2: 0.370777
# [480]   training's l2: 0.131893 valid_1's l2: 0.370472
# [490]   training's l2: 0.129975 valid_1's l2: 0.370255
# [500]   training's l2: 0.128166 valid_1's l2: 0.369937
# [510]   training's l2: 0.126326 valid_1's l2: 0.369629
# [520]   training's l2: 0.124442 valid_1's l2: 0.369385
# [530]   training's l2: 0.122723 valid_1's l2: 0.369097
# [540]   training's l2: 0.120873 valid_1's l2: 0.368783
# [550]   training's l2: 0.119059 valid_1's l2: 0.368589
# [560]   training's l2: 0.117326 valid_1's l2: 0.368406
# [570]   training's l2: 0.115691 valid_1's l2: 0.368176
# [580]   training's l2: 0.11394  valid_1's l2: 0.367913
# [590]   training's l2: 0.112482 valid_1's l2: 0.367713
# [600]   training's l2: 0.110872 valid_1's l2: 0.367532
# [610]   training's l2: 0.109338 valid_1's l2: 0.367429
# [620]   training's l2: 0.107841 valid_1's l2: 0.367264
# [630]   training's l2: 0.106414 valid_1's l2: 0.367067
# [640]   training's l2: 0.105146 valid_1's l2: 0.366882
# [650]   training's l2: 0.103901 valid_1's l2: 0.366686
# [660]   training's l2: 0.102456 valid_1's l2: 0.366548
# [670]   training's l2: 0.101206 valid_1's l2: 0.366375
# [680]   training's l2: 0.10003  valid_1's l2: 0.366243
# [690]   training's l2: 0.098893 valid_1's l2: 0.366121
# [700]   training's l2: 0.0975882    valid_1's l2: 0.366019
# [710]   training's l2: 0.0964479    valid_1's l2: 0.365911
# [720]   training's l2: 0.0952592    valid_1's l2: 0.365819
# [730]   training's l2: 0.0940534    valid_1's l2: 0.365594
# [740]   training's l2: 0.0928993    valid_1's l2: 0.365439
# [750]   training's l2: 0.0918323    valid_1's l2: 0.365317
# [760]   training's l2: 0.0906989    valid_1's l2: 0.365271
# [770]   training's l2: 0.0896862    valid_1's l2: 0.365112
# [780]   training's l2: 0.0886278    valid_1's l2: 0.364994
# [790]   training's l2: 0.0876974    valid_1's l2: 0.364879
# [800]   training's l2: 0.0866325    valid_1's l2: 0.364761
# [810]   training's l2: 0.0856825    valid_1's l2: 0.36465
# [820]   training's l2: 0.0846731    valid_1's l2: 0.364577
# [830]   training's l2: 0.0836916    valid_1's l2: 0.364466
# [840]   training's l2: 0.0828327    valid_1's l2: 0.364365
# [850]   training's l2: 0.0819625    valid_1's l2: 0.36424
# [860]   training's l2: 0.0810534    valid_1's l2: 0.364163
# [870]   training's l2: 0.0802003    valid_1's l2: 0.364079
# [880]   training's l2: 0.0793608    valid_1's l2: 0.363986
# [890]   training's l2: 0.0785968    valid_1's l2: 0.363898
# [900]   training's l2: 0.0778039    valid_1's l2: 0.363868
# [910]   training's l2: 0.0770423    valid_1's l2: 0.36381
# [920]   training's l2: 0.0763346    valid_1's l2: 0.363691
# [930]   training's l2: 0.075627 valid_1's l2: 0.363599
# [940]   training's l2: 0.0748686    valid_1's l2: 0.363531
# [950]   training's l2: 0.0741133    valid_1's l2: 0.363474
# [960]   training's l2: 0.0733781    valid_1's l2: 0.363393
# [970]   training's l2: 0.072655 valid_1's l2: 0.363348
# [980]   training's l2: 0.0719219    valid_1's l2: 0.363273
# [990]   training's l2: 0.071229 valid_1's l2: 0.363202
# [1000]  training's l2: 0.0705697    valid_1's l2: 0.3631
# [1010]  training's l2: 0.0699047    valid_1's l2: 0.363056
# [1020]  training's l2: 0.0692256    valid_1's l2: 0.362993
# [1030]  training's l2: 0.0686068    valid_1's l2: 0.362958
# [1040]  training's l2: 0.0679321    valid_1's l2: 0.362822
# [1050]  training's l2: 0.0673532    valid_1's l2: 0.362768
# [1060]  training's l2: 0.0667219    valid_1's l2: 0.362694
# [1070]  training's l2: 0.0661564    valid_1's l2: 0.362607
# [1080]  training's l2: 0.0656364    valid_1's l2: 0.36258
# [1090]  training's l2: 0.0651985    valid_1's l2: 0.362549
# [1100]  training's l2: 0.0646226    valid_1's l2: 0.362507
# [1110]  training's l2: 0.0641068    valid_1's l2: 0.362465
# [1120]  training's l2: 0.063544 valid_1's l2: 0.362464
# [1130]  training's l2: 0.062961 valid_1's l2: 0.362445
# [1140]  training's l2: 0.06244  valid_1's l2: 0.362359
# [1150]  training's l2: 0.0619567    valid_1's l2: 0.362346
# [1160]  training's l2: 0.0614849    valid_1's l2: 0.362283
# [1170]  training's l2: 0.0610243    valid_1's l2: 0.362254
# [1180]  training's l2: 0.0606116    valid_1's l2: 0.362234
# [1190]  training's l2: 0.0601486    valid_1's l2: 0.362224
# [1200]  training's l2: 0.0597181    valid_1's l2: 0.362182
# [1210]  training's l2: 0.0592891    valid_1's l2: 0.362175
# [1220]  training's l2: 0.0588923    valid_1's l2: 0.362127
# [1230]  training's l2: 0.0585282    valid_1's l2: 0.362117
# [1240]  training's l2: 0.0581275    valid_1's l2: 0.362083
# [1250]  training's l2: 0.0577443    valid_1's l2: 0.362065
# [1260]  training's l2: 0.0573126    valid_1's l2: 0.362045
# [1270]  training's l2: 0.056913 valid_1's l2: 0.362021
# [1280]  training's l2: 0.0565184    valid_1's l2: 0.361971
# [1290]  training's l2: 0.0560824    valid_1's l2: 0.361942
# [1300]  training's l2: 0.0557161    valid_1's l2: 0.361924
# [1310]  training's l2: 0.0553656    valid_1's l2: 0.361921
# [1320]  training's l2: 0.0550336    valid_1's l2: 0.36188
# [1330]  training's l2: 0.0546501    valid_1's l2: 0.361865
# [1340]  training's l2: 0.0543112    valid_1's l2: 0.36184
# [1350]  training's l2: 0.0539146    valid_1's l2: 0.361854
# [1360]  training's l2: 0.0536021    valid_1's l2: 0.361849
# [1370]  training's l2: 0.0532782    valid_1's l2: 0.361807
# [1380]  training's l2: 0.0529476    valid_1's l2: 0.361759
# [1390]  training's l2: 0.0526172    valid_1's l2: 0.361731
# [1400]  training's l2: 0.0522959    valid_1's l2: 0.361677
# [1410]  training's l2: 0.0519921    valid_1's l2: 0.361682
# [1420]  training's l2: 0.0517251    valid_1's l2: 0.361651
# [1430]  training's l2: 0.051411 valid_1's l2: 0.361652
# [1440]  training's l2: 0.0511186    valid_1's l2: 0.361653
# [1450]  training's l2: 0.050814 valid_1's l2: 0.361652
# [1460]  training's l2: 0.0505228    valid_1's l2: 0.361645
# [1470]  training's l2: 0.0502425    valid_1's l2: 0.361616
# [1480]  training's l2: 0.049979 valid_1's l2: 0.361584
# [1490]  training's l2: 0.0497332    valid_1's l2: 0.361598
# [1500]  training's l2: 0.0495102    valid_1's l2: 0.361594
# [1510]  training's l2: 0.0492464    valid_1's l2: 0.36159
# [1520]  training's l2: 0.0490125    valid_1's l2: 0.361548
# [1530]  training's l2: 0.0487863    valid_1's l2: 0.361537
# [1540]  training's l2: 0.0485751    valid_1's l2: 0.361531
# [1550]  training's l2: 0.0483511    valid_1's l2: 0.361555
# [1560]  training's l2: 0.0481132    valid_1's l2: 0.361561
# [1570]  training's l2: 0.0478741    valid_1's l2: 0.361548
# [1580]  training's l2: 0.0476339    valid_1's l2: 0.361518
# [1590]  training's l2: 0.0473682    valid_1's l2: 0.361525
# [1600]  training's l2: 0.0471595    valid_1's l2: 0.361514
# [1610]  training's l2: 0.0469378    valid_1's l2: 0.361487
# [1620]  training's l2: 0.0467321    valid_1's l2: 0.361498
# [1630]  training's l2: 0.0465305    valid_1's l2: 0.361527
# [1640]  training's l2: 0.0463454    valid_1's l2: 0.361528
# [1650]  training's l2: 0.0461356    valid_1's l2: 0.361534
# [1660]  training's l2: 0.0459488    valid_1's l2: 0.361529
# Early stopping, best iteration is:
# [1611]  training's l2: 0.0469155    valid_1's l2: 0.361478
# [LightGBM] [Info] Finished loading 1611 models
# 0.6245196639415834
# 0.531606900382893
# [LightGBM] [Info] Total Bins 6660387
# [LightGBM] [Info] Number of data: 176000, number of used features: 298491
# Training until validation scores don't improve for 50 rounds.
# [10]    training's l2: 0.481484 valid_1's l2: 0.486451
# [20]    training's l2: 0.430633 valid_1's l2: 0.452052
# [30]    training's l2: 0.397733 valid_1's l2: 0.433423
# [40]    training's l2: 0.374269 valid_1's l2: 0.421938
# [50]    training's l2: 0.355369 valid_1's l2: 0.414368
# [60]    training's l2: 0.339443 valid_1's l2: 0.408647
# [70]    training's l2: 0.325348 valid_1's l2: 0.4034
# [80]    training's l2: 0.312787 valid_1's l2: 0.399253
# [90]    training's l2: 0.301275 valid_1's l2: 0.395814
# [100]   training's l2: 0.29093  valid_1's l2: 0.393039
# [110]   training's l2: 0.281292 valid_1's l2: 0.390389
# [120]   training's l2: 0.272389 valid_1's l2: 0.388186
# [130]   training's l2: 0.264154 valid_1's l2: 0.386115
# [140]   training's l2: 0.256768 valid_1's l2: 0.384218
# [150]   training's l2: 0.249864 valid_1's l2: 0.382675
# [160]   training's l2: 0.243264 valid_1's l2: 0.381367
# [170]   training's l2: 0.237117 valid_1's l2: 0.380156
# [180]   training's l2: 0.231323 valid_1's l2: 0.379061
# [190]   training's l2: 0.225909 valid_1's l2: 0.377925
# [200]   training's l2: 0.220627 valid_1's l2: 0.377029
# [210]   training's l2: 0.215585 valid_1's l2: 0.376064
# [220]   training's l2: 0.210903 valid_1's l2: 0.375171
# [230]   training's l2: 0.206443 valid_1's l2: 0.374247
# [240]   training's l2: 0.202298 valid_1's l2: 0.373526
# [250]   training's l2: 0.197993 valid_1's l2: 0.372869
# [260]   training's l2: 0.193909 valid_1's l2: 0.372314
# [270]   training's l2: 0.19006  valid_1's l2: 0.371692
# [280]   training's l2: 0.18624  valid_1's l2: 0.370977
# [290]   training's l2: 0.182668 valid_1's l2: 0.370255
# [300]   training's l2: 0.179212 valid_1's l2: 0.369655
# [310]   training's l2: 0.175978 valid_1's l2: 0.369138
# [320]   training's l2: 0.172578 valid_1's l2: 0.368554
# [330]   training's l2: 0.169429 valid_1's l2: 0.368054
# [340]   training's l2: 0.166488 valid_1's l2: 0.367697
# [350]   training's l2: 0.163639 valid_1's l2: 0.367232
# [360]   training's l2: 0.160883 valid_1's l2: 0.366814
# [370]   training's l2: 0.158173 valid_1's l2: 0.366419
# [380]   training's l2: 0.155473 valid_1's l2: 0.365959
# [390]   training's l2: 0.152831 valid_1's l2: 0.365498
# [400]   training's l2: 0.150357 valid_1's l2: 0.365243
# [410]   training's l2: 0.1479   valid_1's l2: 0.364809
# [420]   training's l2: 0.145547 valid_1's l2: 0.364413
# [430]   training's l2: 0.143009 valid_1's l2: 0.364134
# [440]   training's l2: 0.140704 valid_1's l2: 0.363808
# [450]   training's l2: 0.138506 valid_1's l2: 0.363514
# [460]   training's l2: 0.13627  valid_1's l2: 0.363256
# [470]   training's l2: 0.134021 valid_1's l2: 0.362962
# [480]   training's l2: 0.13202  valid_1's l2: 0.362735
# [490]   training's l2: 0.129996 valid_1's l2: 0.362368
# [500]   training's l2: 0.127932 valid_1's l2: 0.362073
# [510]   training's l2: 0.1259   valid_1's l2: 0.361831
# [520]   training's l2: 0.124025 valid_1's l2: 0.361618
# [530]   training's l2: 0.122297 valid_1's l2: 0.361405
# [540]   training's l2: 0.120578 valid_1's l2: 0.36124
# [550]   training's l2: 0.11873  valid_1's l2: 0.361067
# [560]   training's l2: 0.117112 valid_1's l2: 0.360954
# [570]   training's l2: 0.115506 valid_1's l2: 0.360675
# [580]   training's l2: 0.1139   valid_1's l2: 0.360436
# [590]   training's l2: 0.112304 valid_1's l2: 0.360273
# [600]   training's l2: 0.110692 valid_1's l2: 0.360053
# [610]   training's l2: 0.109094 valid_1's l2: 0.359897
# [620]   training's l2: 0.107621 valid_1's l2: 0.359747
# [630]   training's l2: 0.106297 valid_1's l2: 0.35961
# [640]   training's l2: 0.104903 valid_1's l2: 0.359392
# [650]   training's l2: 0.103552 valid_1's l2: 0.3592
# [660]   training's l2: 0.102231 valid_1's l2: 0.358962
# [670]   training's l2: 0.100915 valid_1's l2: 0.358798
# [680]   training's l2: 0.0995181    valid_1's l2: 0.358603
# [690]   training's l2: 0.098317 valid_1's l2: 0.358435
# [700]   training's l2: 0.0970531    valid_1's l2: 0.358271
# [710]   training's l2: 0.0958372    valid_1's l2: 0.358194
# [720]   training's l2: 0.0946266    valid_1's l2: 0.358084
# [730]   training's l2: 0.0934384    valid_1's l2: 0.357949
# [740]   training's l2: 0.0922832    valid_1's l2: 0.357852
# [750]   training's l2: 0.0910544    valid_1's l2: 0.357778
# [760]   training's l2: 0.0900585    valid_1's l2: 0.357676
# [770]   training's l2: 0.0891037    valid_1's l2: 0.35757
# [780]   training's l2: 0.0880535    valid_1's l2: 0.357432
# [790]   training's l2: 0.0870451    valid_1's l2: 0.357351
# [800]   training's l2: 0.0860725    valid_1's l2: 0.357279
# [810]   training's l2: 0.0852084    valid_1's l2: 0.357193
# [820]   training's l2: 0.0842278    valid_1's l2: 0.357052
# [830]   training's l2: 0.0832272    valid_1's l2: 0.356963
# [840]   training's l2: 0.0821857    valid_1's l2: 0.35681
# [850]   training's l2: 0.0812712    valid_1's l2: 0.356717
# [860]   training's l2: 0.0804171    valid_1's l2: 0.356591
# [870]   training's l2: 0.0796119    valid_1's l2: 0.356469
# [880]   training's l2: 0.078829 valid_1's l2: 0.356343
# [890]   training's l2: 0.0779634    valid_1's l2: 0.356243
# [900]   training's l2: 0.0771527    valid_1's l2: 0.356133
# [910]   training's l2: 0.0764151    valid_1's l2: 0.356053
# [920]   training's l2: 0.0755913    valid_1's l2: 0.355989
# [930]   training's l2: 0.0747694    valid_1's l2: 0.355876
# [940]   training's l2: 0.0740119    valid_1's l2: 0.355803
# [950]   training's l2: 0.0732893    valid_1's l2: 0.355736
# [960]   training's l2: 0.0725836    valid_1's l2: 0.355645
# [970]   training's l2: 0.0718724    valid_1's l2: 0.355568
# [980]   training's l2: 0.0712069    valid_1's l2: 0.355537
# [990]   training's l2: 0.0706038    valid_1's l2: 0.355495
# [1000]  training's l2: 0.0699104    valid_1's l2: 0.355436
# [1010]  training's l2: 0.0692288    valid_1's l2: 0.355337
# [1020]  training's l2: 0.0686097    valid_1's l2: 0.355274
# [1030]  training's l2: 0.0680417    valid_1's l2: 0.355215
# [1040]  training's l2: 0.0674064    valid_1's l2: 0.355153
# [1050]  training's l2: 0.0667932    valid_1's l2: 0.355046
# [1060]  training's l2: 0.0662134    valid_1's l2: 0.355025
# [1070]  training's l2: 0.0656972    valid_1's l2: 0.354925
# [1080]  training's l2: 0.065124 valid_1's l2: 0.354831
# [1090]  training's l2: 0.0646307    valid_1's l2: 0.354741
# [1100]  training's l2: 0.0641139    valid_1's l2: 0.354682
# [1110]  training's l2: 0.0635118    valid_1's l2: 0.354647
# [1120]  training's l2: 0.0629534    valid_1's l2: 0.354583
# [1130]  training's l2: 0.0624699    valid_1's l2: 0.354512
# [1140]  training's l2: 0.0620341    valid_1's l2: 0.354482
# [1150]  training's l2: 0.0615068    valid_1's l2: 0.354458
# [1160]  training's l2: 0.0610801    valid_1's l2: 0.354438
# [1170]  training's l2: 0.0605797    valid_1's l2: 0.354382
# [1180]  training's l2: 0.0601014    valid_1's l2: 0.354381
# [1190]  training's l2: 0.0596412    valid_1's l2: 0.354388
# [1200]  training's l2: 0.0591665    valid_1's l2: 0.354348
# [1210]  training's l2: 0.058666 valid_1's l2: 0.354332
# [1220]  training's l2: 0.0582114    valid_1's l2: 0.35427
# [1230]  training's l2: 0.0578289    valid_1's l2: 0.354237
# [1240]  training's l2: 0.0574048    valid_1's l2: 0.354213
# [1250]  training's l2: 0.0570079    valid_1's l2: 0.354211
# [1260]  training's l2: 0.0566003    valid_1's l2: 0.35421
# [1270]  training's l2: 0.056201 valid_1's l2: 0.354202
# [1280]  training's l2: 0.0558509    valid_1's l2: 0.354126
# [1290]  training's l2: 0.0554665    valid_1's l2: 0.354125
# [1300]  training's l2: 0.0551083    valid_1's l2: 0.354103
# [1310]  training's l2: 0.0547628    valid_1's l2: 0.354038
# [1320]  training's l2: 0.054422 valid_1's l2: 0.354043
# [1330]  training's l2: 0.0540849    valid_1's l2: 0.354035
# [1340]  training's l2: 0.0536987    valid_1's l2: 0.353999
# [1350]  training's l2: 0.0533583    valid_1's l2: 0.353992
# [1360]  training's l2: 0.0530405    valid_1's l2: 0.35393
# [1370]  training's l2: 0.0527385    valid_1's l2: 0.35392
# [1380]  training's l2: 0.0524267    valid_1's l2: 0.35389
# [1390]  training's l2: 0.0520822    valid_1's l2: 0.353837
# [1400]  training's l2: 0.0517895    valid_1's l2: 0.353798
# [1410]  training's l2: 0.0515082    valid_1's l2: 0.35379
# [1420]  training's l2: 0.0512047    valid_1's l2: 0.353796
# [1430]  training's l2: 0.0508893    valid_1's l2: 0.353788
# [1440]  training's l2: 0.0506129    valid_1's l2: 0.353813
# [1450]  training's l2: 0.0503272    valid_1's l2: 0.353771
# [1460]  training's l2: 0.0500574    valid_1's l2: 0.353747
# [1470]  training's l2: 0.0498191    valid_1's l2: 0.353741
# [1480]  training's l2: 0.04957  valid_1's l2: 0.353732
# [1490]  training's l2: 0.0492932    valid_1's l2: 0.353708
# [1500]  training's l2: 0.0490473    valid_1's l2: 0.35369
# [1510]  training's l2: 0.0487843    valid_1's l2: 0.353646
# [1520]  training's l2: 0.0485575    valid_1's l2: 0.353626
# [1530]  training's l2: 0.0483471    valid_1's l2: 0.353599
# [1540]  training's l2: 0.0480994    valid_1's l2: 0.35358
# [1550]  training's l2: 0.0479021    valid_1's l2: 0.353568
# [1560]  training's l2: 0.047687 valid_1's l2: 0.353537
# [1570]  training's l2: 0.0474309    valid_1's l2: 0.353547
# [1580]  training's l2: 0.0472191    valid_1's l2: 0.353522
# [1590]  training's l2: 0.0469991    valid_1's l2: 0.353523
# [1600]  training's l2: 0.0467719    valid_1's l2: 0.353495
# [1610]  training's l2: 0.0465459    valid_1's l2: 0.353476
# [1620]  training's l2: 0.0463089    valid_1's l2: 0.353446
# [1630]  training's l2: 0.0461018    valid_1's l2: 0.353422
# [1640]  training's l2: 0.045898 valid_1's l2: 0.353395
# [1650]  training's l2: 0.04569  valid_1's l2: 0.353398
# [1660]  training's l2: 0.0454866    valid_1's l2: 0.3534
# [1670]  training's l2: 0.0452851    valid_1's l2: 0.353401
# [1680]  training's l2: 0.0450959    valid_1's l2: 0.353389
# [1690]  training's l2: 0.0448912    valid_1's l2: 0.35338
# [1700]  training's l2: 0.0447227    valid_1's l2: 0.353388
# [1710]  training's l2: 0.0445314    valid_1's l2: 0.353362
# [1720]  training's l2: 0.04435  valid_1's l2: 0.353349
# [1730]  training's l2: 0.0441746    valid_1's l2: 0.353329
# [1740]  training's l2: 0.0439818    valid_1's l2: 0.353342
# [1750]  training's l2: 0.043825 valid_1's l2: 0.353316
# [1760]  training's l2: 0.0436399    valid_1's l2: 0.353333
# [1770]  training's l2: 0.0434669    valid_1's l2: 0.353327
# [1780]  training's l2: 0.0432987    valid_1's l2: 0.353332
# [1790]  training's l2: 0.0431438    valid_1's l2: 0.35335
# Early stopping, best iteration is:
# [1749]  training's l2: 0.0438375    valid_1's l2: 0.353315
# [LightGBM] [Info] Finished loading 1749 models
# 0.6271939563179832
# 0.5313265534809998
# [Finished in 47075.3s]