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
    VECTOR_DIR_4 = './word2vec/lg_data_model_comment_256dim'
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
    train_data[['Id','lgb_ci_feat']].to_csv('../stack_cache/5_fold_lgb_ci_feat.csv',index=False)
    test_data[['Id','lgb_ci_prob']].to_csv('../stack_cache/5_fold_lgb_ci_prob.csv',index=False)

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
    train_data[['Id','lgb_zi_feat']].to_csv('../stack_cache/5_fold_lgb_zi_feat.csv',index=False)
    test_data[['Id','lgb_zi_prob']].to_csv('../stack_cache/5_fold_lgb_zi_prob.csv',index=False)


    # print("lgb_all training......")
    # result = train_lgb(y,X,test_hh)
    # res = pd.DataFrame(test_data['Id'])
    # res['prob'] = result    
    # merge = pd.DataFrame(test_id)
    # merge = pd.merge(merge,res,how='left',on='Id')
    # merge = pd.merge(merge,same[['Id','Score']],how='left',on='Id')
    # merge = pd.merge(merge,unkown[['Id','Score']],how='left',on='Id',suffixes=['_same','_unkown'])
    # merge['final_score'] =  merge.prob.where(merge.prob.notnull(),merge.Score_same.where(merge.Score_same.notnull(),merge.Score_unkown))
    # merge['prob1'] = merge['final_score'].apply(lambda x:5 if x>=4.7 else x)
    # merge['prob1'] = merge['prob1'].apply(lambda x:1 if x<=1 else x)  
    # merge[['Id', 'final_score']].to_csv('../result/lg_data_all_mergetfidf_5w2v_notchange_5516.csv',index=False,header=False)
    # merge[['Id', 'prob1']].to_csv('../result/lg_data_all_mergetfidf_5w2v.csv',index=False,header=False) # 5516

