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
from sklearn.naive_bayes import MultinomialNB  

from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import text, sequence
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,HashingVectorizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical,plot_model
from keras.models import Sequential,Model
from keras.layers import Dense, Input, Flatten, Dropout,Conv1D, MaxPooling1D,Embedding,Activation,LSTM,BatchNormalization
from keras.layers.recurrent import LSTM, GRU 
from keras.layers import SpatialDropout1D, concatenate,Concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D,GlobalMaxPool1D
from keras.callbacks import Callback
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.models import load_model

VECTOR_DIR = './word2vec/word2vec_wx'
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 256

import random
random.seed(2018)
np.random.seed(2018)

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

def load_w2v(VECTOR_DIR,EMBEDDING_DIM):
    if((VECTOR_DIR[-4:]=='.txt')|(VECTOR_DIR[-4:]=='.vec')):
        model1 = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_DIR)
    elif(VECTOR_DIR[-4:]=='.bin'):
        model1 = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_DIR, binary=True)
    else:
        model1=Word2Vec.load(VECTOR_DIR)

    return model1


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


def data_and_shuffle(data):
    p1=[]
    for i in data['cutted_Dis']:
        i=str(i)
        u=i.split(' ')
        s=[]
        for j in range(len(u)):
            s.append(j)
        random.shuffle(s)
        h=' '
        for j in s:
            h=h+' '+u[j]
            
        p1.append(h)
    data1=pd.DataFrame()
    data1['cutted_Dis']=p1
    data1['Score']=data['Score']
    data_shuffle = pd.concat([data,data1])
    return data_shuffle



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

def tfidf_process_ci_feats(data,train_data,test_data,num_feats):

    y = train_data['Score']  
    tf1 = TfidfVectorizer(ngram_range=(1,2),token_pattern='\w+',max_features=num_feats,analyzer='word')
    tf1.fit(data['cutted_Dis'])
    data1=tf1.transform(train_data['cutted_Dis'])
    test1 = tf1.transform(test_data['cutted_Dis'])
    print(data1.shape)  
    return data1,test1,y

def tfidf_process_zi_feats(data,train_data,test_data,num_feats):
    data = cut_zi(data)
    train_data = cut_zi(train_data)
    test_data = cut_zi(test_data)
    y = train_data['Score']  
    tf1 = TfidfVectorizer(ngram_range=(1,6),analyzer='char',max_features=num_feats)
    tf1.fit(data['cut_zi'])
    data1=tf1.transform(train_data['cut_zi'])
    test1 = tf1.transform(test_data['cut_zi'])
    print(data1.shape)  
    return data1,test1,y


def get_mlp_model(num_feats):
    inp = Input(shape=(num_feats,))
    x = Dense(1024,activation='relu')(inp)
    x= Dropout(0.2)(x)
    x = Dense(512,activation='relu')(inp)
    x= Dropout(0.2)(x)
    out=Dense(1,name='output')(x)
    model = Model(inputs=[inp],outputs=[out])
    model.compile(loss='mse', optimizer='rmsprop')
    return model



def n_folds_train_mlp(n_folds,y,X,test_hh,num_feats,model_name):
    # skf = list(StratifiedKFold(y, n_folds,random_state=2018,shuffle=True))
    skf = KFold(X.shape[0], n_folds,random_state=2018,shuffle=True)

    daset_blend_train = np.zeros((X.shape[0], 1))
    daset_blend_test = np.zeros((test_hh.shape[0], 1))
    daset_blend_test_0 = np.zeros((test_hh.shape[0], len(skf)))

    for i ,(train,test) in enumerate(skf):
            X_train, y_train, X_test, y_test = X[train], y.values[train], X[test], y.values[test]
            earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=2)
            model = get_mlp_model(num_feats)
            model.fit([X_train], y_train,
                        validation_data=([X_test], y_test),
                        batch_size=64, epochs=15, verbose=1,
                        callbacks=[earlystopping,
                                   keras.callbacks.ModelCheckpoint(model_name + str(i)+'.h5', monitor='val_loss', verbose=0,
                                                             save_best_only=True, mode='auto')])
            model = load_model(model_name + str(i)+'.h5')
            y_sub = model.predict(X_test)
            print(1 /(1 + mean_squared_error(y_test,y_sub)**0.5))
            print(xx_mse_s(y_test, y_sub))
            daset_blend_train[test]=y_sub
            daset_blend_test_0[:,i] = list(model.predict(test_hh))
    daset_blend_test[:, 0] = daset_blend_test_0.mean(1)

    return daset_blend_train,daset_blend_test

def n_folds_train_bayes(n_folds,y,X,test_hh):
    # skf = list(StratifiedKFold(y, n_folds,random_state=2018,shuffle=True))
    skf = KFold(X.shape[0], n_folds,random_state=2018,shuffle=True)

    daset_blend_train = np.zeros((X.shape[0], 1))
    daset_blend_test = np.zeros((test_hh.shape[0], 1))
    daset_blend_test_0 = np.zeros((test_hh.shape[0], len(skf)))

    model = MultinomialNB(alpha = 0.01)
    for i ,(train,test) in enumerate(skf):
            X_train, y_train, X_test, y_test = X[train], y.values[train], X[test], y.values[test]
            model.fit(X_train, y_train)
            y_sub = get_label(model.predict_proba(X_test))
            print(1 /(1 + mean_squared_error(y_test,y_sub)**0.5))
            print(xx_mse_s(y_test, y_sub))
            daset_blend_train[test,0]=y_sub
            daset_blend_test_0[:,i]=get_label(model.predict_proba(test_hh))
    daset_blend_test[:, 0] = daset_blend_test_0.mean(1)

    return daset_blend_train,daset_blend_test

def tfidf_process_ci_feats_keras(data,train_data,test_data,num_feats):

    y = train_data['Score']  
    tokenizer = Tokenizer(num_words=num_feats)
    tokenizer.fit_on_texts(data['cutted_Dis'])
    sequences = tokenizer.texts_to_sequences(train_data['cutted_Dis'])
    X= tokenizer.sequences_to_matrix(sequences, mode='tfidf')
    sequences1= tokenizer.texts_to_sequences(test_data['cutted_Dis'])
    test_hh= tokenizer.sequences_to_matrix(sequences1, mode='tfidf')    
    print(X.shape)  
    return X,test_hh,y

def tfidf_process_zi_feats_keras(data,train_data,test_data,num_feats):
    data = cut_zi(data)
    train_data = cut_zi(train_data)
    test_data = cut_zi(test_data)
    y = train_data['Score']  
    tokenizer = Tokenizer(num_words=num_feats)
    tokenizer.fit_on_texts(data['cut_zi'])
    sequences = tokenizer.texts_to_sequences(train_data['cut_zi'])
    X= tokenizer.sequences_to_matrix(sequences, mode='tfidf')
    sequences1= tokenizer.texts_to_sequences(test_data['cut_zi'])
    test_hh= tokenizer.sequences_to_matrix(sequences1, mode='tfidf')    
    print(X.shape)  
    return X,test_hh,y

def nn_process_ci_feats(data,train_data,test_data,VECTOR_DIR,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH):
    y = train_data['Score']  
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data['cutted_Dis'])
    word_index = tokenizer.word_index
    model1 = load_w2v(VECTOR_DIR,EMBEDDING_DIM)
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items(): 
        if word in model1:
            embedding_matrix[i] = np.asarray(model1[word],
                                             dtype='float32')
    sequences = tokenizer.texts_to_sequences(train_data['cutted_Dis'])
    data1 = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    sequences1 = tokenizer.texts_to_sequences(test_data['cutted_Dis'])
    test1 = pad_sequences(sequences1, maxlen=MAX_SEQUENCE_LENGTH)
    return data1,test1,y,embedding_matrix,word_index

def nn_process_zi_feats(data,train_data,test_data,VECTOR_DIR,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH):
    data = cut_zi(data)
    train_data = cut_zi(train_data)
    test_data = cut_zi(test_data)
    y = train_data['Score']  
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data['cutted_Dis'])
    word_index = tokenizer.word_index
    model1 = load_w2v(VECTOR_DIR,EMBEDDING_DIM)
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items(): 
        if word in model1:
            embedding_matrix[i] = np.asarray(model1[word],
                                             dtype='float32')
    sequences = tokenizer.texts_to_sequences(train_data['cutted_Dis'])
    data1 = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    sequences1 = tokenizer.texts_to_sequences(test_data['cutted_Dis'])
    test1 = pad_sequences(sequences1, maxlen=MAX_SEQUENCE_LENGTH)
    return data1,test1,y,embedding_matrix,word_index

def get_model1(maxlen,max_features,EMBEDDING_DIM,embedding_matrix):    
    filter_sizes = [3,2,4,5,6]
    num_filters = 128

    main_input = Input(shape=(maxlen,),name='main_input')
    x= Embedding(max_features, EMBEDDING_DIM, weights=[embedding_matrix])(main_input)
    x=Dense(256,activation='relu')(x)
    x=BatchNormalization()(x)
    pooled=[]
    for i in filter_sizes:
        conv=Conv1D(num_filters, i,padding='same')(x)
        conv=BatchNormalization()(conv)
        conv=Activation('relu')(conv)
#         conv=Activation('relu')(conv)
        conv=Conv1D(num_filters, i,padding='same')(conv)
        conv=BatchNormalization()(conv)   
        maxx=MaxPooling1D()(conv)        
        pooled.append(maxx)       
    z = Concatenate(axis=1)(pooled)
    z=Flatten()(z)
    z=BatchNormalization()(z)
    z=Activation('relu')(z)
    x=Dense(256,activation='relu')(x)
    x=BatchNormalization()(x)    

#     wv_input = Input(shape=(407,),name='wv_input')  
#     tfidf_input = Input(shape=(6000,),name='tfidf_input') 
    #x = layers.concatenate([lstm_out,wv_input,tfidf_input]) 
    x = Dense(128,activation='relu')(z)
    x=BatchNormalization()(x)

    main_output = Dense(1,name='main_output')(x)
    ensem_model = Model(inputs=[main_input],outputs=[main_output])
    ensem_model.compile(optimizer='adam',
                       loss='mse',)
    return ensem_model

def get_model_ince(maxlen,max_features,EMBEDDING_DIM,embedding_matrix):

    filter_sizes = [3,2,4,5,6]
    num_filters = 128
    
    main_input = Input(shape=(maxlen,),name='main_input')
    ploo=[]
    x= Embedding(max_features, EMBEDDING_DIM, weights=[embedding_matrix])(main_input)
    conv1=Conv1D(num_filters,3,padding='same',activation='relu')(x)
    
    conv2=Conv1D(num_filters,4,padding='same',activation='relu')(x)
    conv2=BatchNormalization()(conv2)
    conv2=Activation('relu')(conv2)
    conv2=Conv1D(num_filters,4,padding='same')(conv2)
    
    
    conv3=Conv1D(num_filters,4,padding='same',activation='relu')(x)
    conv3=BatchNormalization()(conv3)
    conv3=Activation('relu')(conv3)
    conv3=Conv1D(num_filters,4,padding='same')(conv3)
    
    conv4=Conv1D(num_filters,5,padding='same',activation='relu')(x)
    ploo.append(conv1)
    ploo.append(conv2)
    ploo.append(conv3)
    ploo.append(conv4)
    
    x2=Concatenate(axis=1)(ploo)
    
    x2=BatchNormalization()(x2)
    x2=Activation('relu')(x2)
    
    conv5=Conv1D(num_filters,3,padding='same',activation='relu')(x)
    
    conv6=Conv1D(num_filters,4,padding='same',activation='relu')(x2)
    conv6=BatchNormalization()(conv6)
    conv6=Activation('relu')(conv6)
    conv6=Conv1D(num_filters,4,padding='same')(conv6)
    
    
    conv7=Conv1D(num_filters,4,padding='same',activation='relu')(x2)
    conv7=BatchNormalization()(conv7)
    conv7=Activation('relu')(conv7)
    conv7=Conv1D(num_filters,4,padding='same')(conv7)
    
    conv8=Conv1D(num_filters,5,padding='same',activation='relu')(x2)
    ploo2=[]
    ploo2.append(conv5)
    ploo2.append(conv6)
    ploo2.append(conv7)
    ploo2.append(conv8)
    
    x3=Concatenate(axis=1)(ploo2)
    x3=Flatten()(x3)
    x3=BatchNormalization()(x3)
   

    x = Dense(128,activation='relu')(x3)
    

    main_output = Dense(1,name='main_output')(x)
   
    ensem_model = Model(inputs=[main_input],outputs=[main_output])
    ensem_model.compile(optimizer='adam',
                       loss='mse',
                       )
    return ensem_model




# def n_folds_train_nn_model_ince(n_folds,y,X,test_hh,model_name,embedding_matrix,maxlen,max_features,EMBEDDING_DIM):
#     # skf = list(StratifiedKFold(y, n_folds,random_state=2018,shuffle=True))
#     skf = KFold(X.shape[0], n_folds,random_state=2018,shuffle=True)

#     daset_blend_train = np.zeros((X.shape[0], 1))
#     daset_blend_test = np.zeros((test_hh.shape[0], 1))
#     daset_blend_test_0 = np.zeros((test_hh.shape[0], len(skf)))

#     for i ,(train,test) in enumerate(skf):
#         print(i)
#         X_train, y_train, X_test, y_test = X[train], y.values[train], X[test], y.values[test]
#         earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=2)
#         if(i<=3):
#             model = load_model(model_name + str(i)+'.h5')
#             y_sub = model.predict(X_test)
#             print(1 /(1 + mean_squared_error(y_test,y_sub)**0.5))
#             print(xx_mse_s(y_test, y_sub))
#             daset_blend_train[test]=y_sub
#             daset_blend_test_0[:,i] = list(model.predict(test_hh))

#         else:
#             model = get_model_ince(maxlen,max_features,EMBEDDING_DIM,embedding_matrix)
#             model.fit([X_train], y_train,
#                         validation_data=([X_test], y_test),
#                         batch_size=64, epochs=15, verbose=1,
#                         callbacks=[earlystopping,
#                                    keras.callbacks.ModelCheckpoint(model_name + str(i)+'.h5', monitor='val_loss', verbose=0,
#                                                              save_best_only=True, mode='auto')])
#             model = load_model(model_name + str(i)+'.h5')
#             y_sub = model.predict(X_test)
#             print(1 /(1 + mean_squared_error(y_test,y_sub)**0.5))
#             print(xx_mse_s(y_test, y_sub))
#             daset_blend_train[test]=y_sub
#             daset_blend_test_0[:,i] = list(model.predict(test_hh))
#     daset_blend_test[:, 0] = daset_blend_test_0.mean(1)

#     return daset_blend_train,daset_blend_test  



def n_folds_train_nn_model_ince(n_folds,y,X,test_hh,model_name,embedding_matrix,maxlen,max_features,EMBEDDING_DIM):
    # skf = list(StratifiedKFold(y, n_folds,random_state=2018,shuffle=True))
    skf = KFold(X.shape[0], n_folds,random_state=2018,shuffle=True)

    daset_blend_train = np.zeros((X.shape[0], 1))
    daset_blend_test = np.zeros((test_hh.shape[0], 1))
    daset_blend_test_0 = np.zeros((test_hh.shape[0], len(skf)))

    for i ,(train,test) in enumerate(skf):

        X_train, y_train, X_test, y_test = X[train], y.values[train], X[test], y.values[test]
        earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=2)
        model = get_model_ince(maxlen,max_features,EMBEDDING_DIM,embedding_matrix)
        model.fit([X_train], y_train,
                    validation_data=([X_test], y_test),
                    batch_size=64, epochs=15, verbose=1,
                    callbacks=[earlystopping,
                               keras.callbacks.ModelCheckpoint(model_name + str(i)+'.h5', monitor='val_loss', verbose=0,
                                                         save_best_only=True, mode='auto')])
        model = load_model(model_name + str(i)+'.h5')
        y_sub = model.predict(X_test)
        print(1 /(1 + mean_squared_error(y_test,y_sub)**0.5))
        print(xx_mse_s(y_test, y_sub))
        daset_blend_train[test]=y_sub
        daset_blend_test_0[:,i] = list(model.predict(test_hh))
    daset_blend_test[:, 0] = daset_blend_test_0.mean(1)

    return daset_blend_train,daset_blend_test  


def n_folds_train_nn_model1(n_folds,y,X,test_hh,model_name,embedding_matrix,maxlen,max_features,EMBEDDING_DIM):
    # skf = list(StratifiedKFold(y, n_folds,random_state=2018,shuffle=True))
    skf = KFold(X.shape[0], n_folds,random_state=2018,shuffle=True)

    daset_blend_train = np.zeros((X.shape[0], 1))
    daset_blend_test = np.zeros((test_hh.shape[0], 1))
    daset_blend_test_0 = np.zeros((test_hh.shape[0], len(skf)))

    for i ,(train,test) in enumerate(skf):
        X_train, y_train, X_test, y_test = X[train], y.values[train], X[test], y.values[test]
        earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=2)
        model = get_model1(maxlen,max_features,EMBEDDING_DIM,embedding_matrix)
        model.fit([X_train], y_train,
                    validation_data=([X_test], y_test),
                    batch_size=64, epochs=15, verbose=1,
                    callbacks=[earlystopping,
                               keras.callbacks.ModelCheckpoint(model_name + str(i)+'.h5', monitor='val_loss', verbose=0,
                                                         save_best_only=True, mode='auto')])
        model = load_model(model_name + str(i)+'.h5')
        y_sub = model.predict(X_test)
        print(1 /(1 + mean_squared_error(y_test,y_sub)**0.5))
        print(xx_mse_s(y_test, y_sub))
        daset_blend_train[test]=y_sub
        daset_blend_test_0[:,i] = list(model.predict(test_hh))
    daset_blend_test[:, 0] = daset_blend_test_0.mean(1)

    return daset_blend_train,daset_blend_test 

    
if __name__ == "__main__":

    data,train_data,test_data,test_id,same,unkown = get_data()


    print("nn_process_ci_feats......")
    VECTOR_DIR = './word2vec/full_dis_256.model'
    EMBEDDING_DIM = 256
    MAX_SEQUENCE_LENGTH = 100
    X,test_hh,y,embedding_matrix,word_index = nn_process_ci_feats(data,train_data,test_data,VECTOR_DIR,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH)
    print('X.shape:',X.shape)
    print('test_hh.shape:',test_hh.shape)   

    model_name = 'model_nn_model1_'
    nn_model1_ci_feat, nn_model1_ci_prob = n_folds_train_nn_model1(5,y,X,test_hh,model_name,embedding_matrix,MAX_SEQUENCE_LENGTH,len(word_index)+1,EMBEDDING_DIM)
    train_data['nn_model1_ci_feat'] = nn_model1_ci_feat
    test_data['nn_model1_ci_prob'] = nn_model1_ci_prob
    train_data[['Id','nn_model1_ci_feat']].to_csv('../stack_cache/5_fold_nn_model1_ci_feat_init.csv',index=False)
    test_data[['Id','nn_model1_ci_prob']].to_csv('../stack_cache/5_fold_nn_model1_ci_prob_init.csv',index=False)   



    print("nn_process_zi_feats......")
    VECTOR_DIR = './word2vec/lg_data_model_comment_256dim_zi'
    EMBEDDING_DIM = 256
    MAX_SEQUENCE_LENGTH = 100
    X,test_hh,y,embedding_matrix,word_index = nn_process_zi_feats(data,train_data,test_data,VECTOR_DIR,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH)
    print('X.shape:',X.shape)
    print('test_hh.shape:',test_hh.shape)   

    model_name = 'model_nn_model1_zi_'
    nn_model1_zi_feat, nn_model1_zi_prob = n_folds_train_nn_model1(5,y,X,test_hh,model_name,embedding_matrix,MAX_SEQUENCE_LENGTH,len(word_index)+1,EMBEDDING_DIM)
    train_data['nn_model1_zi_feat'] = nn_model1_zi_feat
    test_data['nn_model1_zi_prob'] = nn_model1_zi_prob
    train_data[['Id','nn_model1_zi_feat']].to_csv('../stack_cache/5_fold_nn_model1_zi_feat_init.csv',index=False)
    test_data[['Id','nn_model1_zi_prob']].to_csv('../stack_cache/5_fold_nn_model1_zi_prob_init.csv',index=False)   

    data = data_and_shuffle(data)
    train_data = data_and_shuffle(train_data)

    print("nn_process_ci_feats_shuffle......")
    VECTOR_DIR = './word2vec/full_dis_256.model'
    EMBEDDING_DIM = 256
    MAX_SEQUENCE_LENGTH = 100
    X,test_hh,y,embedding_matrix,word_index = nn_process_ci_feats(data,train_data,test_data,VECTOR_DIR,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH)
    print('X.shape:',X.shape)
    print('test_hh.shape:',test_hh.shape)   

    model_name = 'model_nn_model1_shuffle_'
    nn_model1_ci_shuffle_feat, nn_model1_ci_shuffle_prob = n_folds_train_nn_model1(5,y,X,test_hh,model_name,embedding_matrix,MAX_SEQUENCE_LENGTH,len(word_index)+1,EMBEDDING_DIM)
    train_data['nn_model1_ci_shuffle_feat'] = nn_model1_ci_shuffle_feat
    test_data['nn_model1_ci_shuffle_prob'] = nn_model1_ci_shuffle_prob
    train_data[['Id','nn_model1_ci_shuffle_feat']].to_csv('../stack_cache/5_fold_nn_model1_ci_shuffle_feat_init.csv',index=False)
    test_data[['Id','nn_model1_ci_shuffle_prob']].to_csv('../stack_cache/5_fold_nn_model1_ci_shuffle_prob_init.csv',index=False)   

    print("nn_process_ci_feats......")
    VECTOR_DIR = './word2vec/full_dis_256.model'
    EMBEDDING_DIM = 256
    MAX_SEQUENCE_LENGTH = 100
    X,test_hh,y,embedding_matrix,word_index = nn_process_ci_feats(data,train_data,test_data,VECTOR_DIR,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH)
    print('X.shape:',X.shape)
    print('test_hh.shape:',test_hh.shape)   

    model_name = 'model_nn_model_ince_'
    nn_model_ince_ci_feat, nn_model_ince_ci_prob = n_folds_train_nn_model_ince(5,y,X,test_hh,model_name,embedding_matrix,MAX_SEQUENCE_LENGTH,len(word_index)+1,EMBEDDING_DIM)
    train_data['nn_model_ince_ci_feat'] = nn_model_ince_ci_feat
    test_data['nn_model_ince_ci_prob'] = nn_model_ince_ci_prob
    train_data[['Id','nn_model_ince_ci_feat']].to_csv('../stack_cache/5_fold_nn_model_ince_ci_feat_init.csv',index=False)
    test_data[['Id','nn_model_ince_ci_prob']].to_csv('../stack_cache/5_fold_nn_model_ince_ci_prob_init.csv',index=False) 

