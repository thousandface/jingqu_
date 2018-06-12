import numpy as np
import pandas as pd
import re 
import jieba
import gensim
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import backend as k
from keras import callbacks

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense,Input,LSTM,Embedding,Dropout,Activation
from keras.layers import Bidirectional,GlobalMaxPool1D
from keras.models import Model
from keras import initializers,regularizers,constraints,optimizers,layers
import time
from sklearn.linear_model import RidgeCV,LassoCV,ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV,KFold,cross_val_score
from sklearn.base import BaseEstimator,TransformerMixin,RegressorMixin,clone
from sklearn import metrics
from sklearn.model_selection import cross_val_score,KFold
import pandas as pd
full=pd.read_csv('full_tobe_classify_180316.csv',encoding='gbk')

full['cutted_Dis']=full['cutted_Dis'].astype(str)
import re
r=[]
for i in full['Discuss']:
    r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    tem=re.sub(r1, '', i)
    #str31=re.findall(r'(.*?)(\d+\、)(.*?)',i,re.S) 
    r.append(tem)
full['zi']=r
import random
p1=[]
for i in full['zi']:
    i=str(i)
    s=' '
    for j in range(len(i)):
        s=s+' '+i[j]
       

        
    p1.append(s)
full['cutted_Dis']=p1
s = time.ctime()
print('开始时间',s)
COLUMN_NAME = 'cutted_Dis' 
EPOCHS = 10 
BATCH_SIZE = 64  
KFOLD = 5   
MAX_LEN = 100  
EMBED_SIZE = 128  

full = pd.read_csv(r'./full_tobe_classify_180316.csv',engine='python')
full.loc[full.Dis_length==0,'cutted_Dis'] = '空白'
def escore(y,y_pred):
    mse = k.mean(k.square(y-y_pred))
    return 1 / (k.sqrt(mse) + 1)
def split_char(s):
    return ' '.join([' '.join(i) for i in s.split()])
np.random.seed(2018)

train_index = [i for i in full[full.Score.notnull()].index]
np.random.shuffle(train_index)
test_index = [i for i in full[full.Score.isnull()].index]
y = full.loc[train_index].Score.copy()
y = y.reset_index(drop=True)
full_arr = np.load(r'./full_dis_256_4_11_add_6feats.npy')
W2V_DIM = full_arr.shape[1]
train = full_arr[train_index]
test = full_arr[test_index]

scaler = StandardScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)
print(full_arr.shape,train.shape,test.shape,y.shape)
del full_arr

from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

doc = []
_ = full[COLUMN_NAME].apply(lambda s:doc.extend(s.split()))
fre_dic = defaultdict(int)
for w in doc:
    fre_dic[w]+=1
rare_words = [i for i in fre_dic if fre_dic[i]<2]

vectorizer = TfidfVectorizer(stop_words=rare_words,sublinear_tf=True,max_features=6000,ngram_range=(1,3),analyzer='char')
vectorizer.fit(full[full.Score.notnull()][COLUMN_NAME].unique()) 
train_space = vectorizer.transform(full.loc[train_index][COLUMN_NAME])
test_space = vectorizer.transform(full.loc[test_index][COLUMN_NAME])

del doc,fre_dic,rare_words,vectorizer
print(train_space.shape,test_space.shape)

from gensim.models.word2vec import Word2Vec
import gensim
w2v_model = Word2Vec.load(r'./full_dis_256.model')#这里改成字向量的 

EMBEDDING_DIM = 256
MAX_FEATURES = 10000 
train_df = full.loc[train_index]
test_df = full.loc[test_index]

tokenizer = Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(list(train_df[COLUMN_NAME]))
list_tokenized_train = tokenizer.texts_to_sequences(train_df[COLUMN_NAME])
list_tokenized_test = tokenizer.texts_to_sequences(test_df[COLUMN_NAME])

word_index = tokenizer.word_index 
print('Found %s unique tokens.' % len(word_index))
embedding_matrix = np.zeros((len(word_index) + 1,EMBEDDING_DIM))
for word,i in word_index.items():
    embedding_matrix[i,:] = w2v_model[word] if word in w2v_model else np.random.rand(EMBEDDING_DIM)
print('embedding_matrix shape:',embedding_matrix.shape)
X_train = pad_sequences(list_tokenized_train, maxlen=MAX_LEN)
X_test = pad_sequences(list_tokenized_test, maxlen=MAX_LEN)
WORD_IDX_LEN = len(word_index)

del train_df,test_df,word_index,list_tokenized_train,list_tokenized_test,tokenizer,w2v_model
print('X_train shape:{}\tX_test shape:{}\ty shape:{}'.format(X_train.shape, X_test.shape, y.shape))

reducelr = callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5)
earlystopping = callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=2)
cblst = [earlystopping,reducelr]

kfold = StratifiedKFold(n_splits=KFOLD,shuffle=True,random_state=2018)
cvscores_nn_wv = []
dataset_blend_train_nn_wv = np.zeros((train.shape[0],1))
preds_test_blend_nn_wv = np.zeros((test.shape[0],KFOLD)) 
for i,(train_idx,test_idx) in enumerate(kfold.split(train,y)):
    seq_model0 = Sequential()
    seq_model0.add(Dense(1024,input_dim=W2V_DIM,activation='relu'))
    seq_model0.add(Dense(512,activation='relu'))
    seq_model0.add(Dense(512,activation='relu'))
    seq_model0.add(Dropout(0.1))
    seq_model0.add(Dense(512,activation='relu'))
    seq_model0.add(Dense(512,activation='relu'))
    seq_model0.add(Dropout(0.1))
    seq_model0.add(Dense(512,activation='relu'))
    seq_model0.add(Dense(512,activation='relu'))
    seq_model0.add(Dropout(0.1))
    seq_model0.add(Dense(1))

    seq_model0.compile(optimizer='rmsprop',loss='mse',metrics=[escore])
    hist_nn_wv = seq_model0.fit(train[train_idx],y[train_idx],
                                validation_data=(train[test_idx],y[test_idx]),
                                epochs=EPOCHS,batch_size=BATCH_SIZE,verbose=1,callbacks=cblst)
    scores = seq_model0.evaluate(train[test_idx],y[test_idx],verbose=0)
    print('fold',i,'\t evaluation score:',scores,'\n')
    cvscores_nn_wv.append(scores[1])
    y_sub = seq_model0.predict(train[test_idx])
    dataset_blend_train_nn_wv[test_idx,0] = y_sub.flatten()
    y_pred = seq_model0.predict(test)
    preds_test_blend_nn_wv[:,i] = y_pred.flatten()
np.save(r'./dataset_blend_train_nn_wv.npy',dataset_blend_train_nn_wv)
stk_pred_nn_wv = preds_test_blend_nn_wv.mean(axis=1)
np.save(r'./stk_pred_nn_wv.npy',stk_pred_nn_wv)

cvscores_nn_tfidf = []
dataset_blend_train_nn_tfidf = np.zeros((train_space.shape[0],1))
preds_test_blend_nn_tfidf = np.zeros((test_space.shape[0],5))  
for i,(train_idx,test_idx) in enumerate(kfold.split(train_space,y)):
    seq_model1 = Sequential()
    seq_model1.add(Dense(1024,input_dim=6000,activation='relu'))
    seq_model1.add(Dropout(0.2))
    seq_model1.add(Dense(512,activation='relu'))
    seq_model1.add(Dropout(0.2))
    seq_model1.add(Dense(512,activation='relu'))
    seq_model1.add(Dropout(0.2))
    seq_model1.add(Dense(512,activation='relu'))
    seq_model1.add(Dropout(0.2))
    seq_model1.add(Dense(1))

    seq_model1.compile(optimizer='rmsprop',loss='mse',metrics=[escore])
    hist_nn_tfidf = seq_model1.fit(train_space[train_idx],y[train_idx],
                            validation_data=(train_space[test_idx],y[test_idx]),
                            epochs=EPOCHS,batch_size=BATCH_SIZE,verbose=1,callbacks=cblst)
   
    scores = seq_model1.evaluate(train_space[test_idx],y[test_idx],verbose=0)
    print('fold',i,'\t evaluation score:',scores,'\n')
    cvscores_nn_tfidf.append(scores[1])
   
    y_sub = seq_model1.predict(train_space[test_idx])
    dataset_blend_train_nn_tfidf[test_idx,0] = y_sub.flatten()
   
    y_pred = seq_model1.predict(test_space)
    preds_test_blend_nn_tfidf[:,i] = y_pred.flatten()
np.save(r'./dataset_blend_train_nn_tfidf.npy',dataset_blend_train_nn_tfidf)

stk_pred_nn_tfidf = preds_test_blend_nn_tfidf.mean(axis=1)
np.save(r'./stk_pred_nn_tfidf.npy',stk_pred_nn_tfidf)

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2018)
cvscores_nn_wv_0 = []
dataset_blend_train_nn_wv_0 = np.zeros((train.shape[0],1))
preds_test_blend_nn_wv_0 = np.zeros((test.shape[0],5)) 
for i,(train_idx,test_idx) in enumerate(kfold.split(train,y)):

    seq_model0_0 = Sequential()
    seq_model0_0.add(Dense(1024,input_dim=W2V_DIM,activation='relu')) 
    seq_model0_0.add(Dense(512,activation='relu'))
    seq_model0_0.add(Dense(256,activation='relu'))
    seq_model0_0.add(Dense(1))

    seq_model0_0.compile(optimizer='rmsprop',loss='mse',metrics=[escore])
    seq_model0_0.fit(train[train_idx],y[train_idx],
                                validation_data=(train[test_idx],y[test_idx]),
                                epochs=10,batch_size=128,verbose=1,callbacks=cblst)
    scores = seq_model0_0.evaluate(train[test_idx],y[test_idx],verbose=0)
    print('fold',i,'\t evaluation score:',scores,'\n')
    cvscores_nn_wv_0.append(scores[1])
    y_sub = seq_model0_0.predict(train[test_idx])
    dataset_blend_train_nn_wv_0[test_idx,0] = y_sub.flatten()
    y_pred = seq_model0_0.predict(test)
    preds_test_blend_nn_wv_0[:,i] = y_pred.flatten()
np.save(r'./dataset_blend_train_nn_wv_0.npy',dataset_blend_train_nn_wv_0)
stk_pred_nn_wv_0 = preds_test_blend_nn_wv_0.mean(axis=1)
np.save(r'./stk_pred_nn_wv_0.npy',stk_pred_nn_wv_0)

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2018)
cvscores_nn_wv_1 = []
dataset_blend_train_nn_wv_1 = np.zeros((train.shape[0],1))
preds_test_blend_nn_wv_1 = np.zeros((test.shape[0],5))  
for i,(train_idx,test_idx) in enumerate(kfold.split(train,y)):
    seq_model0_1 = Sequential()
    seq_model0_1.add(Dense(512,input_dim=W2V_DIM,activation='relu'))  
    seq_model0_1.add(Dense(512,activation='relu'))
    seq_model0_1.add(Dense(512,activation='relu'))
    seq_model0_1.add(Dense(512,activation='relu'))
    seq_model0_1.add(Dense(512,activation='relu'))
    seq_model0_1.add(Dense(1))

    seq_model0_1.compile(optimizer='rmsprop',loss='mse',metrics=[escore])
    seq_model0_1.fit(train[train_idx],y[train_idx],
                                validation_data=(train[test_idx],y[test_idx]),
                                epochs=10,batch_size=128,verbose=1,callbacks=cblst)
    #评测
    scores = seq_model0_1.evaluate(train[test_idx],y[test_idx],verbose=0)
    print('fold',i,'\t evaluation score:',scores,'\n')
    cvscores_nn_wv_1.append(scores[1])
    #验证集的预测结果
    y_sub = seq_model0_1.predict(train[test_idx])
    dataset_blend_train_nn_wv_1[test_idx,0] = y_sub.flatten()
    #预测
    y_pred = seq_model0_1.predict(test)
    preds_test_blend_nn_wv_1[:,i] = y_pred.flatten()
np.save(r'./dataset_blend_train_nn_wv_1.npy',dataset_blend_train_nn_wv_1)
stk_pred_nn_wv_1 = preds_test_blend_nn_wv_1.mean(axis=1)
np.save(r'./stk_pred_nn_wv_1.npy',stk_pred_nn_wv_1)

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2018)
cvscores_nn_tfidf_0 = []
dataset_blend_train_nn_tfidf_0 = np.zeros((train_space.shape[0],1))
preds_test_blend_nn_tfidf_0 = np.zeros((test_space.shape[0],5)) 
for i,(train_idx,test_idx) in enumerate(kfold.split(train_space,y)):
    seq_model1_0 = Sequential()
    seq_model1_0.add(Dense(512,input_dim=6000,activation='relu'))
    seq_model1_0.add(Dropout(0.2))
    seq_model1_0.add(Dense(256,activation='relu'))
    seq_model1_0.add(Dropout(0.2))
    seq_model1_0.add(Dense(256,activation='relu'))
    seq_model1_0.add(Dropout(0.2))
    seq_model1_0.add(Dense(256,activation='relu'))
    seq_model1_0.add(Dropout(0.2))
    seq_model1_0.add(Dense(256,activation='relu'))
    seq_model1_0.add(Dropout(0.2))
    seq_model1_0.add(Dense(1))

    seq_model1_0.compile(optimizer='rmsprop',loss='mse',metrics=[escore])
    seq_model1_0.fit(train_space[train_idx],y[train_idx],
                            validation_data=(train_space[test_idx],y[test_idx]),
                            epochs=10,batch_size=128,verbose=1,callbacks=cblst)
    #评测
    scores = seq_model1_0.evaluate(train_space[test_idx],y[test_idx],verbose=0)
    print('fold',i,'\t evaluation score:',scores,'\n')
    cvscores_nn_tfidf_0.append(scores[1])
    #验证集的预测结果
    y_sub = seq_model1_0.predict(train_space[test_idx])
    dataset_blend_train_nn_tfidf_0[test_idx,0] = y_sub.flatten()
    #预测
    y_pred = seq_model1_0.predict(test_space)
    preds_test_blend_nn_tfidf_0[:,i] = y_pred.flatten()
stk_pred_nn_tfidf_0 = preds_test_blend_nn_tfidf_0.mean(axis=1)
np.save(r'./dataset_blend_train_nn_tfidf_0.npy',dataset_blend_train_nn_tfidf_0)
np.save(r'./stk_pred_nn_tfidf_0.npy',stk_pred_nn_tfidf_0)

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2018)
cvscores_nn_tfidf_1 = []
dataset_blend_train_nn_tfidf_1 = np.zeros((train_space.shape[0],1))
preds_test_blend_nn_tfidf_1 = np.zeros((test_space.shape[0],5)) 
for i,(train_idx,test_idx) in enumerate(kfold.split(train_space,y)):
    seq_model1_1 = Sequential()
    seq_model1_1.add(Dense(256,input_dim=6000,activation='relu'))
    seq_model1_1.add(Dropout(0.2))
    seq_model1_1.add(Dense(256,activation='relu'))
    seq_model1_1.add(Dropout(0.2))
    seq_model1_1.add(Dense(1))

    seq_model1_1.compile(optimizer='rmsprop',loss='mse',metrics=[escore])
    seq_model1_1.fit(train_space[train_idx],y[train_idx],
                            validation_data=(train_space[test_idx],y[test_idx]),
                            epochs=10,batch_size=32,verbose=1,callbacks=cblst)
    #评测
    scores = seq_model1_1.evaluate(train_space[test_idx],y[test_idx],verbose=0)
    print('fold',i,'\t evaluation score:',scores,'\n')
    cvscores_nn_tfidf_1.append(scores[1])
    #验证集的预测结果
    y_sub = seq_model1_1.predict(train_space[test_idx])
    dataset_blend_train_nn_tfidf_1[test_idx,0] = y_sub.flatten()
    #预测
    y_pred = seq_model1_1.predict(test_space)
    preds_test_blend_nn_tfidf_1[:,i] = y_pred.flatten()
stk_pred_nn_tfidf_1 = preds_test_blend_nn_tfidf_1.mean(axis=1)
np.save(r'./dataset_blend_train_nn_tfidf_1.npy',dataset_blend_train_nn_tfidf_1)
np.save(r'./stk_pred_nn_tfidf_1.npy',stk_pred_nn_tfidf_1)


kfold = StratifiedKFold(n_splits=KFOLD,shuffle=True,random_state=2018)
cvscores_lstm = []
dataset_blend_train_lstm = np.zeros((X_train.shape[0],1))
preds_test_blend_lstm = np.zeros((X_test.shape[0],KFOLD)) 
for i,(train_idx,test_idx) in enumerate(kfold.split(X_train,y)):
    inp = Input(shape=(MAX_LEN,))
    x = Embedding(WORD_IDX_LEN + 1, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)(inp)
    x = LSTM(128,return_sequences=True,name='lstm_layer')(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(512,activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(512,activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256,activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1)(x)
    lstm_model = Model(inputs=inp,outputs=x)
    lstm_model.compile(loss='mse',
                 optimizer='rmsprop',
                 metrics=[escore])
    hist_lstm = lstm_model.fit(X_train[train_idx],y[train_idx],batch_size=BATCH_SIZE,epochs=EPOCHS,verbose=1,
                               validation_data=(X_train[test_idx],y[test_idx]),
                               callbacks=cblst)
    #评测
    scores = lstm_model.evaluate(X_train[test_idx],y[test_idx],verbose=0)
    print('fold',i,'\t evaluation score:',scores,'\n')
    cvscores_lstm.append(scores[1])
    #验证集的预测结果
    y_sub = lstm_model.predict(X_train[test_idx])
    dataset_blend_train_lstm[test_idx,0] = y_sub.flatten()
    #预测
    y_pred = lstm_model.predict(X_test)
    preds_test_blend_lstm[:,i] = y_pred.flatten()
np.save(r'./dataset_blend_train_lstm.npy',dataset_blend_train_lstm)
stk_pred_lstm = preds_test_blend_lstm.mean(axis=1)
np.save(r'./stk_pred_lstm.npy',stk_pred_lstm)

kfold = StratifiedKFold(n_splits=KFOLD,shuffle=True,random_state=2018)
cvscores_tri_inps = []
dataset_blend_train_tri_inps = np.zeros((X_train.shape[0],1))
preds_test_blend_tri_inps = np.zeros((X_test.shape[0],KFOLD)) 
for i,(train_idx,test_idx) in enumerate(kfold.split(X_train,y)):
    main_input = Input(shape=(MAX_LEN,),name='main_input')
    x = Embedding(WORD_IDX_LEN + 1, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)(main_input)
    lstm_out = LSTM(64,return_sequences=True,name='lstm_layer')(x)
    lstm_out = GlobalMaxPool1D()(lstm_out)

    wv_input = Input(shape=(W2V_DIM,),name='wv_input')  
    tfidf_input = Input(shape=(6000,),name='tfidf_input') 
    x = layers.concatenate([lstm_out,wv_input,tfidf_input])  
    x = Dense(1024,activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(512,activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(512,activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(512,activation='relu')(x)
    x = Dropout(0.1)(x)
    main_output = Dense(1,name='main_output')(x)
    ensem_model = Model(inputs=[main_input,wv_input,tfidf_input],outputs=[main_output])
    ensem_model.compile(optimizer='rmsprop',
                       loss='mse',
                       metrics=[escore])
    hist2 = ensem_model.fit([X_train[train_idx],train[train_idx],train_space[train_idx]],y[train_idx],
                    batch_size=BATCH_SIZE,epochs=EPOCHS,verbose=1,
                    validation_data = ([X_train[test_idx],train[test_idx],train_space[test_idx]],y[test_idx]),
                    callbacks=cblst)
    #评测
    scores = ensem_model.evaluate([X_train[test_idx],train[test_idx],train_space[test_idx]],y[test_idx],verbose=0)
    print('fold',i,'\t evaluation score:',scores,'\n')
    cvscores_tri_inps.append(scores[1])
    #验证集的预测结果
    y_sub = ensem_model.predict([X_train[test_idx],train[test_idx],train_space[test_idx]])
    dataset_blend_train_tri_inps[test_idx,0] = y_sub.flatten()
    #预测
    y_pred = ensem_model.predict([X_test,test,test_space])
    preds_test_blend_tri_inps[:,i] = y_pred.flatten()

np.save(r'./dataset_blend_train_tri_inps.npy',dataset_blend_train_tri_inps)
stk_pred_tri_inps = preds_test_blend_tri_inps.mean(axis=1)
np.save(r'./stk_pred_tri_inps.npy',stk_pred_tri_inps)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
cvscores_cnn = []
dataset_blend_train_cnn = np.zeros((X_train.shape[0], 1))
preds_test_blend_cnn = np.zeros((X_test.shape[0], 5)) 
for i, (train_idx, test_idx) in enumerate(kfold.split(X_train, y)):
    X_t, y_t, X_te, y_te = X_train[train_idx], y[train_idx], X_train[test_idx], y[test_idx]

    cnn_model0 = Sequential()
    cnn_model0.add(Embedding(WORD_IDX_LEN+1, EMBEDDING_DIM,weights=[embedding_matrix], input_length=MAX_LEN, trainable=False))
    cnn_model0.add(layers.Conv1D(128, 5, activation='relu'))
    cnn_model0.add(layers.GlobalMaxPooling1D())
    cnn_model0.add(layers.Dense(512, activation='relu'))
    cnn_model0.add(layers.Dense(512, activation='relu'))
    cnn_model0.add(Dropout(0.1))
    cnn_model0.add(layers.Dense(512, activation='relu'))
    cnn_model0.add(Dropout(0.1))
    cnn_model0.add(layers.Dense(512, activation='relu'))
    cnn_model0.add(Dropout(0.1))
    cnn_model0.add(layers.Dense(1))
    cnn_model0.compile(optimizer='rmsprop', loss='mse', metrics=[escore])
    hist_cnn = cnn_model0.fit(X_t, y_t,
                              epochs=10, batch_size=64, validation_data=(X_te, y_te),
                              callbacks=cblst)
    # 评测
    scores = cnn_model0.evaluate(X_te, y_te, verbose=0)
    print('fold', i, '\t evaluation score:', scores, '\n')
    cvscores_cnn.append(scores[1])
    # 验证集的预测结果
    y_sub = cnn_model0.predict(X_te)
    dataset_blend_train_cnn[test_idx, 0] = y_sub.flatten()
    # 预测
    y_pred = cnn_model0.predict(X_test)
    preds_test_blend_cnn[:, i] = y_pred.flatten()
stk_pred_cnn = preds_test_blend_cnn.mean(axis=1)
np.save(r'./dataset_blend_train_cnn.npy',dataset_blend_train_cnn)
np.save(r'./stk_pred_cnn.npy',stk_pred_cnn)

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2018)
cvscores_cnn_1 = []
dataset_blend_train_cnn_1 = np.zeros((X_train.shape[0],1))
preds_test_blend_cnn_1 = np.zeros((X_test.shape[0],5)) 
extra_conv = False
for i,(train_idx,test_idx) in enumerate(kfold.split(X_train,y)):
    X_t,y_t,X_te,y_te = X_train[train_idx],y[train_idx],X_train[test_idx],y[test_idx]
    embedding_layer = Embedding(WORD_IDX_LEN+1, EMBEDDING_DIM,weights=[embedding_matrix], input_length=MAX_LEN,trainable=False)
    inp = Input(shape=(MAX_LEN,))
    embedding_sequence = embedding_layer(inp)
    convs = []
    filter_sizes = [3,4,5]

    for filter_size in filter_sizes:
        l_conv = layers.Conv1D(filters=128,kernel_size=filter_size,activation='relu')(embedding_sequence)
        l_pool = layers.MaxPool1D(pool_size=3)(l_conv)
        convs.append(l_pool)

    l_merged = layers.Merge(mode='concat',concat_axis=1)(convs)
    conv = layers.Conv1D(filters=128,kernel_size=3,activation='relu')(embedding_sequence)
    pool = layers.MaxPooling1D(pool_size=3)(conv)
    if extra_conv:
        x = Dropout(0.2)(l_merged)
    else:
        x = Dropout(0.2)(pool)
    x = layers.Flatten()(x)
    x = Dense(512,activation='relu')(x)
    x = Dense(512,activation='relu')(x)
    x = Dense(512,activation='relu')(x)
    x = Dense(1)(x)

    cnn_model1 = Model(inp,x)
    cnn_model1.compile(optimizer='rmsprop',loss='mse',metrics=[escore])
    earlystopping = callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=2)
    hist_cnn1 = cnn_model1.fit(X_t,y_t,
                               validation_data=(X_te,y_te),
                             epochs=10,batch_size=64,callbacks=[earlystopping])
    #评测
    scores = cnn_model1.evaluate(X_te,y_te,verbose=0)
    print('fold',i,'\t evaluation score:',scores,'\n')
    cvscores_cnn_1.append(scores[1])
    #验证集的预测结果
    y_sub = cnn_model1.predict(X_te)
    dataset_blend_train_cnn_1[test_idx,0] = y_sub.flatten()
    #预测
    y_pred = cnn_model1.predict(X_test)
    preds_test_blend_cnn_1[:,i] = y_pred.flatten()
stk_pred_cnn_1 = preds_test_blend_cnn_1.mean(axis=1)
np.save(r'./dataset_blend_train_cnn_1.npy',dataset_blend_train_cnn_1)
np.save(r'./stk_pred_cnn_1.npy',stk_pred_cnn_1)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
cvscores_5_0cnn = []
dataset_blend_train_5_0cnn = np.zeros((X_train.shape[0], 1))
preds_test_blend_5_0cnn = np.zeros((X_test.shape[0], 5)) 
for i, (train_idx, test_idx) in enumerate(kfold.split(X_train, y)):
    X_t, y_t, X_te, y_te = X_train[train_idx], y[train_idx], X_train[test_idx], y[test_idx]
    model_5_0cnn = Sequential()
    model_5_0cnn.add(Embedding(WORD_IDX_LEN + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_LEN))
    model_5_0cnn.add(layers.Convolution1D(512, 3, padding='same'))
    model_5_0cnn.add(layers.MaxPool1D(3, 3, padding='same'))
    model_5_0cnn.add(layers.Convolution1D(256, 3, padding='same'))
    model_5_0cnn.add(layers.MaxPool1D(3, 3, padding='same'))
    model_5_0cnn.add(layers.Convolution1D(128, 3, padding='same'))
    model_5_0cnn.add(layers.MaxPooling1D(3, 3, padding='same'))
    model_5_0cnn.add(layers.Flatten())
    model_5_0cnn.add(Dropout(0.1))
    model_5_0cnn.add(layers.BatchNormalization())
    model_5_0cnn.add(Dense(256, activation='relu'))
    model_5_0cnn.add(Dropout(0.1))
    model_5_0cnn.add(Dense(256, activation='relu'))
    model_5_0cnn.add(Dropout(0.1))
    model_5_0cnn.add(Dense(256, activation='relu'))
    model_5_0cnn.add(Dropout(0.1))
    model_5_0cnn.add(Dense(1))
    model_5_0cnn.compile(optimizer='rmsprop', loss='mse', metrics=[escore])
    model_5_0cnn.fit(X_t, y_t, epochs=10, batch_size=64,
                     validation_data=(X_te, y_te),
                     callbacks=cblst)
    # 评测
    scores = model_5_0cnn.evaluate(X_te, y_te, verbose=0)
    print('fold', i, '\t evaluation score:', scores, '\n')
    cvscores_5_0cnn.append(scores[1])
    # 验证集的预测结果
    y_sub = model_5_0cnn.predict(X_te)
    dataset_blend_train_5_0cnn[test_idx, 0] = y_sub.flatten()
    # 预测
    y_pred = model_5_0cnn.predict(X_test)
    preds_test_blend_5_0cnn[:, i] = y_pred.flatten()
stk_pred_5_0cnn = preds_test_blend_5_0cnn.mean(axis=1)
np.save(r'./dataset_blend_train_5_0cnn.npy',dataset_blend_train_5_0cnn)
np.save(r'./stk_pred_5_0cnn.npy',stk_pred_5_0cnn)

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2018)
cvscores_5_3rnn = []
dataset_blend_train_5_3rnn = np.zeros((X_train.shape[0],1))
preds_test_blend_5_3rnn = np.zeros((X_test.shape[0],5))  
for i,(train_idx,test_idx) in enumerate(kfold.split(X_train,y)):
    X_t,y_t,X_te,y_te = X_train[train_idx],y[train_idx],X_train[test_idx],y[test_idx]
    model_5_3bidirecRnn = Sequential()
    model_5_3bidirecRnn.add(Embedding(WORD_IDX_LEN + 1,EMBEDDING_DIM,input_length=MAX_LEN))
    model_5_3bidirecRnn.add(Bidirectional(layers.GRU(256,dropout=0.2,recurrent_dropout=0.1,return_sequences=True)))
    model_5_3bidirecRnn.add(Bidirectional(layers.GRU(256,dropout=0.2,recurrent_dropout=0.1)))
    model_5_3bidirecRnn.add(Dense(1))

    model_5_3bidirecRnn.compile(optimizer='rmsprop',loss='mse',metrics=[escore])
    earlystopping = callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=2)
    model_5_3bidirecRnn.fit(X_t,y_t,batch_size=64,epochs=10,
                    validation_data=(X_te,y_te),
                            callbacks=[earlystopping])
    #评测
    scores = model_5_3bidirecRnn.evaluate(X_te,y_te,verbose=0)
    print('fold',i,'\t evaluation score:',scores,'\n')
    cvscores_5_3rnn.append(scores[1])
    #验证集的预测结果
    y_sub = model_5_3bidirecRnn.predict(X_te)
    dataset_blend_train_5_3rnn[test_idx,0] = y_sub.flatten()
    #预测
    y_pred = model_5_3bidirecRnn.predict(X_test)
    preds_test_blend_5_3rnn[:,i] = y_pred.flatten()
stk_pred_5_3rnn = preds_test_blend_5_3rnn.mean(axis=1)
np.save(r'./dataset_blend_train_5_3rnn.npy',dataset_blend_train_5_3rnn)
np.save(r'./stk_pred_5_3rnn.npy',stk_pred_5_3rnn)

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2018)
cvscores_5_4C_Rnn = []
dataset_blend_train_5_4C_Rnn = np.zeros((X_train.shape[0],1))
preds_test_blend_5_4C_Rnn = np.zeros((X_test.shape[0],5)) 
for i,(train_idx,test_idx) in enumerate(kfold.split(X_train,y)):
    X_t,y_t,X_te,y_te = X_train[train_idx],y[train_idx],X_train[test_idx],y[test_idx]
    model_5_4C_Rnn = Sequential()
    model_5_4C_Rnn.add(Embedding(WORD_IDX_LEN+1,EMBEDDING_DIM,input_length=MAX_LEN))
    model_5_4C_Rnn.add(layers.Convolution1D(256,3,padding='same',strides=1))
    model_5_4C_Rnn.add(layers.Activation('relu'))
    model_5_4C_Rnn.add(layers.MaxPool1D(pool_size=2))
    model_5_4C_Rnn.add(layers.GRU(256,dropout=0.2,recurrent_dropout=0.1,return_sequences=True))
    model_5_4C_Rnn.add(layers.GRU(256,dropout=0.2,recurrent_dropout=0.1))
    model_5_4C_Rnn.add(Dense(1))

    model_5_4C_Rnn.compile(optimizer='rmsprop',loss='mse',metrics=[escore])
    model_5_4C_Rnn.fit(X_t,y_t,batch_size=64,epochs=10,
                    validation_data=(X_te,y_te),callbacks=cblst)
    #评测
    scores = model_5_4C_Rnn.evaluate(X_te,y_te,verbose=0)
    print('fold',i,'\t evaluation score:',scores,'\n')
    cvscores_5_4C_Rnn.append(scores[1])
    #验证集的预测结果
    y_sub = model_5_4C_Rnn.predict(X_te)
    dataset_blend_train_5_4C_Rnn[test_idx,0] = y_sub.flatten()
    #预测
    y_pred = model_5_4C_Rnn.predict(X_test)
    preds_test_blend_5_4C_Rnn[:,i] = y_pred.flatten()
stk_pred_5_4C_Rnn = preds_test_blend_5_4C_Rnn.mean(axis=1)
np.save(r'./dataset_blend_train_5_4C_Rnn.npy',dataset_blend_train_5_4C_Rnn)
np.save(r'./stk_pred_5_4C_Rnn.npy',stk_pred_5_4C_Rnn)


print('保存结果')

models_pred= pd.DataFrame({'Id':full[full.Score.isnull()].Id,
                           'stk_pred_nn_wv':stk_pred_nn_wv,
                           'stk_pred_nn_tfidf':stk_pred_nn_tfidf,
                           'stk_pred_nn_tfidf_0':stk_pred_nn_tfidf_0,
                           'stk_pred_nn_tfidf_1':stk_pred_nn_tfidf_1,
                           'stk_pred_nn_wv_0':stk_pred_nn_wv_0,
                           'stk_pred_nn_wv_1':stk_pred_nn_wv_1,
                           'stk_pred_cnn':stk_pred_cnn,
                           'stk_pred_lstm':stk_pred_lstm,
                           'stk_pred_tri_inps':stk_pred_tri_inps,
                           'stk_pred_cnn_1':stk_pred_cnn_1,
                           'stk_pred_5_0cnn':stk_pred_5_0cnn,
                           'stk_pred_5_3rnn':stk_pred_5_3rnn,
                           'stk_pred_5_4C_Rnn':stk_pred_5_4C_Rnn
                          })
models_cvscores = pd.DataFrame({'cvscores_nn_wv':cvscores_nn_wv,
                               'cvscores_nn_tfidf':cvscores_nn_tfidf,
                                'cvscores_nn_tfidf_0':cvscores_nn_tfidf_0,
                                'cvscores_nn_tfidf_1':cvscores_nn_tfidf_1,
                                'cvscores_nn_wv_0':cvscores_nn_wv_0,
                                'cvscores_nn_wv_1':cvscores_nn_wv_1,
                                'cvscores_cnn':cvscores_cnn,
                               'cvscores_lstm':cvscores_lstm,
                               'cvscores_tri_inps':cvscores_tri_inps,
                                'cvscores_cnn_1':cvscores_cnn_1,
                                'cvscores_5_0cnn':cvscores_5_0cnn,
                                'cvscores_5_3rnn':cvscores_5_3rnn,
                                'cvscores_5_4C_Rnn':cvscores_5_4C_Rnn,
                               })

dataset_train_blends = pd.DataFrame({'dataset_blend_train_nn_tfidf':dataset_blend_train_nn_tfidf.flatten(),
                                    'dataset_blend_train_nn_wv':dataset_blend_train_nn_wv.flatten(),
                                    'dataset_blend_train_nn_tfidf_0':dataset_blend_train_nn_tfidf_0.flatten(),
                                     'dataset_blend_train_nn_tfidf_1':dataset_blend_train_nn_tfidf_1.flatten(),
                                     'dataset_blend_train_nn_wv_0':dataset_blend_train_nn_wv_0.flatten(),
                                     'dataset_blend_train_nn_wv_1':dataset_blend_train_nn_wv_1.flatten(),
                                     'dataset_blend_train_cnn':dataset_blend_train_cnn.flatten(),
                                    'dataset_blend_train_tri_inps':dataset_blend_train_tri_inps.flatten(),
                                    'dataset_blend_train_lstm':dataset_blend_train_lstm.flatten(),
                                     'dataset_blend_train_cnn_1':dataset_blend_train_cnn_1.flatten(),
                                     'dataset_blend_train_5_0cnn':dataset_blend_train_5_0cnn.flatten(),
                                     'dataset_blend_train_5_3rnn':dataset_blend_train_5_3rnn.flatten(),
                                     'dataset_blend_train_5_4C_Rnn':dataset_blend_train_5_4C_Rnn.flatten()
                                    })

print(models_pred.shape,models_cvscores.shape,dataset_train_blends.shape)
models_cvscores.loc[5] = models_cvscores.mean() 

dataset_train_blends.to_csv(r'./full_dataset_train_blends.csv',index=None)
models_pred.to_csv(r'./full_models_pred.csv',index=None)
models_cvscores.to_csv(r'./models_cvscores.csv',index=None)




