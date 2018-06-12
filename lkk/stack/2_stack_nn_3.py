import pandas as pd

b = pd.read_csv('../data/full_tobe_classify_180316.csv',encoding='GBK')
test_data = b[b.Score.isnull()].reset_index()
test_id=test_data['Id'].values.copy()

# test_data=pd.read_csv('lg_test_data.csv',encoding='gbk')
# test_id=test_data['Id'].values.copy()
# 
# import keras
import keras.backend as k
from keras.models import Model
from keras.layers import Dense,Embedding,Input,Activation,BatchNormalization
from keras.layers import Conv1D
from keras.layers.core import Flatten
from keras.layers.pooling import MaxPooling1D

from keras.models import Model
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, AveragePooling2D, GlobalMaxPooling1D,LSTM
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback

filter_sizes = [3,2,4,5,6,1,7,8,9]
num_filters = 128

def get_model1():
    
    main_input = Input(shape=(maxlen,),name='main_input')
    embed_size =256
    x= Embedding(max_features, embed_size, weights=[embedding_matrix])(main_input)
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
                       loss='mse',
                       metrics=[escore])
    return ensem_model

def get_model_textcnn():
    main_input = Input(shape=(maxlen,),name='main_input')
    embed_size =256
    x= Embedding(max_features, embed_size, weights=[embedding_matrix])(main_input)
    wv_input = Input(shape=(263,),name='wv_input')  
    tfidf_input = Input(shape=(5000,),name='tfidf_input')
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
    z = layers.concatenate([z,wv_input,tfidf_input])
    z = Dense(1024,activation='relu')(z)
    z = BatchNormalization()(z)
    z = Dense(512,activation='relu')(z)
    z = BatchNormalization()(z)
    z = Dense(512,activation='relu')(z)
    z=BatchNormalization()(z)
    z=Activation('relu')(z)
    

#     wv_input = Input(shape=(407,),name='wv_input')  
#     tfidf_input = Input(shape=(6000,),name='tfidf_input') 
    #x = layers.concatenate([lstm_out,wv_input,tfidf_input]) 
    x = Dense(128,activation='relu')(z)

    main_output = Dense(1,name='main_output')(x)
    ensem_model = Model(inputs=[main_input,wv_input,tfidf_input],outputs=[main_output])
    ensem_model.compile(optimizer='adam',
                       loss='mse',
                       metrics=[escore])
    return ensem_model
    

#3 44 44 5 44445
def get_model_ince():
    
    main_input = Input(shape=(maxlen,),name='main_input')
    embed_size =256
    ploo=[]
    x= Embedding(max_features, embed_size, weights=[embedding_matrix])(main_input)
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
                       metrics=[escore])
    return ensem_model
def escore(y,y_pred):
    mse = k.mean(k.square(y-y_pred))
    return 1 / (k.sqrt(mse) + 1)




#数据处理
import pandas as pd
data=pd.read_csv('../data/full_tobe_classify_180316.csv',encoding='gbk')
data['cutted_Dis']=data['cutted_Dis'].astype(str)
train_index = data[data.Score.notnull()].index
test_index = data[data.Score.isnull()].index

train_data=data.loc[[i for i in train_index]]
test_data=data.loc[[i for i in test_index]]
train_data.reset_index(drop=True,inplace=True)
test_data.reset_index(drop=True,inplace=True)

import random
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

train_index = data1[data1.Score.notnull()].index
test_index = data1[data1.Score.isnull()].index

train_data1=data1.loc[[i for i in train_index]]
test_data1=data1.loc[[i for i in test_index]]
train_data1.reset_index(drop=True,inplace=True)
test_data1.reset_index(drop=True,inplace=True)

data2=pd.concat([data,data1])
train_data3=pd.concat([train_data,train_data1])
train_data4=pd.DataFrame()
train_data4['Score']=train_data3['Score'].values
train_data4['cutted_Dis']=train_data3['cutted_Dis'].values


from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers.recurrent import LSTM, GRU 
from keras.models import Sequential
MAX_SEQUENCE_LENGTH = 100 # 每条新闻最大长度
EMBEDDING_DIM = 256 # 词向量空间维度
VALIDATION_SPLIT = 0.16 # 验证集比例
TEST_SPLIT = 0.2 # 测试集比例
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential
from keras.utils import plot_model
from keras.layers import Embedding
import gensim
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate,Concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from scipy.sparse import csr_matrix, hstack,vstack


def xx_mse_s(y_true,y_pre):
    y_true = y_true
    y_pre = pd.DataFrame({'res':list(y_pre)})

    y_pre['res'] = y_pre['res'].astype(int)
    return 1 / ( 1 + mean_squared_error(y_true,y_pre['res'].values)**0.5)

tokenizer = Tokenizer()

tokenizer.fit_on_texts(data['cutted_Dis'])

# tokenizer1 = Tokenizer()

# tokenizer1.fit_on_texts(data1['cutted_Dis'])

# tokenizer1 = Tokenizer()

# tokenizer1.fit_on_texts(data['cutted_Dis'])

word_index = tokenizer.word_index
# word_index1 = tokenizer1.word_index

VECTOR_DIR = '../stack/word2vec/full_dis_256.model'
from keras.utils import plot_model
from keras.layers import Embedding
from gensim.models.word2vec import Word2Vec
model1=Word2Vec.load(VECTOR_DIR)

#w2v_model = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_DIR, binary=True,encoding='gbk')
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items(): 
    if word in model1:
        embedding_matrix[i] = np.asarray(model1[word],
                                         dtype='float32')
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

# embedding_matrix1 = np.zeros((len(word_index1) + 1, EMBEDDING_DIM))
# for word, i in word_index1.items(): 
#     if word in model1:
#         embedding_matrix1[i] = np.asarray(model1[word],
#                                          dtype='float32')
        
# embedding_layer1 = Embedding(len(word_index1) + 1,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix1],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=False)

sequences = tokenizer.texts_to_sequences(train_data['cutted_Dis'])
X= pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# sequences2 = tokenizer1.texts_to_sequences(train_data1['cutted_Dis'])
# X1= pad_sequences(sequences2, maxlen=MAX_SEQUENCE_LENGTH)

# sequences5 = tokenizer1.texts_to_sequences(test_data3['cutted_Dis'])
# test3= pad_sequences(sequences5, maxlen=MAX_SEQUENCE_LENGTH)

sequences1= tokenizer.texts_to_sequences(test_data['cutted_Dis'])
test_hh= pad_sequences(sequences1, maxlen=MAX_SEQUENCE_LENGTH)


y=train_data['Score']

from sklearn.cross_validation import StratifiedKFold
n_folds=5
# skf = list(StratifiedKFold(y, n_folds))
skf = KFold(X.shape[0], n_folds,random_state=2018,shuffle=True)
y1=y
cv_pred = []

xx_mse = []
#labels = to_categorical(np.asarray(y1))
labels = y1
maxlen=100
embed_size=256
max_features=len(word_index)+1
def get_label(x):
    val_=[]
    for i in x:
        reslt=i[1]*1+i[2]*2+i[3]*3+i[4]*4+i[5]*5
        val_.append(reslt)
    return val_

    
    
daset_blend_train = np.zeros((X.shape[0], 1))
daset_blend_test = np.zeros((test_hh.shape[0], 1))

daset_blend_test_0 = np.zeros((test_hh.shape[0], len(skf)))

for i ,(train_fold,test_fold) in enumerate(skf):
    X_train, X_validate, label_train, label_validate = X[train_fold, :], X[test_fold, :], labels[train_fold], y[test_fold]
#     X_train1, X_validate1, label_train1, label_validate1 = X1[train_fold, :], X1[test_fold, :], y1[train_fold], y1[test_fold]
#     X_train = np.concatenate([X_train,X_train1])
# #     X_validate = np.concatenate([X_validate,X_validate1])
#     label_train=np.concatenate([label_train,label_train1])
#     label_validate=np.concatenate([label_validate,label_validate1])
    model=get_model1()

    earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=2)
#     model.fit([X_train,train[train_fold],train_t[train_fold]],label_train,
#                     batch_size=64,epochs=8,verbose=1,
                    
#                     callbacks=[earlystopping])
    model.fit(X_train,label_train,
                    batch_size=64,epochs=5,verbose=1,
                    callbacks=[earlystopping])
    y_sub = model.predict(X_validate)
#     y_sub = model.predict([X_validate,train[test_fold],train_t[test_fold]])
    print(xx_mse_s(label_validate, y_sub))
    daset_blend_train[test_fold,]=y_sub
#     daset_blend_test_0[:,i]=model.predict(test_hh)
#    cv_pred.append(model.predict([test_hh,test,test_t]))
    cv_pred.append(model.predict(test_hh))
        
       
        
# daset_blend_test[:, ] = daset_blend_test_0.mean(1)

# import numpy as np+
# print('xx_result',np.mean(xx_mse))
 

re=[]
for i in range(34930):  
    s=(cv_pred[0][i][0]+cv_pred[1][i][0]+cv_pred[2][i][0]+cv_pred[3][i][0]+cv_pred[4][i][0])
    s=s/5
    re.append(s)
res = pd.DataFrame()
res['Id'] = list(test_id)
res['Score'] = list(re)  
res.to_csv('../stack_cache/stcking_ince___sfceshi_diejia_test_model1_no_shuffle.csv')

st_train1=pd.DataFrame()
st_train1['train1']=daset_blend_train[:,0]
st_train1['y']=train_data['Score']
st_train1.to_csv('../stack_cache/stcking_ince___sfceshi_diejia_train_model1_no_shuffle.csv')