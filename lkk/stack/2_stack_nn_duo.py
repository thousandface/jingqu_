import keras
import keras.backend as k
from keras.models import Model
from keras.layers import Dense,Embedding,Input,Activation,BatchNormalization
from keras.layers import Conv1D
from keras.layers.core import Flatten
from keras.layers.pooling import MaxPooling1D

import pandas as pd
import numpy as np
#263
full=pd.read_csv('../data/full_tobe_classify_180316.csv',encoding='gbk')
train_index = [i for i in full[full.Score.notnull()].index]#这里的full就是full_tobe_classify_180316
test_index =  [i for i in full[full.Score.isnull()].index]
full_arr = np.load(r'../stack/word2vec/full_discuss_256_3-17_add_7feats.npy')
train = full_arr[train_index]
test = full_arr[test_index]
y = full.loc[train_index].Score.copy()
y = y.reset_index(drop=True)


import pandas as pd
data=pd.read_csv('../data/full_tobe_classify_180316.csv',encoding='gbk')
data['cutted_Dis']=data['cutted_Dis'].astype(str)
train_index = data[data.Score.notnull()].index
test_index = data[data.Score.isnull()].index

train_data=data.loc[[i for i in train_index]]
test_data=data.loc[[i for i in test_index]]
train_data.reset_index(drop=True,inplace=True)
test_data.reset_index(drop=True,inplace=True)

from sklearn.feature_extraction.text import TfidfVectorizer
def get_data():
#     train = pd.read_csv('train_first.csv')
#     test = pd.read_csv('predict_first.csv')
    train=train_data
    test=test_data
    data = pd.concat([train, test])
    print('train %s test %s'%(train.shape,test.shape))
    print('train columns',train.columns)
    return data,train.shape[0],train['Score'],test['Id']
def pre_process():
    data,nrw_train,y,test_id = get_data()
#     tf1 = TfidfVectorizer(max_features=1000,ngram_range=(1,2),analyzer='char') 
#     data1=tf1.fit_transform(data['Discuss'])
# #     tf1 = TfidfVectorizer(max_features=5000,ngram_range=(1,2),analyzer='char')     
# #     dd=vec.fit_transform(data['clearns'])
# #     discuss_tf=tfidf_transform.fit_transform(dd)
#     tf = TfidfVectorizer(max_features=3000,ngram_range=(1,2))
#     discuss_tf = tf.fit_transform(data['poseg'])
    tf2 = TfidfVectorizer(max_features=5000,ngram_range=(1,2))
    dis1=tf2.fit_transform(data['cutted_Dis'])
    #data = hstack((data1,discuss_tf,dis1)).tocsr()
    data=dis1
    return data[:nrw_train],data[nrw_train:],y,test_id

train_t,test_t,y,test_id = pre_process()

from keras.models import Model
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, AveragePooling2D, GlobalMaxPooling1D,LSTM
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
filter_sizes = [3,2,4,5,6,1,7,8,9]
num_filters = 128   
def get_model_textcnn():
    main_input = Input(shape=(maxlen,),name='main_input')
    embed_size =256
    x= Embedding(max_features, embed_size, weights=[embedding_matrix])(main_input)
    wv_input = Input(shape=(262,),name='wv_input')  
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
    

def escore(y,y_pred):
    mse = k.mean(k.square(y-y_pred))
    return 1 / (k.sqrt(mse) + 1)

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

from keras import callbacks


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

n_folds=2
# skf = list(StratifiedKFold(y, n_folds,shuffle=True,random_state=2018))
skf=KFold(X.shape[0],n_folds=5,shuffle=True,random_state=2018)
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
    #X_train1, X_validate1, label_train1, label_validate1 = X1[train_fold, :], X1[test_fold, :], y1[train_fold], y1[test_fold]
#     X_train = np.concatenate([X_train,X_train1])
# #     X_validate = np.concatenate([X_validate,X_validate1])
#     label_train=np.concatenate([label_train,label_train1])
#     label_validate=np.concatenate([label_validate,label_validate1])
    model=get_model_textcnn()

    earlystopping = callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=2)
    model.fit([X_train,train[train_fold],train_t[train_fold]],label_train,
                    batch_size=64,epochs=5,verbose=1,
                    
                    callbacks=[earlystopping])
#     model.fit(X_train,label_train,
#                     batch_size=64,epochs=3,verbose=1,
#                     callbacks=[earlystopping])
#     y_sub = model.predict(X_validate)
    y_sub = model.predict([X_validate,train[test_fold],train_t[test_fold]])
    print(xx_mse_s(label_validate, y_sub))
    daset_blend_train[test_fold,]=y_sub
#     daset_blend_test_0[:,i]=model.predict(test_hh)
    cv_pred.append(model.predict([test_hh,test,test_t]))
#     cv_pred.append(model.predict(test_hh))
        
       
        
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
res.to_csv('../stack_cache/st_duo_test.csv')

st_train1=pd.DataFrame()
st_train1['train1']=daset_blend_train[:,0]
st_train1['y']=train_data['Score']
st_train1.to_csv('../stack_cache/st_duo_train.csv')