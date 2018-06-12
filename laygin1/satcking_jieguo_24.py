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

full = pd.read_csv(r'full_tobe_classify_180316.csv',engine='python')
np.random.seed(2018)

train_index = [i for i in full[full.Score.notnull()].index]
np.random.shuffle(train_index)
test_index = [i for i in full[full.Score.isnull()].index]
y = full.loc[train_index].Score.copy()
y = y.reset_index(drop=True)

sample = pd.read_csv(r'YNU.EDU2018-ScenicWord_submite_sample2.csv',names=['Id','Score'])
full_same_dis = pd.read_csv(r'full_same_dis_filled_180316.csv',engine='python')
full_unknown_dis = pd.read_csv(r'full_unknown_dis_filled_180316.csv',engine='python')
dataset_train_blends=pd.read_csv('./full_dataset_train_blends.csv')
models_pred=pd.read_csv('./full_models_pred.csv')
n_folds = 5
def mse_cv(model,X,y):
    kf = KFold(n_folds,shuffle=True,random_state=2017).get_n_splits(X)
    mse = -cross_val_score(model,X,y,scoring='neg_mean_squared_error',cv=kf)
    metric = 1 /(np.sqrt(mse) +1)  
    return '{}+{:.4}'.format(metric.mean(),metric.std())
def eval_score(y,y_pred):
    return 1 / (np.sqrt(metrics.mean_squared_error(y,y_pred)) + 1)
def ridge(X_train,y_train):
    ridge = RidgeCV(alphas = [0.001,0.003,0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60,80,100,120])
    ridge.fit(X_train, y_train)
    alpha = ridge.alpha_
    ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85,
                              alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                              alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],
                    cv = 10)
    ridge.fit(X_train, y_train)
    coefs = ridge.coef_
    print("Ridge picked " ,sum(coefs != 0)," features and eliminated the other " ,sum(coefs == 0), " features")
    ridge = make_pipeline(RobustScaler(),ridge)
    ridge.fit(X_train, y_train)
    mse_cv_score = mse_cv(ridge,X_train,y_train)
    print("Ridge MSE on Training set:\t", mse_cv_score,'\n')
    return ridge,mse_cv_score

meta_model_ridge,_ = ridge(dataset_train_blends.values,y)
meta_model_xgb = XGBRegressor(n_jobs=-1)
meta_model_xgb = meta_model_xgb.fit(dataset_train_blends.values,y)
mse_cv(meta_model_xgb,dataset_train_blends.values,y)
pred_cols = [i for i in models_pred.columns if i != 'Id']
meta_modedls = [meta_model_ridge,meta_model_xgb]
preds_stacking = pd.DataFrame({'Id':models_pred.Id})
for meta in meta_modedls:
    pred = meta.predict(models_pred[pred_cols].values)
    preds_stacking[meta.__class__.__name__] = pred
preds_final = pd.DataFrame({'Id':sample.Id})

for pred in [i for i in preds_stacking.columns if i != 'Id']:
    merged_pred = pd.merge(sample[['Id']],preds_stacking[['Id',pred]],how='left',on='Id').\
    merge(full_same_dis[['Id','Score']],on='Id',how='left').\
    merge(full_unknown_dis[['Id','Score']],on='Id',how='left',suffixes=['_same','_unknow'])

    merged_pred[pred] = merged_pred[pred].where(merged_pred[pred].notnull(),
                                                                    merged_pred.Score_same.where(merged_pred.Score_same.notnull(),
                                                                                                 merged_pred.Score_unknow))
    new_sample = merged_pred[['Id',pred]]
    preds_final[pred] = pd.merge(preds_final[['Id']],new_sample[['Id',pred]],how='left',on='Id',)[pred]


preds_final['stack_mean'] = preds_final[['Pipeline','XGBRegressor']].mean(axis=1)

preds_final.to_csv(r'./preds_final_stacking_24.csv',index=None)

e = time.ctime()
