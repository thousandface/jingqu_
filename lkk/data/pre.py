import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
# %matplotlib inline

import re
from sklearn.model_selection import StratifiedKFold,KFold
import random
import jieba



data_path = './train_first.csv'
df_0 = pd.read_csv(data_path,header = 0, encoding='utf8')
df = df_0.copy()
test_data_path = './predict_first.csv'
test_df_0 = pd.read_csv(test_data_path,header = 0, encoding='utf8')
test_df = test_df_0.copy()


#去除首尾空格
df['Discuss'] = df['Discuss'].apply(lambda x:x.replace(' ',''))
test_df['Discuss'] = test_df['Discuss'].apply(lambda x:x.replace(' ',''))
#停用词
stop_word = []
stop_words_path = 'stopWordList.txt'
with open(stop_words_path,encoding='utf8') as f:
    for line in f.readlines():
        stop_word.append(line.strip())
stop_word.append(' ')
def clean_str(stri):
    stri = re.sub(r'[a-zA-Z0-9]+','',stri)
    cut_str = jieba.cut(stri.strip()) #去除首尾空格
    list_str = [word for word in cut_str if word not in stop_word]
    stri = ' '.join(list_str)
    return stri
df['Discuss'] = df['Discuss'].map(lambda x : clean_str(x))
test_df['Discuss'] = test_df['Discuss'].map(lambda x : clean_str(x))
#空格填充
def fillnull(x):
    if x == '':
        return '空白'
    else:
        return x
df['Discuss'] = df['Discuss'].map(lambda x: fillnull(x))
test_df['Discuss'] = test_df['Discuss'].map(lambda x: fillnull(x))

df.to_csv('./train_pre.csv',index=False)
test_df.to_csv('./test_pre.csv',index=False)





data_path = './train_second.csv'
df_0 = pd.read_csv(data_path,header = 0, encoding='utf8')
df = df_0.copy()
test_data_path = './predict_second.csv'
test_df_0 = pd.read_csv(test_data_path,header = 0, encoding='utf8')
test_df = test_df_0.copy()


#去除首尾空格
df['Discuss'] = df['Discuss'].apply(lambda x:x.replace(' ',''))
test_df['Discuss'] = test_df['Discuss'].apply(lambda x:x.replace(' ',''))
#停用词
stop_word = []
stop_words_path = 'stopWordList.txt'
with open(stop_words_path,encoding='utf8') as f:
    for line in f.readlines():
        stop_word.append(line.strip())
stop_word.append(' ')
def clean_str(stri):
    stri = re.sub(r'[a-zA-Z0-9]+','',stri)
    cut_str = jieba.cut(stri.strip()) #去除首尾空格
    list_str = [word for word in cut_str if word not in stop_word]
    stri = ' '.join(list_str)
    return stri
df['Discuss'] = df['Discuss'].map(lambda x : clean_str(x))
test_df['Discuss'] = test_df['Discuss'].map(lambda x : clean_str(x))
#空格填充
def fillnull(x):
    if x == '':
        return '空白'
    else:
        return x
df['Discuss'] = df['Discuss'].map(lambda x: fillnull(x))
test_df['Discuss'] = test_df['Discuss'].map(lambda x: fillnull(x))

df.to_csv('./train_pre_B.csv',index=False)
test_df.to_csv('./test_pre_B.csv',index=False)