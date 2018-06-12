import pandas as pd
import numpy as np

def get_data():
    a = pd.read_csv('./lkk/data/full_same_dis_filled_180316.csv',encoding='GBK')
    b = pd.read_csv('./lkk/data/full_tobe_classify_180316.csv',encoding='GBK')
    b['cutted_Dis'] = b['cutted_Dis'].fillna("空白")
    c = pd.read_csv('./lkk/data/full_unknown_dis_filled_180316.csv',encoding='GBK')
    test_init = pd.read_csv('./lkk/data/predict_second.csv')
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

stack_prob = pd.read_csv('./lkk/stack_cache/stack_prob.csv')
stack_all_init = pd.read_csv('./lkk/stack_cache/stack_all_init.csv')



stack_mean = pd.read_csv('./laygin1/preds_final_stacking_4_0560.csv')
stack_mean = stack_mean[['Id','stack_mean']]

lg_24_prob = pd.read_csv('./laygin1/preds_final_stacking_24.csv')
lg_24_prob['lg_24_prob'] = (lg_24_prob['Pipeline'] + lg_24_prob['XGBRegressor'])/2
lg_24_prob = lg_24_prob[['Id','lg_24_prob']]



res = pd.merge(stack_prob,stack_all_init,on='Id',how='left')
res = pd.merge(res,stack_mean,on='Id',how='left')
res = pd.merge(res,lg_24_prob,on='Id',how='left')

res['prob'] = (res['stack_prob']*0.4+res['stack_all_init']*0.3 + res['stack_mean']*0.4 + res['lg_24_prob']*0.2)/1.3 # 

merge = pd.DataFrame(test_id)
merge = pd.merge(merge,res[['Id','prob']],how='left',on='Id')
merge = pd.merge(merge,same[['Id','Score']],how='left',on='Id')
merge = pd.merge(merge,unkown[['Id','Score']],how='left',on='Id',suffixes=['_same','_unkown'])
merge['final_score'] =  merge.prob.where(merge.prob.notnull(),merge.Score_same.where(merge.Score_same.notnull(),merge.Score_unkown))
merge['prob1'] = merge['final_score'].apply(lambda x:5 if x>=4.7 else x)
merge['prob1'] = merge['prob1'].apply(lambda x:1 if x<=1 else x)

rep = pd.read_csv('./lkk/stack_cache/guize.csv',encoding='gbk')
rep = rep[['Id','Score']]
rep.columns = ['Id','rep']

r = pd.merge(merge,rep,on="Id",how='left')

bb = stack_all_init.copy()
bb['mean_prob'] = bb['stack_all_init']
s = pd.merge(r,bb[['Id','mean_prob']],on='Id')
s['rep'] = s['rep'].fillna(s['prob'])
s['rep'] = s['rep'].fillna(s['mean_prob'])
s['rep1'] = s['rep'].apply(lambda x:5 if x>=4.7 else x)
s['rep1'] = s['rep1'].apply(lambda x:1 if x<=1 else x)
s[['Id', 'rep1']].to_csv('411_zhongji_shepipipiguai.csv',index=False,header=False) #