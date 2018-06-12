lkk代码运行环境介绍

软件： anaconda3+python3+tensorflow+keras+jieba+fasttext+glove+lightGBM+xgboost
硬件： 11g显存以上显卡，128g内存，ubuntu

词向量文件包括网上下载以及用比赛数据训练的，打包在stack/word2vec文件夹
（

由于中文百科的词向量过大，给出百度云的下载链接
下载链接：链接: https://pan.baidu.com/s/1o7MWrnc 密码:wzqv
将news_12g_baidubaike_20g_novel_90g_embedding_64.bin，
news12g_bdbk20g_nov90g_dim128.bin，
news_12g_baidubaike_20g_novel_90g_embedding_64.model.syn0.npy，
news12g_bdbk20g_nov90g_dim128.model.syn0.npy，
news_12g_baidubaike_20g_novel_90g_embedding_64.model
news12g_bdbk20g_nov90g_dim128.model
文件放入stack/word2vec文件夹

微信词向量的下载 链接: https://pan.baidu.com/s/1htC495U 密码: 4ff8

包含文件：word2vec_wx, word2vec_wx.syn1neg.npy, word2vec_wx.syn1.npy, word2vec_wx.wv.syn0.npy，4个文件都是Gensim加载模型所必需的。
将文件放入stack/word2vec文件夹

）

训练词向量代码在代码文件中。

data下包含了预处理代码以及原始数据跟预处理数据
stack下包含了比赛代码

---------------------------------
<1>
按顺序运行以下python文件
（注：下面代码生成的中间文件已经放在stack_cache文件中，可以不运行）
2_stack_nn_1.py
2_stack_nn_2.py
2_stack_nn_3.py
2_stack_nn_4.py
2_stack_nn_duo.py
2_stack_lgb.py
2_stack_ridge.py
2_stack_mlp_bayes.py
2_stack_fasttext.py
---------------------------------
<2>
运行完毕后，运行
2_stack_train.py

在stack_cache文件夹生成stack_prob.csv
---------------------------------
<3>
（注：下面代码生成的中间文件已经放在stack_cache文件中，可以不运行）
按顺序运行以下python文件
2_stack_lgb_init.py
2_stack_ridge_init.py
2_stack_mlp_bayes_init.py
2_stack_nn_init.py
---------------------------------
<4>
运行完毕后，运行
2_stack_train_init.py

在stack_cache文件夹生成stack_all_init.csv

