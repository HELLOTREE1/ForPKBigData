# -*- coding: utf-8 -*-
print("is coming!!!")

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

#preproccess
df_train=pd.read_csv('./v0.0/train_set.csv')
df_test=pd.read_csv('./v0.0/test_set.csv')
df_train.drop(columns=['article','id'],inplace=True)
df_test.drop(columns=['article'],inplace=True)

#feature
"""
将数据集的字符文本转换为数字向量
"""
vectorizer=CountVectorizer(ngram_range=(1,2),min_df=3,max_features=100000)
vectorizer.fit(df_train['word_seg'])
x_train=vectorizer.transform(df_train['word_seg'])
x_test=vectorizer.transform(df_test['word_seg'])
y_train=df_train['class']-1

# classifier
"""
训练一个分类器
"""
lg=LogisticRegression(C=4,dual=True)
lg.fit(x_train,y_train)

#predict
y_test=lg.predict(x_test)

#save results
df_test['class']=y_test.tolist()
df_test['class']=df_test['class']+1
df_result=df_test.loc[:,['id','class']]
df_result.to_csv('./results/result.csv',index=False)

print('over')