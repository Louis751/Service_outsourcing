#!/usr/bin/env python
# coding: utf-8

# # 重要版本！！！

# ## 导入数据

# In[1]:


import jieba
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


# In[2]:


def format_output(_str):
    print(">>> " + _str)
    return None


# In[3]:


# 导入数据集
format_output("正在导入数据...")
train_tsv = "./data/train.tsv"
# test_tsv = "./data.test.tsv"
train_df = pd.read_csv(train_tsv,sep='\t')
# test_df = pd.read_csv(test_tsv,sep='\t')
format_output("加载数据集已成功！")


# In[4]:


# 初步分列
item_name = train_df['ITEM_NAME']
tri_type = train_df['TYPE']


# In[5]:


ttest = train_df.head()


# In[6]:


format_output("正在基于‘TYPE’列创建order-fimaly-species分列，请稍后...")
# 将原始分列标签按照order-fimaly-species分列，并保存数据
temp_df = train_df['TYPE'][:].str.split('--',expand=True)
train_ndf = train_df.drop('TYPE',axis=1).join(temp_df)
train_ndf.rename(columns={0:'ORDER',1:'FAMILY',2:'SPECIES'},inplace=True)
format_output("创建分列成功！")
train_ndf.to_csv("./data/tri_type.csv",index=False,encoding="utf_8_sig")
# train_ndf.info()
format_output("调整分列后的数据集已导出！")


# In[7]:


# 单独抽出分列
order_ = train_ndf['ORDER']
fimaly_ = train_ndf['FAMILY']
species_ = train_ndf['SPECIES']
# print(order_)


# In[8]:


# 各列、各类别统计数目
order_counts = order_.value_counts()
fimaly_counts = fimaly_.value_counts()
species_counts = species_.value_counts()
# print(order_counts)


# In[9]:


# 训练集和测试集的科学比例拆分
x_train,x_test,y_train,y_test = train_test_split(item_name,order_,test_size=0.25)


# ## 降噪处理

# In[10]:


# 消除噪点，即数字、标点符号、特殊字符等非语义类文本
from string import printable
# printable = digits + ascii_letters + punctuation + whitespace

def remove_noise(_str):
    noise = printable + "★【】！（）：；“”‘’《》，。、？"
    return _str.translate(str.maketrans('','',noise))


# ## jieba分词

# In[11]:


# jieba分词函数
def cutword(dataSet):
    format_output("正在切分字段，这可能需要一段时间！")
    dataSetcw = [""] * len(dataSet)
    cw = lambda x: list(jieba.cut(x,cut_all=False))
    for i in range(len(dataSet)):
        dataSetcw[i] = ' '.join(cw(dataSet.iloc[i]))
        dataSetcw[i] = remove_noise(dataSetcw[i])
    format_output("切分完成！")
    return dataSetcw


# In[12]:


print(cutword(ttest['ITEM_NAME'])[1])


# In[15]:


item_name_cw = cutword(item_name)


# In[17]:


# 将文本中的词语转换为词频矩阵
cv = CountVectorizer(min_df=1)
# 计算各个词语出现的次数
cv_fit = cv.fit_transform(item_name_cw[:])

# set of words（SOW） 词集模型 - 获取词袋中所有文本关键词
print("打印所有的特征名称")
print(cv.get_feature_names())

# # bag of words（BOW） 词袋模型
# print("打印整个文本矩")
# print(cv_fit.toarray())
# temp = cv_fit.toarray()
# # print("矩阵大小：",temp.shape)
# print("打印所有的列相加(统计特征名称出现次数)")
# print(cv_fit.toarray().sum(axis=0))
# print("打印所有的行相加")
# print(cv_fit.toarray().sum(axis=1))


# In[ ]:


lose_num = cutword(ttest['ITEM_NAME'])[1]
lose_num = remove_noise(lose_num)
lose_num


# In[ ]:


type(lose_num)


# In[ ]:




