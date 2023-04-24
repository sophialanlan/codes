#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv('data2.csv',date_parser='date',encoding='gbk')


# In[3]:


data=data.dropna(axis=0)


# In[4]:


data


# In[5]:


data.index=pd.to_datetime(data.date)


# In[6]:


feature=data.columns[1:]


# In[7]:


df1=data[feature].pct_change()*100


# In[8]:


df2=df1.dropna(how='any',axis=0)


# In[9]:


data0=pd.read_csv('data1.csv',date_parser='Date')
data0=data0.dropna(axis=0)
data0.index=pd.to_datetime(data0.Date)
data0['return'] =data0['NAV_adj'].pct_change()*100


# In[10]:


model_data=pd.merge(df2,data0['return'],how='inner',left_index=True, right_index=True)


# In[11]:


model_data


# In[12]:


data1=model_data.to_period('Q')


# In[13]:


data2=data1[data1['return'].notnull()]


# In[14]:


data2


# In[15]:


l=data1.index.unique()


# In[16]:


l


# In[17]:


from sklearn.ensemble import RandomForestRegressor


# In[27]:


def pic(model,i):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    le=len(importances)
    plt.figure(figsize=(10,5))
    plt.rcParams['font.family'] = 'SimHei'
    plt.title('Feature Importance-'+str(l[i]))
    plt.bar(range(le),importances[indices],align='center')
    plt.xticks(range(le),feature[indices],rotation=90)
    plt.xlim([-1,le])
    plt.tight_layout()
    plt.savefig('picture/'+'行业'+str(l[i]) + "'.png")
    plt.show()


# In[28]:


def rf(i):
    train=data2.loc[l[i]]
    x=train[feature]
    y=train['return']
    randomforest = RandomForestRegressor(random_state=0, n_jobs=-1)
    model = randomforest.fit(x,y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(x.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feature[indices[f]], importances[indices[f]]))
    pic(model,i)


# In[29]:


for i in range(len(l)):
    rf(i)


# In[21]:


from sklearn import tree
model_decision_tree_regression = tree.DecisionTreeRegressor()
 
# 2.线性回归
from sklearn.linear_model import LinearRegression
model_linear_regression = LinearRegression()
 
# 3.SVM回归
from sklearn import svm
model_svm = svm.SVR()
 
# 4.kNN回归
from sklearn import neighbors
model_k_neighbor = neighbors.KNeighborsRegressor()
 
# 5.随机森林回归
from sklearn import ensemble
model_random_forest_regressor = ensemble.RandomForestRegressor(n_estimators=20)  # 使用20个决策树
 
# 6.Adaboost回归
from sklearn import ensemble
model_adaboost_regressor = ensemble.AdaBoostRegressor(n_estimators=50)  # 这里使用50个决策树
 
# 7.GBRT回归
from sklearn import ensemble
model_gradient_boosting_regressor = ensemble.GradientBoostingRegressor(n_estimators=100)  # 这里使用100个决策树
 
# 8.Bagging回归
from sklearn import ensemble
model_bagging_regressor = ensemble.BaggingRegressor()
 
# 9.ExtraTree极端随机数回归
from sklearn.tree import ExtraTreeRegressor
model_extra_tree_regressor = ExtraTreeRegressor()


# In[22]:


import numpy as np
import matplotlib.pyplot as plt

# 训练 Lasso 模型
from sklearn.linear_model import Lasso
alpha = 0.1
lasso = Lasso(alpha=alpha)
model= lasso.fit(data2[feature], data2['return'])
importance = model.coef_


# In[23]:


feature


# In[ ]:




