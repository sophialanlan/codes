#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
import multiprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# In[2]:


rate_list=['process','performance','people','parent','price']
rate_p=['process_p','performance_p','people_p','parent_p','price_p']
rate_n=['process_n','performance_n','people_n','parent_n','price_n']
pillar_p=['process_1','performance_1','people_1','parent_1','price_1']
pillar_n=['process_2','performance_2','people_2','parent_2','price_2']
ana_list=['anaprorat','anaperrat','anapplrat','anaparrat','anaprcrat']
quan_list=['quaprorat','quaperrat','quapplrat','quaparrat','quaprcrat']
cl=[ 'Gold','Silver','Bronze','Neutral','Negative']
pillar_class=['process_cl','performance_cl','people_cl','parent_cl','price_cl']
pillar_rate=['process_ra','performance_ra','people_ra','parent_ra','price_ra']

feature=[ 'star', 'star3y', 'star5y', 'star10y',
       'numinv3y', 'numinv5y', 'numinv10y', 'rkcat1m', 'rkcat3m', 'rkcat6m',
       'rkcatytd', 'rkcat1y', 'rkcat3y', 'rkcat5y', 'rkcat10y', 'rkcat15y',
        'ret', 'groret', 'tna', 'netflow', 'sale',
       'newsale',  'redemp', 'dividend', 'netflowest', 'mktappest',
       'income', 'carbscore','rkcarbscore', 
       'sustscore', 'numhold', 'tophold', 'groexp', 'netexp', 'netinc',
       'turnover', 'accfee', 'adminfee', 'advfee', 'auditfee', 'bodfee',
       'custfee', 'distfee', 'expwaiver', 'groinc', 'insurfee', 'interest',
       'legalfee', 'orgfee', 'otherfee', 'profee', 'regisfee', 'shrepfee',
       'tax', 'transfee', 'rkincret', 'rkcapret', 'famreten1y', 'famreten5y',
       'famtenure', 'famsuc3y', 'famsuc5y', 'famsuc10y', 'famnetexp']

feature1=[ 'star', 'star3y', 'star5y', 'star10y',
       'numinv3y', 'numinv5y', 'numinv10y', 'rkcat1m', 'rkcat3m', 'rkcat6m',
       'rkcatytd', 'rkcat1y', 'rkcat3y', 'rkcat5y', 'rkcat10y', 'rkcat15y',
        'ret', 'groret', 'tna', 'netflow', 'sale',
       'newsale',  'redemp', 'dividend', 'netflowest', 'mktappest',
       'income', 'carbscore','rkcarbscore', 
       'sustscore', 'groexp', 'netexp', 'netinc',
       'turnover', 'accfee', 'adminfee', 'advfee', 'auditfee', 'bodfee',
       'custfee', 'distfee', 'expwaiver', 'groinc', 'insurfee', 'interest',
       'legalfee', 'orgfee', 'otherfee', 'profee', 'regisfee', 'shrepfee',
       'tax', 'transfee', 'rkincret', 'rkcapret', 'famreten1y', 'famreten5y',
       'famtenure', 'famsuc3y', 'famsuc5y', 'famsuc10y', 'famnetexp']

year=[  'star', 'star3y', 'star5y', 'star10y', 'numinv3y', 'numinv5y', 'numinv10y','rkcat1y', 'rkcat3y', 'rkcat5y', 'rkcat10y', 'rkcat15y',
      'famsuc3y', 'famsuc5y', 'famsuc10y', 'famreten1y', 'famreten5y']

month=[ 'rkcat1m', 'rkcat3m', 'rkcat6m',
       'rkcatytd',
        'ret', 'groret', 'tna', 'netflow', 'sale',
       'newsale',  'redemp', 'dividend', 'netflowest', 'mktappest',
       'income', 'carbscore','rkcarbscore', 
       'sustscore', 'numhold', 'tophold', 'groexp', 'netexp', 'netinc',
       'turnover', 'accfee', 'adminfee', 'advfee', 'auditfee', 'bodfee',
       'custfee', 'distfee', 'expwaiver', 'groinc', 'insurfee', 'interest',
       'legalfee', 'orgfee', 'otherfee', 'profee', 'regisfee', 'shrepfee',
       'tax', 'transfee', 'rkincret', 'rkcapret',
       'famtenure',  'famnetexp']

var=['secid', 'fundid', 'date', 'star', 'star3y', 'star5y', 'star10y',
       'numinv3y', 'numinv5y', 'numinv10y', 'rkcat1m', 'rkcat3m', 'rkcat6m',
       'rkcatytd', 'rkcat1y', 'rkcat3y', 'rkcat5y', 'rkcat10y', 'rkcat15y',
       'anarat', 'anaprorat', 'anaperrat', 'anapplrat', 'anaparrat',
       'anaprcrat', 'quarat', 'quaprorat', 'quaperrat', 'quapplrat',
       'quaparrat', 'quaprcrat', 'ret', 'groret', 'tna', 'netflow', 'sale',
       'newsale', 'othersale', 'redemp', 'dividend', 'netflowest', 'mktappest',
       'income', 'carbscore', 'carbclass', 'rkcarbscore', 
       'sustscore', 'numhold', 'tophold', 'groexp', 'netexp', 'netinc',
       'turnover', 'accfee', 'adminfee', 'advfee', 'auditfee', 'bodfee',
       'custfee', 'distfee', 'expwaiver', 'groinc', 'insurfee', 'interest',
       'legalfee', 'orgfee', 'otherfee', 'profee', 'regisfee', 'shrepfee',
       'tax', 'transfee', 'rkincret', 'rkcapret', 'famreten1y', 'famreten5y',
       'famtenure', 'famsuc3y', 'famsuc5y', 'famsuc10y', 'famnetexp']


cat=[ 'star', 'star3y', 'star5y', 'star10y',
       'numinv3y', 'numinv5y', 'numinv10y', 'rkcat1m', 'rkcat3m', 'rkcat6m',
       'rkcatytd', 'rkcat1y', 'rkcat3y', 'rkcat5y', 'rkcat10y', 'rkcat15y','ret', 'groret', 'tna', 'netflow', 'sale',
       'newsale', 'othersale', 'redemp', 'dividend', 'netflowest', 'mktappest',
       'income', 'carbscore',  'rkcarbscore',
       'sustscore', 'groexp', 'netexp', 'netinc',
       'turnover', 'accfee', 'adminfee', 'advfee', 'auditfee', 'bodfee',
       'custfee', 'distfee', 'expwaiver', 'groinc', 'insurfee', 'interest',
       'legalfee', 'orgfee', 'otherfee', 'profee', 'regisfee', 'shrepfee',
       'tax', 'transfee', 'rkincret', 'rkcapret',  'famsuc3y', 'famsuc5y', 'famsuc10y', 'famnetexp']

time_list=[20170700,20170800,20170900,20171000,20171100,20171200,20180100,20180200,20180300,20180400,
           20180500,20180600,20180700,20180800,20180900,20181000,20181100,20181200,20190100,20190200,
           20190300,20190400,20190500,20190600]


# In[3]:


def f1(x):
    if x in ['Positive','High','Above Average']:
        a=1
    elif x in ['Negative','Low','Below Average','Average','Neutral']:
        a=0
    else:
        a=None
    return a

def f2(x):
    if x in ['Negative','Low','Below Average']:
        a=1
    elif x in ['Positive','High','Above Average','Average','Neutral']:
        a=0
    else:
        a=None
    return a


def trans(df):
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def impute_m(data):
    data.reset_index(drop=True, inplace=True)
    for col in month:
        data[col]=data[col].fillna(data[col].median())
    return data

def impute_y(data):
    data.reset_index(drop=True, inplace=True)
    for col in year:
        data[col].fillna(data[col].median(),inplace=True)
    return data

def impute(data):
    data.reset_index(drop=True, inplace=True)
    for col in feature:
        data[col].fillna(data[col].median(),inplace=True)
    return data


def normal(data):
    data.reset_index(drop=True, inplace=True)
    stdsc=StandardScaler()
    data[feature]=stdsc.fit_transform(data[feature])
    return data

def inter(x_train,x_test):
    a=x_train.dropna(axis=1,how='all').columns
    b=x_test.dropna(axis=1,how='all').columns
    c=list(a.intersection(b))
    return c

def apply_parallel(df_grouped,func):

    results = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name,group in df_grouped)

    return pd.concat(results)


    
def evaluate_acc(model, x, y):
    return sum(model.predict(x)==y)/len(y)


def cate(data):
    for col in cat:
        data[col]=data[col]-data[col].mean()
    return data


# In[4]:


def handle(data):
    for k in range(5):
        data[rate_p[k]]=data[ana_list[k]].apply(lambda x:f1(x))
#        data_out[rate_p[k]]=data_out[ana_list[k]].apply(lambda x:f1(x))
        data[rate_n[k]]=data[ana_list[k]].apply(lambda x:f2(x))
#        data_out[rate_n[k]]=data_out[ana_list[k]].apply(lambda x:f2(x))

    data_1=data.groupby(['Morningstar_Category','date'],as_index=False).apply(cate)
    data_1.reset_index(drop=True,inplace=True)

    data_2=data_1.groupby(['region','year'],as_index=False).apply(impute_y)
    data_2.reset_index(drop=True,inplace=True)
    
    data_3=data_2.groupby(['region','date'],as_index=False).apply(impute_m)
    data_3.reset_index(drop=True,inplace=True)
    
    if data_3[feature].isnull().any().any():
        data_3=data_3.groupby('region').apply(impute)
        data_3.reset_index(drop=True,inplace=True)
        data_3=impute(data_3)
        
    data_4=data_3.groupby('region',as_index=False).apply(normal)
    data_4.reset_index(drop=True,inplace=True)
    
    return data_4
    


# In[5]:


data=pd.read_csv('deal_0.csv')


# In[6]:


df1=handle(data)


# In[7]:


p_acc={}
n_acc={}
for j in range(5):
    pillar=rate_list[j]
    df1[pillar]=0
    df1[pillar_p[j]]=0
    df1[pillar_n[j]]=0
    p_acc[pillar]=[]
    n_acc[pillar]=[]
df2=df1.loc[df1['date']<time_list[0]]
df2.to_csv('deal_1.csv')
#df_list.append(df2)
for time in time_list:
    train=df1.loc[(df1['date']<time)&(df1[ana_list].notnull().T.all())]
    test=df1[(df1['date']>time)&(df1['date']<time+100)]
    x = train[feature]
    x=x.dropna(axis=1,how='all')
    y_1=train[rate_p]
    y_2=train[rate_n]
    model_1=[]
    model_2=[]
    for k in range(5):
        clf1=RandomForestClassifier(n_estimators = 20,max_features=10,n_jobs=-1)
        clf1.fit(x, y_1.iloc[:,k])
        clf2=RandomForestClassifier(n_estimators = 20,max_features=10,n_jobs=-1)
        clf2.fit(x, y_2.iloc[:,k])
        model_1.append(clf1)
        model_2.append(clf2)

    for i in range(5):
        pillar=rate_list[i]
        clf1=model_1[i]
        clf2=model_2[i]

        x_quan,y_quan_p,y_quan_n=test[feature],test[rate_p[i]],test[rate_n[i]]

        ana_df1=test[test[ana_list].notnull().T.all()]

        x_ana,y_ana_p,y_ana_n=ana_df1[feature],ana_df1[ana_list[i]].apply(f1),ana_df1[ana_list[i]].apply(f2)

        acc_p=evaluate_acc(clf1,x_ana, y_ana_p)
        acc_n=evaluate_acc(clf2,x_ana, y_ana_n)

        p_acc[pillar].append(acc_p)
        n_acc[pillar].append(acc_n)
        
        test[pillar_p[i]]=clf1.predict(x_quan)
        test[pillar_n[i]]=clf2.predict(x_quan)

        quan_prob1 = clf1.predict_proba(x_quan)
        quan_prob2=clf2.predict_proba(x_quan)
        test[pillar]=(quan_prob1[:,1]+1-quan_prob2[:,1])/2
    
    test.to_csv('deal_1.csv',mode='a+',header=False)


# In[8]:


from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

def tiaocan(x_train,y_train):
    rfc = RandomForestClassifier()


    tuned_parameter = [{'min_samples_leaf':[10,30,50], 'n_estimators':[10,30,50]}]


    clf = GridSearchCV(estimator=rfc,param_grid=tuned_parameter, cv=5, n_jobs=-1)


    clf.fit(x_train, y_train)

    print('Best parameters:')

    print(clf.best_params_)


# In[9]:


p_acc={}
n_acc={}
for pillar in rate_list:
    df1[pillar]=0
    p_acc[pillar]=[]
    n_acc[pillar]=[]
df2=df1.loc[df1['date']<time_list[0]]

time = time_list[0]
train=df1.loc[(df1['date']<time)&(df1[ana_list].notnull().T.all())]
test=df1[(df1['date']>time)&(df1['date']<time+100)]
x = train[feature]
x=x.dropna(axis=1,how='all')
y_1=train[rate_p]
y_2=train[rate_n]
model_1=[]
model_2=[]
for k in range(5):
    tiaocan(x,y_1)
    tiaocan(x,y_2)


# In[9]:


np.save('p',p_acc)


# In[10]:


np.save('n',n_acc)


# In[8]:


n_acc


# In[9]:


p_acc


# In[10]:


plt.xticks(range(24),time_list,rotation=90)
for i in range(5):
    plt.plot(range(24),p_acc[rate_list[i]],label =rate_list[i])
plt.legend()#显示图例，如果注释改行，即使设置了图例仍然不显示
plt.show()#显示图片，如果注释改行，即使设置了图片仍然不显示


# In[11]:


plt.xticks(range(24),time_list,rotation=90)
for i in range(5):
    plt.plot(range(24),n_acc[rate_list[i]],label =rate_list[i])
plt.legend()#显示图例，如果注释改行，即使设置了图例仍然不显示
plt.show()#显示图片，如果注释改行，即使设置了图片仍然不显示


# In[17]:


def g1(x):
    if (x>0.75):
        return 1
    else:
        return 0
    
def g2(x):
    if (x<0.25):
        return 1
    else:
        return 0

def g(data,i):
    a=sum(data[pillar_p[i]]==data[rate_list[i]].apply(g1))/data.shape[0]
    b=sum(data[pillar_n[i]]==data[rate_list[i]].apply(g2))/data.shape[0]
    return (a,b)


# In[13]:


df=pd.read_csv('deal_1.csv')


# In[18]:


p_acc_1={}
n_acc_1={}
for j in range(5):
    pillar=rate_list[j]
    p_acc_1[pillar]=[]
    n_acc_1[pillar]=[]
for time in time_list:
    data=df[(df['date']>time)&(df1['date']<time+100)]
    for i in range(5):
        pillar=rate_list[i]

        acc=g(data,i)
        
        p_acc_1[pillar].append(acc[0])
        n_acc_1[pillar].append(acc[1])
        


# In[19]:


p_acc_1


# In[20]:


n_acc_1


# In[22]:


plt.xticks(range(24),time_list,rotation=90)
for i in range(5):
    plt.plot(range(24),p_acc_1[rate_list[i]],label =rate_list[i])
plt.legend()#显示图例，如果注释改行，即使设置了图例仍然不显示
plt.show()#显示图片，如果注释改行，即使设置了图片仍然不显示


# In[23]:


plt.xticks(range(24),time_list,rotation=90)
for i in range(5):
    plt.plot(range(24),n_acc_1[rate_list[i]],label =rate_list[i])
plt.legend()#显示图例，如果注释改行，即使设置了图例仍然不显示
plt.show()#显示图片，如果注释改行，即使设置了图片仍然不显示


# In[1]:





# In[ ]:


d

