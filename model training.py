#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
import multiprocessing


# In[2]:


rate_list=['process','performance','people','parent','price']
rate_p=['process_p','performance_p','people_p','parent_p','price_p']
rate_n=['process_n','performance_n','people_n','parent_n','price_n']
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
    if (x<0.25):
        return 'Negative'
    elif (x<=0.75) & (x>=0.25):
        return 'Neutral'
    elif (x>0.75):
        return 'Positive'

def f2(x):
    if x=='Positive':
        return 1
    elif x=='Neutral':
        return 0.5
    elif x=='Negative':
        return 0
    

def f3(x):
    if x=='Gold':
        return 5
    elif x=='Silver':
        return 4
    elif x=='Bronze':
        return 3
    elif x=='Neutral':
        return 2
    elif x == 'Negative':
        return 1
    else:
        return None
    
def f4(x):
    if x<0.15:
        return 'Negative'
    elif 0.15<=x<0.7:
        return 'Neutral'
    elif 0.7<=x<0.85:
        return 'Bronze'
    elif 0.85<=x<0.95:
        return 'Silver'
    elif 0.95<=x:
        return 'Gold'   


def rank(data):
    a=data.sort_values('newrate')
    a.reset_index(drop=True,inplace=True)
    e=a.shape[0]
    a['rank']=a.index/e
    a['newclass']=a['rank'].apply(f4)
    return a
    
def acc(x,y):
    return sum(x==y)/len(y)

def smooth1(data):
    a=data.sort_values('date')
    a.reset_index(drop=True,inplace=True)
    l=data.shape[0]
    if l>2:
        for t in range(2,l):
            try:
                w=a.loc[t-2:t,rate_list]
                a.loc[t,rate_list]=w.mean()  
            except:
                print(t,l)
    return a

def smooth(data):
    a=data.sort_values('date')
    a.reset_index(drop=True, inplace=True)
    l=data.shape[0]
    if l>2:
        for t in range(2,l):
            w=a.loc[t-2:t,rate_list].mean()
            a.loc[t,rate_list]=w

    return a

def weigh(a,b):
    w=(a*b).sum()/b.sum()
    return w


d=['people','process']
b=['people_cl','process_cl']
c=['anapplrat','anaprorat']

def pp(data):
    for i in range(2):
        if data[c[i]].isnull().all():
            if data['tna'].notnull().all():
                data_=data[[d[i],'tna']]
                w=weigh(data[d[i]],data['tna'])
            else:
                w=data[d[i]].mean()

            data[b[i]]=f1(w)
        else:
            v=list(data[c[i]])
            j=0
            while v[j]!=v[j]:
                j=j+1
            data[b[i]]=v[j]
    return data

def parent(data):
    if data['anaparrat'].isnull().all():
        data['parent_cl']=data['parent'].apply(f1)
    else:
        p=list(data['anaparrat'])
        i=0
        while p[i]!=p[i]:
            i=i+1
        data['parent_cl']=p[i]
    return data
        

def lr(data):
    x=data[pillar_rate]
    y=data['rate']
    clf=LinearRegression()
    clf.fit(x,y)
    cov=clf.coef_
    para=clf.intercept_
    return(cov,clf.score(x,y),clf)

def apply_parallel(df_grouped,func):

    results = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name,group in df_grouped)

    return pd.concat(results)


# In[4]:


def g1(data,i):
    df=data[data[ana_list[i]].notnull()]
    a=sum(df[ana_list[i]]==df[pillar_class[i]])/df.shape[0]
    return a
def g2(data,i):
    df=data[data[quan_list[i]].notnull()]
    a=sum(df[quan_list[i]]==df[pillar_class[i]])/df.shape[0]
    return a
def acc(data,g):
    data=data[(data['date']>time_list[0])&(data['date']<time_list[-1]+100)]
    time_=[2017.7,2017.8,2017.9,2017.10,2017.11,2017.12,2018.1,2018.2,2018.3,2018.4,2018.5,2018.6,2018.7,2018.8,2018.9,2018.10,2018.11,2018.12,2019.1,2019.2,2019.3,2019.4,2019.5,2019.6]
    plt.xticks(range(24),time_,rotation=60)
    for i in range(5):
        a=data.groupby('date').apply(g,i)
        plt.plot(range(24),a,label =rate_list[i])
    plt.legend()#显示图例，如果注释改行，即使设置了图例仍然不显示
    plt.show()#显示图片，如果注释改行，即使设置了图片仍然不显示
    


# In[5]:


data=pd.read_csv('deal_1.csv')


# In[6]:


def h1(data,i):
    df=data[data[ana_list[i]].notnull()]
    a=sum(df[ana_list[i]]==df[rate_list[i]].apply(f1))/df.shape[0]
    return a
def h2(data,i):
    df=data[data[quan_list[i]].notnull()]
    a=sum(df[quan_list[i]]==df[rate_list[i]].apply(f1))/df.shape[0]
    return a


# In[7]:


acc(data,h1)


# In[8]:


acc(data,h2)


# In[ ]:


data0=data[data['date']>time_list[0]]
data1=data[data['date']<time_list[0]]
df_grouped_1 = data0.groupby('secid')
df1 = apply_parallel(df_grouped_1,smooth)
df1=pd.concat([data1,df1])
df1[pillar_class]=df1[rate_list].applymap(lambda x:f1(x))
df1['tna']=df1['tna'].replace(0,np.nan)


# In[ ]:


acc(df1,g1)


# In[ ]:


acc(df1,g2)


# In[ ]:


df_grouped_2 = df1.groupby('fundid')
df2= apply_parallel(df_grouped_2,pp)


# In[ ]:


acc(df2,g1)


# In[ ]:


acc(df2,g2)


# In[ ]:


df_grouped_3 = df2.groupby('Branding_Name')
df3= apply_parallel(df_grouped_3,parent)


# In[ ]:


acc(df3,g1)


# In[ ]:


acc(df3,g2)


# In[ ]:


df3[pillar_rate]=df3[pillar_class].applymap(lambda x:f2(x))
df3['rate']=df3['anarat'].apply(f3)
ana=df3[(df3['rate'].notnull())&(df3[ana_list].notnull().T.all())]
ana[pillar_rate]=ana[ana_list].applymap(lambda x:f2(x))
for time in time_list:
    ana_1=ana[ana['date']<time]
    (cov,score,clf)=lr(ana_1)
    X1=df3[(df3['date']>time)&(df3['date']<time+100)]
    X=X1[pillar_rate]
    X1['newrate']=clf.predict(X)
    try:
        reader=open('deal_2.csv')
        X1.to_csv('deal_2.csv',mode='a+',header=False)
    except:
        X1.to_csv('deal_2.csv')


# In[ ]:


df


# In[ ]:




