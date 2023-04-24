#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
import multiprocessing


# In[9]:


rate_list=['process','performance','people','parent','price']
rate_p=['process_p','performance_p','people_p','parent_p','price_p']
rate_n=['process_n','performance_n','people_n','parent_n','price_n']
pillar_list=['process_','performance_','people_','parent_','price_']
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
time_=[2017.7,2017.8,2017.9,2017.10,2017.11,2017.12,2018.1,2018.2,2018.3,2018.4,2018.5,2018.6,2018.7,2018.8,2018.9,2018.10,2018.11,2018.12,2019.1,2019.2,2019.3,2019.4,2019.5,2019.6]
    


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


def weigh(a,b):
    w=(a*b).sum()/b.sum()
    return w

        

def lr(data):
    x=data[pillar_rate]
    y=data['rate']
    clf=LinearRegression(fit_intercept=False)
    clf.fit(x,y)
    cov=clf.coef_
    para=clf.intercept_
    return(cov,clf.score(x,y),clf)

def apply_parallel(df_grouped,func):

    results = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name,group in df_grouped)

    return pd.concat(results)


def compare(x1,x2):
    if (x1=='Negative') and (x2=='Neutral'):
        return 0.02
    elif (x1=='Neutral') and (x2=='Bronze'):
        return 0.02
    elif (x1=='Bronze') and (x2=='Silver'):
        return 0.01
    elif (x1=='Silver') and (x2=='Gold'):
        return 0.01
    elif (x2=='Negative') and (x1=='Neutral'):
        return 0.02
    elif (x2=='Neutral') and (x1=='Bronze'):
        return 0.02
    elif (x2=='Bronze') and (x1=='Silver'):
        return 0.01
    elif (x2=='Silver') and (x1=='Gold'):
        return 0.01
    else:
        return 0


def buffer(data):
    data=data.sort_values('date')
    data.reset_index(drop=True,inplace=True)
    for i in range(data.shape[0]-1):
        dis=compare(data.loc[i,'newclass'],data.loc[i+1,'newclass'])
        dis_1=data.loc[i+1,'rank']-data.loc[i,'rank']
        dis_1=abs(dis_1)
        dis=np.float(dis)
        if dis_1<=dis:
            data.loc[i+1,'newclass']=data.loc[i,'newclass']
    return data

def f5(x):
    c=[]
    for i in range(5):
        cla=cl[i]
        c.append(ana_[cla]['fundid'])
    if x<c[4]:
        return 'Negative'
    elif c[4]<=x<c[3]+c[4]:
        return 'Neutral'
    elif c[3]+c[4]<=x<c[3]+c[4]+c[2]:
        return 'Bronze'
    elif c[2]+c[3]+c[4]<=x<c[2]+c[3]+c[4]+c[1]:
        return 'Silver'
    elif c[2]+c[3]+c[4]+c[1]<=x:
        return 'Gold'
    
def rank2(data):
    a=data.sort_values('newrate')
    a.reset_index(drop=True,inplace=True)
    e=a.shape[0]
    a['rank']=a.index/e
    a['newclass']=a['rank'].apply(f5)
    return a

def count(data,label):
    a=data.groupby(label).count()/data.shape[0]
    return a

def full(data,label):
    select = lambda row: row[label] in cl
    df= data[data.apply(select, axis=1)]
    a=sum(df['newclass']==df[label])/df.shape[0]
    return a


# In[70]:


def recommend(x):
    if x in ['Gold','Silver','Bronze']:
        return 0  
    elif x in['Neutral','Negative']:
        return 1
    
def r1(data):
    r=sum(data['newclass'].apply(recommend)==data['anarat'].apply(recommend))/data.shape[0]
    return r

def r2(data):
    r=sum(data['newclass'].apply(recommend)==data['quarat'].apply(recommend))/data.shape[0]
    return r

def acc_1(data,f):
    plt.xticks(range(24),time_,rotation=60)
    for i in range(5):
        a=data.groupby('date').apply(f,i)
        plt.plot(range(24),a,label =rate_list[i])
    b=data.groupby('date').apply(full,'anarat')
    plt.plot(range(24),b,label ='rat')
    plt.legend()#显示图例，如果注释改行，即使设置了图例仍然不显示
    plt.show()#显示图片，如果注释改行，即使设置了图片仍然不显示
    
def acc_2(data,f):
    plt.xticks(range(24),time_,rotation=60)
    for i in range(5):
        a=data.groupby('date').apply(f,i)
        plt.plot(range(24),a,label =rate_list[i])
    b=data.groupby('date').apply(full,'quarat')
    plt.plot(range(24),b,label ='rat')
    plt.legend()#显示图例，如果注释改行，即使设置了图例仍然不显示
    plt.show()#显示图片，如果注释改行，即使设置了图片仍然不显示
    
def p1(data,i):
    df=data[data[ana_list[i]].notnull()]
    a=sum(df[ana_list[i]]==df[pillar_class[i]])/df.shape[0]
    return a

def p2(data,i):
    df=data[data[quan_list[i]].notnull()]
    a=sum(df[quan_list[i]]==df[pillar_class[i]])/df.shape[0]
    return a

def q1(data,i):
    df=data[data[ana_list[i]].notnull()]
    a=sum(df[ana_list[i]]==df[rate_list[i]].apply(f1))/df.shape[0]
    return a

def q2(data,i):
    df=data[data[quan_list[i]].notnull()]
    a=sum(df[quan_list[i]]==df[rate_list[i]].apply(f1))/df.shape[0]
    return a


# In[5]:


data=pd.read_csv('deal_2.csv')
df1=data[data['anarat'].notnull()]
df2=data[data['quarat'].notnull()]


# In[6]:


select = lambda row: row['anarat'] in cl
ana1= df1[df1.apply(select, axis=1)]
ana2=ana1[['anarat','fundid']].groupby('anarat').count()/ana1.shape[0]
ana_=ana2.T.to_dict()


# In[7]:


df_grouped_1 = df1.groupby('date')
df3= apply_parallel(df_grouped_1,rank2)
df_grouped_2 = df2.groupby('date')
df4= apply_parallel(df_grouped_2,rank)
df=pd.concat([df3,df4])


# In[66]:


acc_1(df3,p1)


# In[67]:


acc_2(df4,p2)


# In[27]:


df_ana=df3[['anarat','date','fundid','newclass']]
df_ana['r1']=df_ana['anarat'].apply(recommend)
plt.xticks(range(24),time_list,rotation=60)
labels=['recommend','unrecommend']
for i in range(2):
    data=df_ana[df_ana['r1']==i]
    r=data.groupby('date').apply(r1)
    plt.plot(range(24),r,label=labels[i])
plt.legend()#显示图例，如果注释改行，即使设置了图例仍然不显示
plt.show()


# In[28]:


df_qua=df4[['quarat','date','fundid','newclass']]
df_qua['r1']=df_qua['quarat'].apply(recommend)
plt.xticks(range(24),time_list,rotation=60)
labels=['recommend','unrecommend']
for i in range(2):
    data=df_qua[df_qua['r1']==i]
    r=data.groupby('date').apply(r2)
    plt.plot(range(24),r,label=labels[i])
plt.legend()#显示图例，如果注释改行，即使设置了图例仍然不显示
plt.show()


# In[15]:


df_grouped_3 = df.groupby('secid')
df5= apply_parallel(df_grouped_3,buffer)
df5.reset_index(drop=True,inplace=True)


# In[16]:


df5_ana=df5.loc[df5['anarat'].notnull(),['newclass','date','anarat','quarat','fundid','recommend']]
df5_quan=df5.loc[df5['quarat'].notnull(),['newclass','date','anarat','quarat','fundid','recommend']]
k1=df5_ana.groupby('date').apply(count,'newclass')
kk1=k1.groupby('newclass').mean()
k2=df5_quan.groupby('date').apply(count,'newclass')
kk2=k2.groupby('newclass').mean()

k3=df5_ana.groupby('date').apply(count,'anarat')
kk3=k3.groupby('anarat').mean()
k4=df5_quan.groupby('date').apply(count,'quarat')
kk4=k4.groupby('quarat').mean()


# In[29]:


df_ana=df5_ana[['anarat','date','fundid','newclass']]
df_ana['r1']=df_ana['anarat'].apply(recommend)
plt.xticks(range(24),time_list,rotation=60)
labels=['recommend','unrecommend']
for i in range(2):
    data=df_ana[df_ana['r1']==i]
    r=data.groupby('date').apply(r1)
    plt.plot(range(24),r,label=labels[i])
plt.legend()#显示图例，如果注释改行，即使设置了图例仍然不显示
plt.show()


# In[30]:


df_qua=df5_quan[['quarat','date','fundid','newclass']]
df_qua['r1']=df_qua['quarat'].apply(recommend)
plt.xticks(range(24),time_list,rotation=60)
labels=['recommend','unrecommend']
for i in range(2):
    data=df_qua[df_qua['r1']==i]
    r=data.groupby('date').apply(r2)
    plt.plot(range(24),r,label=labels[i])
plt.legend()#显示图例，如果注释改行，即使设置了图例仍然不显示
plt.show()


# In[19]:


kk1


# In[20]:


kk2


# In[21]:


kk3


# In[22]:


kk4


# In[23]:


def g(data):
    data1=data.groupby(['newclass','anarat']).count()/data.shape[0]
    return data1
data2=df5_ana.groupby('date').apply(g)
data3=data2.groupby(['newclass','anarat']).mean()


# In[34]:


def table(a):
    for i in range(5):
        c=[]
        for j in range(5):
            try:
                b=a[cl[i],cl[j]]
                b='%.5f' % b
            except:
                b=0
            c.append(b)
        print(cl[i]+'-new'+'&'+str(c[0])+'&'+str(c[1])+'&'+str(c[2])+'&'+str(c[3])+'&'+str(c[4])+'\\\ \hline')


# In[35]:


a=data3.loc[:,'fundid'].to_dict()


# In[36]:


table(a)


# In[37]:


def g(data):
    data1=data.groupby(['newclass','quarat']).count()/data.shape[0]
    return data1
data2=df5_quan.groupby('date').apply(g)
data3=data2.groupby(['newclass','quarat']).mean()


# In[38]:


b=data3.loc[:,'fundid'].to_dict()
table(b)


# In[39]:


a={}
b={}
def g1(data,j):
    return sum(data['anarat']==cl[j])/data.shape[0]

def g2(data,j):
    return sum(data['quarat']==cl[j])/data.shape[0]

for i in range(5):
    cla1=df5_ana.loc[df5_ana['newclass']==cl[i]]
    cla2=df5_quan.loc[df5_quan['newclass']==cl[i]]
    a[cl[i]]=[]
    b[cl[i]]=[]
    for j in range(5):
        a0=cla1.groupby('date').apply(g1,j)
        a1=a0.tolist()
        a2=np.mean(a1)
        a[cl[i]].append(a2)
        b0=cla2.groupby('date').apply(g2,j)
        b1=b0.tolist()
        b2=np.mean(b1)
        b[cl[i]].append(b2)


# In[42]:


def table_1(a):
    for i in range(5):
        c=[]
        for j in range(5):
            try:
                b=a[cl[i]][j]
                b='%.5f' % b
            except:
                b=0
            c.append(b)
        print(cl[i]+'-new'+'&'+str(c[0])+'&'+str(c[1])+'&'+str(c[2])+'&'+str(c[3])+'&'+str(c[4])+'\\\ \hline')


# In[43]:


table_1(a)


# In[44]:


table_1(b)


# In[68]:


acc_1(df5,p1)


# In[69]:


acc_2(df5,p2)


# In[48]:


acc_1(df5,q1)


# In[49]:


acc_2(df5,q2)


# In[50]:


data=df5[df5['date']>time_list[-1]]


# In[51]:


df1=data[data[ana_list].notnull().T.all()]
df2=data[data[quan_list].notnull().T.all()]


# In[52]:


def pi(data,pillar):
    acc=[]
    for i in range(5):
        a=sum(data[pillar[i]]==data[rate_list[i]].apply(f1))/data.shape[0]
        acc.append(a)
    return acc


# In[53]:


pi(df1,ana_list)


# In[54]:


pi(df2,quan_list)


# In[55]:


def pi(data,pillar):
    acc=[]
    for i in range(5):
        a=sum(data[pillar_class[i]]==data[pillar[i]])/data.shape[0]
        acc.append(a)
    return acc


# In[56]:


pi(df1,ana_list)


# In[57]:


pi(df2,quan_list)


# In[58]:


df5.region.unique()


# In[59]:


df5.groupby('region').count()/df5.shape[0]


# In[60]:


region=['United States','Canada','Luxembourg','United Kingdom','Ireland','Australia','India']


# In[61]:


def g1(data,i):
    df=data[data[ana_list[i]].notnull()]
    a=sum(df[ana_list[i]]==df[pillar_class[i]])/df.shape[0]
    return a
def g2(data,i):
    df=data[data[quan_list[i]].notnull()]
    a=sum(df[quan_list[i]]==df[pillar_class[i]])/df.shape[0]
    return a
def acc(data,g):
    time_=[2017.7,2017.8,2017.9,2017.10,2017.11,2017.12,2018.1,2018.2,2018.3,2018.4,2018.5,2018.6,2018.7,2018.8,2018.9,2018.10,2018.11,2018.12,2019.1,2019.2,2019.3,2019.4,2019.5,2019.6]

    plt.xticks(range(24),time_,rotation=60)
    for i in range(5):
        a=data.groupby('date').apply(g,i)
        plt.plot(range(24),a,label =rate_list[i])
    b=data.groupby('date').apply(full,'anarat')
    plt.plot(range(24),b,label ='rat')
    plt.legend()#显示图例，如果注释改行，即使设置了图例仍然不显示
    plt.show()#显示图片，如果注释改行，即使设置了图片仍然不显示
    


# In[62]:


for j in range(7):
    df=df5[df5['region']==region[j]]
    plt.title(region[j])
    acc(df,g1)


# In[63]:


def acc(data,g):
    time_=[2017.7,2017.8,2017.9,2017.10,2017.11,2017.12,2018.1,2018.2,2018.3,2018.4,2018.5,2018.6,2018.7,2018.8,2018.9,2018.10,2018.11,2018.12,2019.1,2019.2,2019.3,2019.4,2019.5,2019.6]

    plt.xticks(range(24),time_,rotation=60)
    for i in range(5):
        a=data.groupby('date').apply(g,i)
        plt.plot(range(24),a,label =rate_list[i])
    b=data.groupby('date').apply(full,'quarat')
    plt.plot(range(24),b,label ='rat')
    plt.legend()#显示图例，如果注释改行，即使设置了图例仍然不显示
    plt.show()#显示图片，如果注释改行，即使设置了图片仍然不显示


# In[64]:


for j in range(7):
    df=df5[df5['region']==region[j]]
    try:
        plt.title(region[j])
        acc(df,g2)
    except:
        print(str(region[j]))
        df=df[df[quan_list].notnull().T.all()]
        print(df.shape)


# In[74]:


df5.loc[0:2,rate_list].mean()


# In[77]:


df5.loc[2,rate_list]=df5.loc[0:2,rate_list].mean()


# In[78]:


df5.loc[0:2,rate_list]


# In[ ]:




