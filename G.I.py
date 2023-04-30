#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (FunctionTransformer,StandardScaler)
from sklearn.model_selection import (train_test_split,KFold,StratifiedKFold,cross_val_score,GridSearchCV,learning_curve,validation_curve)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier,GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[24]:


df=pd.read_csv("https://raw.githubusercontent.com/dsrscientist/dataset3/main/glass.csv")
df


# In[25]:


header_names=['ID_number','RI','Na','Mg','AI','SI','K','Ca','Ba','Fe','Types_of_glass']
df=pd.read_csv("https://raw.githubusercontent.com/dsrscientist/dataset3/main/glass.csv",header=None,names=header_names)
df.head()


# In[26]:


df.shape


# In[27]:


df.info()


# In[28]:


df.isnull().sum()


# In[29]:


df.columns


# In[30]:


df.dtypes


# In[31]:


df.describe()


# In[32]:


df['Types_of_glass'].value_counts()


# In[33]:


df1=df.drop(['ID_number'],axis=1)


# In[34]:


df1.shape


# In[35]:


features=df1.columns[:-1].tolist()
features


# In[36]:


attr=df1.columns
for i in attr:
    fig=plt.figure()
    ax=sns.boxplot(x='Types_of_glass',y=i,data=df1)


# In[37]:


plt.figure(figsize=(10,10))
sns.pairplot(df1[attr])
plt.show()


# In[38]:


df.info()


# In[40]:


from scipy import stats
z=abs(stats.zscore(df1))

np.where(z>3)

data=df1[(z<3).all(axis=1)]

data.shape
data


# In[41]:


label=['Types_of_glass']

x=data[features]

y=data[label]


# In[42]:


x.shape


# In[43]:


import warnings
warnings.filterwarnings('ignore')

x2=x.values

from matplotlib import pyplot as plt
import seaborn as sns
for i in range (1,9):
        sns.distplot(x2[i])
        plt.xlabel(features[i])
        plt.show()


# In[44]:


x2=pd.DataFrame(x)

plt.figure(figsize=(8,8))
sns.pairplot(data=x2)
plt.show()


# In[48]:


correlation=x.corr()
plt.figure(figsize=(15,15))
sns.heatmap(correlation,cbar=True,square=True,annot=True,fmt='.1f',annot_kws={'size':15},xticklabels=features,yticklabels=features)
plt.show()


# In[51]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)
x=pd.DataFrame(x)


# In[52]:


x.head()


# In[53]:


y.head()


# In[54]:


x2=x
from matplotlib import pyplot as plt
import seaborn as sns
for i in range (1,9):
    sns.distplot(x2[i])
    plt.xlabel(features[i])
    plt.show()


# In[61]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0,stratify=y)


# In[62]:


y_train=y_train.values.ravel()
y_test=y_test.values.ravel()


# In[67]:


print('shape of x_train='+ str(x_train.shape))
print('shape of x_test='+ str(x_test.shape))
print('shape of y_train='+ str(x_train.shape))
print('shape of y_test='+ str(x_test.shape))


# In[69]:


Scores=[]
scores=[]
for i in range (2,11):
    knn =KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    score=knn.score(x_test,y_test)
    scores.append(scores)
scores.append(max(scores))
print(knn.score(x_train,y_train))
print(Scores)


# In[72]:


for i in range(1):
    tree=DecisionTreeClassifier(random_state=0)
    tree.fit(x_train,y_train)
    score=tree.score(x_test,y_test)
    Scores.append(score)
print(tree.score(x_train,y_train))
print(Scores)


# In[74]:


for i in range(1):
    logistic=LogisticRegression(random_state=0,solver='lbfgs',multi_class='multinomial',max_iter=100)
    logistic.fit(x_train,y_train)
    score=logistic.score(x_test,y_test)
    scores.append(scores)
    
print(logistic.score(x_train,y_train))
print(Scores)


# In[ ]:


Range=[10,20,30,50,70,80,100,120]

for i in range(1):
    forest=RandomForestClassifier(criterion='gini',n_estimators=10,min_samples_leaf=1,min_samples_split=4,random_state=1,n_estimators=20)
    forest.fit(x_train,y_train)
    score=forest.score(x_test,y_test)
    scores.append(score)
    
print(forest.score(x_train,y_train))
print(scores)
print(Scores)


# In[ ]:




