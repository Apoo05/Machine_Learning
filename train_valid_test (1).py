#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import pandas as pd
import numpy as np
iris=datasets.load_iris()


# In[2]:


iris.keys()


# In[3]:


iris=pd.DataFrame(data=np.c_[iris['data'],iris['target']],columns=iris['feature_names']+['target'])


# In[4]:


iris


# In[5]:


species=[]
for i in range(len(iris['target'])):
    if iris['target'][i]==0:
        species.append('Setosa')
    elif iris['target'][i]==1:
        species.append('Versicolor')
    else:
        species.append('Virginica')


# In[6]:


iris['species']=species


# In[7]:


iris


# In[8]:


iris.groupby('species').size()


# In[9]:


iris.describe()


# In[10]:


iris.isnull().sum()


# In[11]:


from sklearn.model_selection import train_test_split
x=iris.drop(['target','species'],axis=1)
y=iris['target']


# In[12]:


x_train,x_valid_test,y_train,y_valid_test=train_test_split(x,y,test_size=0.3,train_size=0.7,random_state=50)


# In[13]:


x_valid,x_test,y_valid,y_test=train_test_split(x_valid_test,y_valid_test,test_size=0.5)


# In[26]:


print(len(x_valid_test))


# In[14]:


print(len(x_valid))


# In[15]:


print(len(x_test))


# In[16]:


x_train.shape


# In[17]:


y_train.shape


# In[18]:


from sklearn.linear_model import LogisticRegression
log_model=LogisticRegression()
log_model.fit(x_train,y_train)
pred=log_model.predict(x_test)


# In[19]:


training_prediction=log_model.predict(x_train)
testing_prediction=log_model.predict(x_test)


# In[27]:


print(classification_report(y_train,training_prediction))


# In[28]:


from sklearn.metrics import classification_report,confusion_matrix
print("Classification report:")
print(classification_report(y_test,testing_prediction))


# In[23]:


from sklearn.linear_model import LogisticRegression
log_model=LogisticRegression()
log_model.fit(x_train,y_train)
pred=log_model.predict(x_valid)


# In[24]:


valid_prediction=log_model.predict(x_valid)


# In[29]:


from sklearn.metrics import classification_report,confusion_matrix
print("Classification report:")
print(classification_report(y_valid,valid_prediction))


# In[ ]:




