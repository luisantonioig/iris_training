
# coding: utf-8

# In[16]:


import numpy as np
from sklearn import tree
import pandas as pd


# In[17]:


data = pd.read_csv('iris.csv', header=None)


# In[18]:


data.head()


# In[19]:


X = data[[0,1,2,3]]
y = data[[4]]


# In[20]:


X.head()


# In[21]:


y.head()


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


X_train, X_test,y_train,y_test = train_test_split(X,y, 
                                                 test_size=0.4,
                                                 random_state = 10)


# In[30]:


X_train.shape


# In[32]:


X_test.shape


# In[33]:


y_train.shape


# In[34]:


y_test.shape


# In[35]:


modelo = modelo = tree.DecisionTreeClassifier()


# In[37]:


modelo.fit(X_train, y_train)


# In[41]:


res = modelo.predict(X_test)


# In[40]:


y_test

