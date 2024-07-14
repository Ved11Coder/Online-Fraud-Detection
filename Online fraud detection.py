#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np 


# In[3]:


data = pd.read_csv("credit card.csv")
print(data.head())


# In[4]:


print(pd.isnull(data).sum())


# In[5]:


print(data.type.value_counts())


# In[6]:


t1 = data["type"].value_counts()
transactions = t1.index
quantity = t1.values


# In[7]:


import plotly.express as px


# In[8]:


fig = px.pie(data,values=quantity,names=transactions,title="Distribution")
fig.show()


# In[9]:


print(data["isFraud"].corr(data["isFraud"]))
print(data["amount"].corr(data["isFraud"]))
print(data["isFlaggedFraud"].corr(data["isFraud"]))
print(data["step"].corr(data["isFraud"]))
print(data["oldbalanceOrg"].corr(data["isFraud"]))
print(data["newbalanceDest"].corr(data["isFraud"]))
print(data["oldbalanceDest"].corr(data["isFraud"]))
print(data["newbalanceOrig"].corr(data["isFraud"]))


# In[10]:


data["type"] = data["type"].map({"CASH_OUT":1,"PAYMENT":2,"CASH_IN":3,"TRANSFER":4,"DEBIT":5})
data["isFraud"] = data["isFraud"].map({0:"Not Fraud",1:"Fraud"})
data.head()


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


x = np.array(data[["type","amount","oldbalanceOrg","newbalanceOrig"]])
y = np.array(data[["isFraud"]])


# In[13]:


from sklearn.tree import DecisionTreeClassifier


# In[14]:


xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.10,random_state=42)
xtrain


# In[15]:


ytrain


# In[16]:


xtest


# In[17]:


ytest


# In[18]:


model = DecisionTreeClassifier()
model.fit(xtrain,ytrain)
print(model.score(xtest,ytest))


# In[19]:


features = np.array([[4, 250000, 250000, 10000]])
print(model.predict(features))


# In[ ]:




