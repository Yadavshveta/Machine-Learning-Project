#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


credit_card_data = pd.read_csv('creditcard.csv')


# In[3]:


credit_card_data.head()


# In[4]:


credit_card_data.tail()


# In[5]:


credit_card_data.info()


# In[6]:


credit_card_data.isnull().sum()


# In[7]:


credit_card_data['Class'].value_counts()

This dataset is highly unbalaced0 --> Normal Transaction

1 --> fraudulent transaction
# In[8]:


legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[9]:


print(legit.shape)
print(fraud.shape)


# In[10]:


fraud.Amount.describe()


# In[11]:


credit_card_data.groupby('Class').mean()

Under-Sampling

Build a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions

Number of Fraudulent Transactions --> 492
# In[12]:


legit_sample = legit.sample(n=492)

Concatenating two DataFrames
# In[13]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)


# In[14]:


new_dataset.head()


# In[15]:


new_dataset.tail()


# In[16]:


new_dataset['Class'].value_counts()


# In[17]:


new_dataset.groupby('Class').mean()

Splitting the data into Features & Targets
# In[18]:


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']


# In[19]:


print(X)


# In[20]:


print(Y)

Split the data into Training data & Testing Data
# In[21]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[22]:


print(X.shape, X_train.shape, X_test.shape)

Model Training

Logistic Regression
# In[23]:


model = LogisticRegression()


# In[24]:


model.fit(X_train, Y_train)


# In[25]:


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[26]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[28]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[29]:


print('Accuracy score on Test Data : ', test_data_accuracy)


# In[ ]:




