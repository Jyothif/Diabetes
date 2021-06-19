#!/usr/bin/env python
# coding: utf-8

# ## Objective
# `To diagnostically predict whether or not a patient has diabetes,based on  certain dignostic measurements included in dataset.This dataset here is females atleast 21 years old.`

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("diabetes.csv")


# In[3]:


df.head(20)


# In[4]:


df.info()


# In[5]:


df = df.rename(columns = {"DiabetesPedigreeFunction": "DPedigreeF"})


# In[6]:


df.info()


# In[7]:


df_copy = df.copy(deep=True)


# In[8]:


df_copy.isnull().sum()


# In[9]:


df_copy[['Glucose','BloodPressure','SkinThickness','Insulin',]]   


# In[10]:


df_copy.describe()


# In[11]:



#replacing Nan values with Zero's

Zero_nt_accepted = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]

for col in Zero_nt_accepted:
    df_copy[col].replace(0,np.nan,inplace=True)
    
df_copy.head()


# In[12]:


df_copy.isnull().sum()


# In[13]:


df_copy['BloodPressure'].head(10)


# In[14]:


df_copy['BloodPressure'].fillna(df['BloodPressure'].mean())


# In[15]:


for col in Zero_nt_accepted:
    df_copy[col].replace(np.nan,df[col].mean(),inplace=True)


# In[16]:


df_copy.describe()


# # Model building

# In[18]:


from sklearn.model_selection import train_test_split
X=df_copy.drop(columns = "Outcome")
y = df_copy['Outcome']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state = 0)


# In[ ]:


X.head()


# In[19]:


y.head()


# In[20]:


#creating random forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train,y_train)


# In[22]:


#creating a pickle file for the classifier
import pickle
filename = 'diabetes.pkl'
pickle.dump(classifier,open(filename,'wb'))


# In[ ]:




