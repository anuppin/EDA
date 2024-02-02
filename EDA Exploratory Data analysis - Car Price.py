#!/usr/bin/env python
# coding: utf-8

# ## Import the necessary packages

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Read the dataset

# In[3]:


carsdf= pd.read_csv('D:\BUSINESS ANALYSIS\Python\Dataset\carprice.csv')


# In[4]:


carsdf.head()


# ## Check number of rows and columns

# In[5]:


carsdf.shape


#  ## Check the dataype, column names and constrains

# In[6]:


carsdf.info()


# In[7]:


carsdf.describe()


# In[8]:


carsdf['wheelbase'].mean()


# The Average size of the wheelbase for casrs is 98.97

# In[9]:


carsdf['fueltype'].value_counts()


# There are more number of gas vehicles than the diesel vehicles

# In[10]:


carsdf['aspiration'].value_counts()


# Standard models are more in number than turbo

# In[11]:


carsdf['symboling'].dtype


# In[14]:


carsdf['symboling'].astype('category')


# In[15]:


carsdf['symboling'].astype('category').dtype


# In[16]:


carsdf['symboling'].astype('category').value_counts()


# ## Data Imputation- Null Values

# In[17]:


carsdf.isnull().sum()


# In[19]:


carsdf.isnull().sum().plot(kind='bar')


# In[21]:


carsdf['enginelocation'].value_counts()


# More number of Front wheel drive vehicles

# In[26]:


carsdf['enginelocation'].mode()[0]


# In[27]:


carsdf['enginelocation'].fillna(value=carsdf['enginelocation'].mode()[0],inplace=True)


# In[28]:


carsdf.isnull().sum().plot(kind='bar')


# In[29]:


carsdf['peakrpm'].dtype


# In[30]:


carsdf['peakrpm'].mean()


# In[32]:


carsdf['peakrpm'].fillna(carsdf['peakrpm'].mean(),inplace=True)


# In[33]:


carsdf['carwidth'].mean()


# In[34]:


carsdf['carwidth'].fillna(carsdf['carwidth'].mean(),inplace=True)


# ## SimpleImputer

# In[36]:


carsdf['wheelbase'].isnull().sum()


# In[37]:


carsdf['wheelbase'].dtype


# In[35]:


from sklearn.impute import SimpleImputer


# In[52]:


median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')


# In[55]:


carsdf['wheelbase'] = median_imputer.fit_transform


# In[56]:


carsdf['wheelbase'].isnull().sum()


# ## Visulization of the data - Check the distributions

# In[57]:


carsdf['price']


# In[58]:


carsdf['price'].dtype


# In[59]:


plt.hist(carsdf['price'])


# We are observing skewness in the column(right skewed).

# In[60]:


sns.distplot(carsdf['price'])


# ## Check for outliers

# In[61]:


plt.boxplot(carsdf['price'])


# In[63]:


sns.boxplot(carsdf['price'])


# ## Check the relationships b/w the columns

# In[65]:


plt.scatter(carsdf['enginesize'],carsdf['price'])


# This is echibiting a linear relationship. As the enginge size is increasing, the price is also increasing.

# In[68]:


sns.scatterplot (x = 'enginesize', y= 'price', data= carsdf)


# In[69]:


carsdf.corr()


# In[70]:


sns.heatmap(carsdf.corr())


# In[71]:


sns.heatmap(carsdf.corr(),annot=True)


# In[74]:


sns.countplot(x='fueltype', data=carsdf, hue='enginelocation')


# In[75]:


carsdf['enginetype'].unique()


# In[76]:


carsdf['carbody'].unique()


# In[77]:


carsdf.groupby('carbody').agg({'price':'mean'}).sort_calues


# The average price of convertible is the highest of all car categories.

# In[78]:


carsdf.groupby('carbody').agg({'price':'mean'}).sort_values('price',ascending=False)


# In[79]:


pd.pivot_table(carsdf,index=['carbody','fueltype'],values='price',aggfunc='mean')


# In[81]:


carsdf[carsdf.duplicated()]


# In[83]:


carsdf[carsdf['carbody']=='hatchback']


# In[87]:


plt.figure(figsize=(20,15))
sns.heatmap(carsdf.corr(),annot=True)


# In[ ]:




