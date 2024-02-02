#!/usr/bin/env python
# coding: utf-8

# ## Import the necessary packages



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Read the dataset
carsdf= pd.read_csv('D:\BUSINESS ANALYSIS\Python\Dataset\carprice.csv')
carsdf.head()


# ## Check number of rows and columns
carsdf.shape


#  ## Check the dataype, column names and constrains
carsdf.info()
carsdf.describe()
carsdf['wheelbase'].mean()


# The Average size of the wheelbase for casrs is 98.97
carsdf['fueltype'].value_counts()


# There are more number of gas vehicles than the diesel vehicles
carsdf['aspiration'].value_counts()


# Standard models are more in number than turbo
carsdf['symboling'].dtype
carsdf['symboling'].astype('category')
carsdf['symboling'].astype('category').dtype
carsdf['symboling'].astype('category').value_counts()


# ## Data Imputation- Null Values
carsdf.isnull().sum()
carsdf.isnull().sum().plot(kind='bar')
carsdf['enginelocation'].value_counts()


# More number of Front wheel drive vehicles
carsdf['enginelocation'].mode()[0]
carsdf['enginelocation'].fillna(value=carsdf['enginelocation'].mode()[0],inplace=True)
carsdf.isnull().sum().plot(kind='bar')
carsdf['peakrpm'].dtype
carsdf['peakrpm'].mean()
carsdf['peakrpm'].fillna(carsdf['peakrpm'].mean(),inplace=True)
carsdf['carwidth'].mean()
carsdf['carwidth'].fillna(carsdf['carwidth'].mean(),inplace=True)


# ## SimpleImputer
carsdf['wheelbase'].isnull().sum()
carsdf['wheelbase'].dtype


from sklearn.impute import SimpleImputer

median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
carsdf['wheelbase'] = median_imputer.fit_transform
carsdf['wheelbase'].isnull().sum()


# ## Visulization of the data - Check the distributions
carsdf['price']
carsdf['price'].dtype
plt.hist(carsdf['price'])


# We are observing skewness in the column(right skewed).
sns.distplot(carsdf['price'])


# ## Check for outliers
plt.boxplot(carsdf['price'])
sns.boxplot(carsdf['price'])


# ## Check the relationships b/w the columns
plt.scatter(carsdf['enginesize'],carsdf['price'])


# This is echibiting a linear relationship. As the enginge size is increasing, the price is also increasing.
sns.scatterplot (x = 'enginesize', y= 'price', data= carsdf)
carsdf.corr()
sns.heatmap(carsdf.corr())
sns.heatmap(carsdf.corr(),annot=True)
sns.countplot(x='fueltype', data=carsdf, hue='enginelocation')
carsdf['enginetype'].unique()
carsdf['carbody'].unique()
carsdf.groupby('carbody').agg({'price':'mean'}).sort_calues


# The average price of convertible is the highest of all car categories.
carsdf.groupby('carbody').agg({'price':'mean'}).sort_values('price',ascending=False)
pd.pivot_table(carsdf,index=['carbody','fueltype'],values='price',aggfunc='mean')
carsdf[carsdf.duplicated()]
carsdf[carsdf['carbody']=='hatchback']
plt.figure(figsize=(20,15))
sns.heatmap(carsdf.corr(),annot=True)






