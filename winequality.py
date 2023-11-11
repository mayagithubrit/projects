#!/usr/bin/env python
# coding: utf-8

# In[16]:


# importing the dependencies
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.linear_model import LinearRegression


# In[17]:


# data collection


# In[18]:


dataset=pd.read_csv("C:\\Users\\Sachin\\Downloads\\winequality-red.csv",sep=';')


# In[19]:


dataset


# In[20]:


dataset.head()


# In[21]:


#shape of datasets
print("Shape of our dataset of Red-Wine:{s}".format(s = data.shape))
print("Colum headers/names:{s}".format(s = list(data)))


# In[22]:


dataset.info()


# In[23]:


dataset.describe()


# In[24]:


dataset['quality'].unique()


# In[25]:


dataset.quality.value_counts().sort_index()


# In[31]:


sns.countplot(x = 'quality', data=dataset)


# In[32]:


dataset['alcohol'].describe()


# In[34]:


dataset['sulphates'].describe()


# In[35]:


dataset['citric acid'].describe()


# In[37]:


dataset['residual sugar'].describe()


# In[53]:


Q1 = dataset.quantile(0.25)
Q3 = dataset.quantile(0.75)
IQR = Q3-Q1
print(IQR)


# In[ ]:


# the data points where we have false means these values are valid whereas true indicates presence of an outlier.


# In[60]:


print(dataset < (Q1 - 1.5 * IQR))  or (dataset > (Q3 + 1.5 * IQR))


# In[65]:


dataset_out = dataset[~((dataset<(Q1 - 1.5 * IQR)) |(dataset > (Q3 + 1.5 * IQR))).any(axis=1)]
dataset_out.shape


# In[66]:


dataset_out


# In[68]:


correlations = dataset_out.corr()['quality'].drop('quality')
print(correlations)


# In[69]:


sns.heatmap(dataset.corr())
plt.show()


# In[72]:


#impact of various factor an quality
correlations.sort_values(ascending=False)


# In[83]:


def  get_features(correlation_threshold):
    abs_corrs = correlations.abs()
    high_correlations=abs_corrs[abs_corrs> correlation_threshold].index.values.tolist()
    return high_correlations


# In[84]:


#taking features with correlation more than 0.05 as input x and quality as target variables y
features = get_features(0.05)
print(features)
x = dataset_out[features]
y = dataset_out['quality']


# In[85]:


#to finding the no of outliers we have in our dataset with properties
bx=sns.boxplot(x='quality', y='alcohol',data=dataset)
bx.set(xlabel ='Quality',ylabel='Alcohol',title='Alcohol% in different samples')


# In[86]:


bx=sns.boxplot(x='quality', y='citric acid',data=dataset)
bx.set(xlabel ='Quality',ylabel='Citric Acid',title='Citric Acid % in different samples')


# In[87]:


bx=sns.boxplot(x='quality', y='fixed acidity',data=dataset)
bx.set(xlabel ='Quality',ylabel='Fixed Acidity',title='Fixed Acidity % in different samples')


# In[88]:


x


# In[89]:


y


# In[96]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=3)


# In[97]:


x_train.shape


# In[98]:


x_test.shape


# In[99]:


y_train.shape


# In[100]:


y_test.shape


# In[103]:


# fitting linear regression to training data
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[104]:


# to retrieve the intercept
regressor.intercept_


# In[105]:


# this gives the coefficints of the 10 featuers selected above.
regressor.coef_


# In[106]:


train_pred=regressor.predict(x_train)
train_pred


# In[107]:


test_pred = regressor.predict(x_test)
test_pred


# In[108]:


y


# In[109]:


train_rmse = metrics.mean_squared_error(train_pred,y_train)**0.5
train_rmse


# In[110]:


test_rmse = metrics.mean_squared_error(test_pred,y_test)**0.5
test_rmse


# In[111]:


# rounding off the predicted values for test set
predicted_data = np.round(test_pred)
predicted_data


# In[112]:


print('mean absolute error:', metrics.mean_absolute_error(y_test,test_pred))
print('mean squared error:',metrics.mean_squared_error(y_test,test_pred))
rmse= np.sqrt(metrics.mean_squared_error(y_test,test_pred))
print('root mean squared error:',rmse)


# In[113]:


from sklearn.metrics import r2_score
r2_score(y_test,test_pred)


# In[114]:


coeffecients= pd.DataFrame(regressor.coef_,features)
coeffecients.column = ['Coeffecient']
coeffecients


# In[ ]:




