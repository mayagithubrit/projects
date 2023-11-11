#!/usr/bin/env python
# coding: utf-8

# In[169]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="white",color_codes=True)


# In[170]:


nm=['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width','Species']


# In[171]:


iris=pd.read_csv("C:\\Users\\Sachin\\Downloads\\dataset.txt",header=None,names=nm)


# In[172]:


iris


# In[173]:


iris.head()


# In[174]:


iris["Species"].value_counts()


# # Scatter Plot

# In[175]:


sns.FacetGrid(iris, hue="Species",height=6).map(plt.scatter,"Petal.Length","Sepal.Width").add_legend()


# In[176]:


sns.FacetGrid(iris, hue="Species",height=6).map(plt.scatter,"Petal.Width","Sepal.Length").add_legend()


# In[177]:


x=iris.iloc[:,0:4]


# In[178]:


x


# In[179]:


type(x)


# In[180]:


y=iris[['Species']]


# In[181]:


y


# In[182]:


type(y)


# converting categorical variables into numbers

# In[183]:


from sklearn.preprocessing import LabelEncoder


# In[184]:


le=LabelEncoder()


# In[185]:


y=le.fit_transform(y)


# In[186]:


y


# In[187]:


from sklearn.model_selection import train_test_split


# In[188]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)


# In[189]:


x_train.shape


# In[190]:


y_train.shape


# In[191]:


x_test.shape


# In[192]:


y_test.shape


# In[193]:


from sklearn.neighbors import KNeighborsClassifier


# In[194]:


knn=KNeighborsClassifier(n_neighbors=4)


# In[195]:


knn.fit(x_train,y_train)


# In[196]:


y_pred=knn.predict(x_test)


# make prediction

# In[207]:


y_pred


# In[198]:


y_test


# In[199]:


from sklearn.metrics import accuracy_score


# In[200]:


x_test.shape


# In[201]:


acc=accuracy_score(y_pred,y_test)


# In[205]:


acc


# summarize the fit of the model

# In[208]:


from sklearn import metrics


# In[213]:


print(metrics.classification_report(y_test,y_pred))


# In[214]:


print(metrics.confusion_matrix(y_test,y_pred))


# In[215]:


import matplotlib.pyplot as plt


# In[216]:


plt.scatter(x_test['Sepal.Length'],x_test['Petal.Length'],c=y_pred)
plt.xlabel('Sepal.Length')
plt.xlabel('Petal.Length')
plt.show


# In[ ]:




