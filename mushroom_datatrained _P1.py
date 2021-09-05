#!/usr/bin/env python
# coding: utf-8

# In[1]:


print('mushroom project')


# In[2]:


import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data = pd.read_csv(r"C:\Users\gavel\Downloads\mushroom.csv")
data.shape


# In[4]:


data.head()


# In[5]:


from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()


# In[6]:


for col in data.columns:
    data[col]=lbl.fit_transform(data[col])
data


# In[7]:


data.head()


# In[8]:


#split the x and y variables
y=data['class']
x=data.iloc[:,1:23]


# In[9]:


x.shape


# In[10]:


y.shape


# In[11]:


x.head


# In[12]:


y.head


# In[13]:


#I want to use PCA on this data. First normalise the data using StandardScalar so that the data is now between -1 and 1
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
x


# In[14]:


#using principal component analysis
#Even though the number of variables is not too high, I would still like to use PCA to see which variables describe the maximum variance in data
from sklearn.decomposition import PCA
pca = PCA()
x_pca = pca.fit_transform(x)


# In[15]:


#plot a Scree plot of the Principal Components
plt.figure(figsize=(16,11))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.grid()


# In[16]:


#from the graph, first 17 components describe the maximum variance(more than 90% of the data). We shall use them for our subsequent analysis.
new_pca = PCA(n_components=17)


# In[17]:


x_new = new_pca.fit_transform(x)


# In[18]:


#using KMeans to plot the clusters. We know that we habe 2 classes of the target variable. So n_clusters=2
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=2)


# In[19]:


k_means.fit_predict(x_new )


# In[20]:


#plot the clusters.
colors = ['r','g']
for i in range(len(x_new)):
    plt.scatter(x_new[i][0], x_new[i][1], c=colors[k_means.labels_[i]], s=10)
plt.show()


# In[21]:


x_new.shape


# In[22]:


#separate the train and test data
from sklearn.model_selection import train_test_split


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.25, random_state = 6)


# In[24]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[25]:


#using Logistic regression to build the first model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_predict =lr.predict(x_test)


# In[26]:


lr_predict_prob = lr.predict_proba(x_test)
print(lr_predict)
print(lr_predict_prob[:,1])


# In[27]:


#import metrics
from sklearn.metrics import confusion_matrix, accuracy_score


# In[28]:


lr_conf_matrix = confusion_matrix(y_test, lr_predict)
lr_accuracy = accuracy_score(y_test, lr_predict)
print(lr_conf_matrix)
print(lr_accuracy)


# In[29]:


#roc curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test,lr_predict_prob[:,1] )


# In[30]:


#auc score
from sklearn.metrics import auc
lr_auc = auc(fpr, tpr)
print(lr_auc)


# In[31]:


#plotting ROC curve
plt.figure(figsize=(10,9))
plt.plot(fpr, tpr, label = 'AUC= %0.2f' % lr_auc )
plt.plot([0,1],[0,1], linestyle = '--')
plt.legend()


# In[32]:


#Using Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
gnb_predict = gnb.predict(x_test)
gnb_predict_prob = gnb.predict_proba(x_test)


# In[33]:


print(gnb_predict)
print(gnb_predict_prob)


# In[34]:


gnb_conf_matrix = confusion_matrix(y_test, gnb_predict)
gnb_accuracy_score = accuracy_score(y_test, gnb_predict)
print(gnb_conf_matrix)
print(gnb_accuracy_score)


# In[35]:


#calculate ROC and AUC
fpr, tpr, thresholds = roc_curve(y_test, gnb_predict_prob[:,1])
#print auc
gnb_auc = auc(fpr, tpr)
print(gnb_auc)


# In[36]:


#plot ROC curve
plt.figure(figsize=(10,9))
plt.plot(fpr, tpr, label = 'AUC %0.2f' % gnb_auc)
plt.plot([0,1],[0,1], linestyle = '--')
plt.legend()


# In[37]:


#lets use Decision Trees to classify 
#use the number of trees as 10 first
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=10)


# In[38]:


dt.fit(x_train,y_train)
dt_predict = dt.predict(x_test)
dt_predict_prob = dt.predict_proba(x_test)


# In[39]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[40]:


dt_conf_matrix = confusion_matrix(y_test, dt_predict)
dt_accuracy_score = accuracy_score(y_test, dt_predict)
print(dt_conf_matrix)
print(dt_accuracy_score)


# In[41]:


#calculate auc and plot roc
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, dt_predict_prob[:,1])
dt_auc = auc(fpr, tpr)
print(dt_auc)


# In[42]:


#plot ROC curve
plt.figure(figsize=(10,9))
plt.plot(fpr, tpr, label = 'AUC %0.2f' % dt_auc)
plt.plot([0,1],[0,1], linestyle = '--')
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.legend()
plt.grid()


# In[43]:


#another testing method
#using random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=10) #10 trees
rf.fit(x_train, y_train)
rf_predict = rf.predict(x_test)
rf_predict_prob = rf.predict_proba(x_test)


# In[44]:


rf_conf_matrix = confusion_matrix(y_test,rf_predict)
rf_accuracy_score = accuracy_score(y_test, rf_predict)
print(rf_conf_matrix)
print(rf_accuracy_score)
#random forest has a higher accuracy score than the decision tree
#Decision tree = 99.3
#Random forest = 99.9


# In[45]:


fpr, tpr, thresholds = roc_curve(y_test, rf_predict_prob[:,1])
rf_auc = auc(fpr, tpr)
print(rf_auc)


# In[46]:


#plot the ROC curve
plt.figure(figsize=(10,9))
plt.plot(fpr, tpr, label = 'AUC: %0.2f' % rf_auc)
plt.plot([1,0],[1,0], linestyle = '--')
plt.legend(loc=0)
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.grid()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




