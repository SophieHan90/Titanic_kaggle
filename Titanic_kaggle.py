#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster

# ## Predict survival on the Titanic
# - Defining the problem statement
# - Collecting the data
# - Exploratory data analysis
# - Feature engineering
# - Modelling
# - Testing

# ## 1. Defining the problem statement
# Complete the analysis of what sorts of people were likely to survive. 
# 
# In particular, we ask you to apply the tools of machine learning to predict which passengers survived the Titanic tragedy.

# In[1]:


from IPython.display import Image
Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/5095eabce4b06cb305058603/5095eabce4b02d37bef4c24c/1352002236895/100_anniversary_titanic_sinking_by_esai8mellows-d4xbme8.jpg")


# ## 2. Collecting the data

# ### load train, test dataset using Pandas

# In[2]:


import pandas as pd

train = pd.read_csv('downloads/train.csv')
test = pd.read_csv('downloads/test.csv')


# ## 3. Exploratory data analysis
# Printing first 5 rows of the train dataset.

# In[3]:


train.head()


# ### Data Dictionary
# Survived: 0 = No, 1 = Yes
# 
# pclass: Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd
# 
# sibsp: # of siblings / spouses aboard the Titanic
# 
# parch: # of parents / children aboard the Titanic
# 
# ticket: Ticket number
# 
# cabin: Cabin number
# 
# embarked: Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton

# #### Total rows and columns
# We can see that there are 891 rows and 12 columns in our training dataset.

# In[4]:


test.head()


# In[5]:


train.shape


# In[6]:


test.shape


# In[7]:


train.info()


# In[8]:


test.info()


# In[9]:


train.isnull().sum()


# In[10]:


test.isnull().sum()


# ### import python lib for visualization

# In[11]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set() # setting seaborn default for plots


# Bar Chart for Categorical Features
# 
# - Pclass
# - Sex
# - SibSp ( # of siblings and spouse)
# - Parch ( # of parents and children)
# - Embarked
# - Cabin

# In[12]:


def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar', stacked=True, figsize=(10,5))


# In[13]:


bar_chart('Sex')


# In[14]:


bar_chart('Pclass')


# In[15]:


bar_chart('SibSp')


# In[16]:


bar_chart('Parch')


# In[17]:


bar_chart('Embarked')


# ## 4. Feature engineering
# Feature engineering is the process of using domain knowledge of the data
# to create features (feature vectors)^ that make machine learning algorithms work.
# 
# feature vector is an n-dimensional vector of numerical features that represent some objects.
# Many algorithms in machine learning require a numerical representation of objects,
# since such representations facilitate processing and statistical analysis.

# In[18]:


train.head()


# ### 4.1 how Titanic sank?
# sank from the bow of the ship where third class rooms located
# conclusion, Pclass is key feature for classifier

# In[19]:


Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/t/5090b249e4b047ba54dfd258/1351660113175/TItanic-Survival-Infographic.jpg?format=1500w")


# In[20]:


train.head(10)


# ### 4.2 Name

# In[21]:


train_test_data = [train, test] # combining train and test dataset


# In[22]:


train_test_data


# In[23]:


for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[24]:


train['Title'].value_counts()


# In[25]:


test['Title'].value_counts()


# #### Title map
# 
# Mr : 0 
# 
# Miss : 1 
# 
# Mrs: 2 
# 
# Others: 3

# In[26]:


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[27]:


train.head()


# In[28]:


test.head()


# In[29]:


bar_chart('Title')


# In[30]:


# delete unnecessary feature from dataset
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)


# In[31]:


train.head()


# In[32]:


test.head()


# ### 4.3 Sex
# male: 0 
# 
# female: 1

# In[33]:


sex_mapping = {"male":0, "female":1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# In[34]:


bar_chart('Sex')


# ### 4.4 Age
# #### 4.4.1 some age is missing
# Using Title's average age for missing Age

# In[35]:


# fill missing age with median age for each title (Mr, Mrs, Miss, Others)
train['Age'].fillna(train.groupby('Title')['Age'].transform('mean'), inplace = True)
test['Age'].fillna(test.groupby('Title')['Age'].transform('mean'), inplace = True)


# In[36]:


facet = sns.FacetGrid(train, hue="Survived",aspect=5)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
 
plt.show()


# In[37]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(0, 20)


# In[38]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(20, 40)


# In[39]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(40, 60)


# In[40]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(60)


# In[41]:


train.info()


# #### 4.4.2 Binning
# Binning/Converting Numerical Age to Categorical Variable
# 
# feature vector map:
# 
# child: 0
# 
# young: 1
# 
# adult: 2
# 
# mid-age: 3
# 
# senior: 4

# In[42]:


for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 15, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 15) & (dataset['Age'] <= 25), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 35), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4


# In[43]:


train.head()


# In[44]:


bar_chart('Age')


# #### 4.5 Embarked
# ##### 4.5.1 filling missing values

# In[45]:


Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# - More than 50% of 1st, 2nd, and 3rd class are from S embark

# *Fill out missing embark with S embark*

# In[46]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[47]:


train.isna()


# In[48]:


embark_mapping = {"S":0, "C":1, "Q":2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embark_mapping)


# In[49]:


train.head()


# In[50]:


test.head()


# In[51]:


train.isnull().sum()


# In[52]:


test.isnull().sum()


# #### 4.6 Fare

# In[53]:


# fill missing Fare with median fare for each Pclass
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
train.head(50)


# In[54]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
 
plt.show()


# In[55]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
plt.xlim(0, 20)


# In[56]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
plt.xlim(20, 40)


# In[57]:


for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3


# In[58]:


train.head(50)


# In[59]:


test.head(50)


# #### 4.7 Cabin

# In[60]:


train.Cabin.value_counts()


# In[61]:


for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]


# In[62]:


Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[63]:


Pclass1


# In[64]:


Pclass2


# In[65]:


Pclass3


# In[66]:


cabin_mapping = {"A":0, "B":0.2, "C":0.4, "D":0.6, "E":0.8, "F":1.0, "G":1.2, "T":1.4}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)


# In[67]:


train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


# In[68]:


train.head()


# In[69]:


test.head()


# #### 4.8 FamilySize

# In[70]:


#Combine 'SibSp(sibilings)' + 'Parch(parents)'
train['Family size'] = train['SibSp'] + train['Parch']+1
test['Family size'] = test['SibSp'] + test['Parch']+1


# In[71]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Family size',shade= True)
facet.set(xlim=(0, train['Family size'].max()))
facet.add_legend()
plt.xlim(0)


# In[72]:


train['Family size'].value_counts()


# In[73]:


family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['Family size'] = dataset['Family size'].map(family_mapping)


# In[74]:


train.head()


# In[75]:


test.head()


# In[76]:


features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)
train_data = train.drop('Survived', axis=1)
target = train['Survived']

train_data.shape, target.shape


# In[77]:


train.head()


# In[78]:


test.head()


# ### 5. Modelling

# In[79]:


# Importing Classifier Modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np


# #### 5.1 Cross Validation (K-fold)

# In[80]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# #### 5.1.2 kNN

# In[81]:


clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[82]:


#kNN Score
round(np.mean(score)*100, 2)


# #### 5.2.2 Decision Tree

# In[83]:


clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[84]:


#Desicision Score
round(np.mean(score)*100, 2)


# #### 5.2.3 Random Forest

# In[85]:


clf = RandomForestClassifier(n_estimators=300)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

clf=RandomForestClassifier(n_estimators=100)


# In[86]:


#Random Forest Score
round(np.mean(score)*100, 2)


# #### 5.2.4 Naive Bayes

# In[87]:


clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[88]:


#Naive Bayes Score
round(np.mean(score)*100, 2)


# #### 5.2.5 SVM

# In[92]:


clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[93]:


round(np.mean(score)*100,2)


# ## 6. Testing

# In[94]:


clf = SVC()
clf.fit(train_data, target)

test_data = test.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)


# In[99]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission_titanic.csv', index=False)


# In[100]:


submission = pd.read_csv('submission_titanic.csv')
submission.head()


# # References
# This notebook is created by learning from the following notebooks:
# 
# Mukesh ChapagainTitanic Solution: A Beginner's Guide
# 
# How to score 0.8134 in Titanic Kaggle Challenge
# 
# Titanic: factors to survive
# 
# Titanic Survivors Dataset and Data Wrangling

# In[ ]:




