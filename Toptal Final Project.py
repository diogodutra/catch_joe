#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score


# # Load dataset

# In[2]:


file = 'data/dataset.json'

df = pd.read_json(file)

# create target label as a separate column for supervised learning
df['joe'] = df['user_id'] == 0

df.head()


# # Data Exploration

# ## Gender

# The first feature to be explored is `gender`. Let's explore the hypothesis that Joe never changed their gender.

# In[3]:


sns.catplot(x='joe', hue='gender', kind='count', data=df)


# It seems that Joe is a male, since there is no log on his name as female. This insight is useful to discard around 40% of the dataset.

# ## Location

# In[4]:


locations = list(set(df['location']))
print(len(locations), 'locations in the dataset:', *locations)


# Let's now explore `location`. Unless Joe works in a cruising ship, probably he has limited variation of location around the globe so let's explore this hypothesis.

# In[5]:


sns.catplot(x='joe', hue='location', kind='count', data=df)


# In[6]:


Counter(df[df['joe']]['location'])


# Joe has access logs from Paris, Chicago and Toronto only. This is helpful to discard the other 18 locations.

# ## Language

# Let's now explore `locale`. It is rare to find an active polyglot so let's explore this hypothesis.

# In[7]:


sns.catplot(x='joe', hue='locale', kind='count', data=df)


# In[8]:


Counter(df[df['joe']]['locale'])


# Despite the fact that Joe has many access from France, USA and Canada, his sessions are always in Russian language. Again, this eliminates all the other languages.

# ## Operating System

# If Joe is not a geek than he is probably using only one or two different `os`.

# In[9]:


sns.catplot(x='joe', hue='os', kind='count', data=df)


# Indeed, Joe uses only Ubuntu and Windows 10. This rules out MacOS, Debian and the rest of the Microsoft's OS.

# ## Browser

# For the same reason explained before for the OS, Joe is probably using only a couple of `browsers`.

# In[10]:


sns.catplot(x='joe', hue='browser', kind='count', data=df)


# Again, Joe uses only Firefox and Chrome, leaving out Internet Explorer and Safari.

# # Predictive Model

# The previously mentioned features are good enough to safely tell whenever is not Joe. However, how many logs by chance match exactly at the same time all these features? 

# In[17]:


df_like_joe = df.copy()

# extract set of single entries from Joe's logs
features = ['gender', 'location', 'os', 'browser', 'locale']
filter_data = {feat: set(df[df['joe']][feat]) for feat in features}

for feature, valid_entries in filter_data.items():
    df_like_joe = df_like_joe[df_like_joe[feature].isin(valid_entries)]

    
# extract set of multiple website entries from Joe's logs
sites_joe = {site.get('site') for sites in df_like_joe['sites'] for site in sites}
df_like_joe = df_like_joe[list(map(lambda x:
                any(site.get('site') in sites_joe for site in x), df_like_joe['sites']))]
    
    
print('Original dataset contains', df.shape[0], 'logs.')
print('Like-Joe dataset contains', df_like_joe.shape[0], 'logs',
     "({0:.0%}).".format(df_like_joe.shape[0] / df.shape[0]))


# Filtering out those logs that do not match Joe's history is enough to discard a large piece of the dataset.
# 
# But how many of the left logs are our Joe indeed?

# In[12]:


count = Counter(df_like_joe['joe'])
print(count)

is_joe = np.asarray(list(count.values()))
is_joe = list(is_joe / is_joe.sum())

print('False and True logs ratio from Joe:', ', '.join('{0:.1%}'.format(i) for i in is_joe))


# In[13]:


user_id_like_joe = set(df_like_joe['user_id'])
print(len(user_id_like_joe), 'total of user_id with same logs than Joe:', *user_id_like_joe)


# Despite the fact that the filter previously mentioned has efficiently removed most of the dataset, there are yet some logs from a few people with enough occurencies to be a majority over Joe. This is yet something to be tackled, since we don't want these people being taken as Joe.

# # Decision Tree

# Maybe, the cross combination of restrictions across different features is enough to find Joe out of the other few people above. For instance, Joe might be the only one who uses Firefox (`browser`) on Windows 10 (`os`).
# 
# Let's create a simple Decision Tree, train it on the categorical features and check it's performance to detect Joe.

# In[71]:


df_ml = df[features + ['joe',]].copy()

df_ml[features] = df_ml[features].astype('category')

le = dict()
for feat in features + ['joe']:
    le[feat] = LabelEncoder()
    df_ml[feat] = le[feat].fit_transform(df_ml[feat])
    
    
df_ml.head()


# In[114]:


from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[133]:


test_ratio = .5

X, y = df_ml[features].values, df_ml['joe'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_ratio, random_state=42)


model = DecisionTreeClassifier() #KNeighborsClassifier(2) #BaggingClassifier(KNeighborsClassifier()) #AdaBoostClassifier() #RandomForestClassifier()
model.fit(X_train, y_train)
model.score


# In[138]:


y_pred = model.predict(X_test)

print('{0:.0%}'.format(recall_score(y_pred, y_test)), 'of the Joe\'s accesses are detected.')
print('{0:.0%}'.format(precision_score(y_pred, y_test)), 'of the Joe\'s detections are true.')


# The Decision Tree presented an average performance. However, it is far from flawless since there are too many false detections that lead some accesses being misclassified as Joe's.
# 
# Other similar types of classifiers (ie: AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
# and KNeighborsClassifier) resulted in the performance, which indicates that we need to further explore the rest of the non-categorical features.

# # Save

# In[14]:


# convert Notebook to Python for better version control
get_ipython().system(' jupyter nbconvert --to script "Toptal Final Project.ipynb" --output-dir="./code/diogo-dutra"')

