#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import tree
from dtreeviz.trees import dtreeviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/Diogo/anaconda3/Library/bin/graphviz'


# The present Jupyter Notebook explains the process of creating a predictive model to classify an user access as **Joe** or **not-Joe** using this [dataset](https://drive.google.com/file/d/1nATkzOZUe6w5IWcFNE3AakzBl-6P-5Hw/view?usp=sharing).
# 
# The sections are numbered as follows:
# 
# 1. Load dataset
# 1. Data Exploration
#     - Location
#     - Gender
#     - Language
#     - Operating System
#     - Browser
#     - Time of the day
#     - Day of the week
#     - Day of the month
#     - Month of the year
#     - Duration
# 1. Predictive Model
#     - Naïve Analysis
#     - Decision Tree
#     - Adding websites
# 1. Save

# # 1 Load dataset

# In[2]:


file = 'data/dataset.json'

df = pd.read_json(file)

# create target label as a separate column for supervised learning
user_id_joe = 0
df['joe'] = df['user_id'] == user_id_joe

df.head()


# # 2 Data Exploration

# ## 2.1 Gender

# The first feature to be explored is `gender`. Let's explore the hypothesis that Joe never changed their gender.

# In[3]:


sns.catplot(x='joe', hue='gender', kind='count', data=df)


# It seems that Joe is a male, since there is no log on his name as female. This insight is useful to discard around 40% of the dataset.

# ## 2.2 Location

# In[4]:


locations = list(set(df['location']))
print(len(locations), 'locations in the dataset:', *locations)


# Let's now explore `location`. Unless Joe works in a cruising ship, probably he has limited variation of location around the globe so let's explore this hypothesis.

# In[5]:


sns.catplot(x='joe', hue='location', kind='count', data=df)


# In[6]:


Counter(df[df['joe']]['location'])


# Joe has access logs from Paris, Chicago and Toronto only. This is helpful to discard the other 18 locations.

# ## 2.3 Language

# Let's now explore `locale`. It is rare to find an active polyglot so let's explore this hypothesis.

# In[7]:


sns.catplot(x='joe', hue='locale', kind='count', data=df)


# In[8]:


Counter(df[df['joe']]['locale'])


# Despite the fact that Joe has many access from France, USA and Canada, his sessions are always in Russian language. Again, this eliminates all the other languages.

# ## 2.4 Operating System

# If Joe is not a geek than he is probably using only one or two different `os`.

# In[9]:


sns.catplot(x='joe', hue='os', kind='count', data=df)


# Indeed, Joe uses only Ubuntu and Windows 10. This rules out MacOS, Debian and the rest of the Microsoft's OS.

# ## 2.5 Browser

# For the same reason explained before for the OS, Joe is probably using only a couple of `browsers`.

# In[10]:


sns.catplot(x='joe', hue='browser', kind='count', data=df)


# Again, Joe uses only Firefox and Chrome, leaving out Internet Explorer and Safari.

# ## 2.6 Time of the day

# Let's now verify the hypothesis that Joe accesses internet only in some specific hours of the day.

# In[11]:


df['hour'] = [int(time.split(':')[0]) for time in df['time']]

sns.catplot(x='joe', hue='hour', kind='count', data=df)


# In[12]:


print('Hours of the day with Joe\'s accesses:', *set(df[df['joe']]['hour']))


# It seems that Joe never accesses internet at some specific hours of the day. Therefore, this is yet another relevant information to be used by our classifier.

# ## 2.7 Day of the week

# Following the rationale from the previous subsection, let's verify the hypothesis that Joe accesses internet only in some specific days of the week.

# In[13]:


df['weekday'] = [date.day_name() for date in df['date']]

sns.catplot(x='joe', hue='weekday', kind='count', data=df)


# In[14]:


Counter(df[df['joe']]['weekday'])


# There is no particular day of the week that shows an unusual history of access from Joe, so let's drop this feature.

# ## 2.8 Day of the month

# Let's verify if Joe has different frequency of accesses along the days of the month.

# In[15]:


df['monthday'] = [date.day for date in df['date']]

sns.catplot(x='joe', hue='monthday', kind='count', data=df)


# There is no unusual pattern to be extracted out of the day of the month.

# ## 2.9 Month of the year

# Now let's check if there is any useful pattern along the months of the year.

# In[16]:


df['month'] = [date.month for date in df['date']]

sns.catplot(x='joe', hue='month', kind='count', data=df)


# Again, nothing useful from the month of the year.

# ## 2.10 Duration

# In[17]:


df['duration'] = [sum(map(lambda x: x.get('length'), sites)) for sites in df['sites']]


# Plotting the duration would be a bit harder to analyse. Intead, let's compare the duration statistics of the population against Joe's.

# In[18]:


df['duration'].describe()


# In[19]:


df[df['joe']]['duration'].describe()


# Joe's duration of access is fit within the statistical boundaries of the population, which means that there is nothing unusual. Nonetheless, let's keep this feature since it might have some useful correlation with other features.

# # 3 Predictive Model

# ## 3.1 Naïve Analysis

# The previously mentioned features are good enough to safely tell whenever is not Joe. However, how many logs by chance match exactly at the same time all these features? 

# In[20]:


df_like_joe = df.copy()

# extract set of single entries from Joe's logs
features = ['gender', 'location', 'os', 'browser', 'locale', 'hour', 'duration']
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

# In[21]:


count = Counter(df_like_joe['joe'])
print(count)

is_joe = np.asarray(list(count.values()))
is_joe = list(is_joe / is_joe.sum())

print('False and True logs ratio from Joe:', ', '.join('{0:.1%}'.format(i) for i in is_joe))


# In[22]:


user_id_like_joe = set(df_like_joe['user_id'])
print(len(user_id_like_joe), 'total of user_id with same logs than Joe:', *user_id_like_joe)


# Despite the fact that the filter previously mentioned has efficiently removed most of the dataset, there are yet some logs from a few people with enough occurencies to be a majority over Joe. This is yet something to be tackled, since we don't want these people being taken as Joe.

# ## 3.2 Decision Tree

# Maybe, the cross combination of restrictions across different features is enough to find Joe out of the other few people above. For instance, Joe might be the only one who uses Firefox (`browser`) on Windows 10 (`os`).
# 
# Let's create a simple Decision Tree, train it on the single-entries categorical features and check it's performance to detect Joe.

# In[23]:


def categorize(df, features):

    df[features] = df[features].astype('category')

    le = dict()
    for feat in features:
        le[feat] = LabelEncoder()
        df[feat] = le[feat].fit_transform(df[feat])
        
    return df, le
        
        
df_ml = df[features + ['joe',]].copy()
df_ml, le = categorize(df_ml, features + ['joe'])
df_ml.head()


# In[24]:


def split_data(df, label, **kwargs):
    features = set(df_ml.columns) - {label}
    X, y = df[features].values, df[label].values
    return train_test_split(X, y, **kwargs)
    
    
X_train, X_test, y_train, y_test = split_data(
    df_ml, 'joe', test_size=.5, random_state=42)


# In[25]:


def create_model(X_train, y_train, classifier=DecisionTreeClassifier(max_depth=3)):
    return classifier.fit(X_train, y_train)


model = create_model(X_train, y_train)


# In[26]:


def print_scores(y_pred, y_test):
    print('{0:.2%}'.format(accuracy_score(y_pred, y_test)), 'is the accuracy of the classifier.')
    print('{0:.0%}'.format(recall_score(y_pred, y_test)), 'of the Joe\'s accesses are detected.')
    print('{0:.0%}'.format(precision_score(y_pred, y_test)), 'of the Joe\'s detections are true.')
    
    
y_pred = model.predict(X_test)
print_scores(y_pred, y_test)


# The Decision Tree presents an average performance. However, it is far from excellent since there are too many false detections which causes some accesses to be misclassified as Joe's.

# Before we move on to improve the performance, here is a question. How exactly does this Decision Tree above work in order to classify?
# 
# In order to help us answer this question, let's plot the nodes of the Decision Tree as a graph plot below.

# In[27]:


dtreeviz(model, X_train, y_train,
                target_name="target",
                feature_names=features+['site'],
                class_names=['not Joe', 'Joe'])


# In[28]:


le['locale'].inverse_transform([18])


# The graph above shows that the Decision Tree queries the features in the following order:
# 1. If the language (`locale`) is less than 17.5 then is not Joe with 100% of certainty; else ...
# 1. If the language (`locale`) is more than 18.5 then is is not Joe with 100% of certainty; else ...
# 1. If the duration of access is less than 3.5 than it is Joe with 100% of certainty; else ...
# 1. We run out of questions so it guesses it is Joe with 18% of centainty.
# 
# Mind that `locale = 18` is the Russian language as coded by the `LabelEncoder`. Therefore, the first 2 questions above are mainly telling us that if the language of access is not Russian than it is not 
# Joe for sure.
# 
# Another observtion is that the `duration` alone feature was not useful, as explained in the previous section. However, it became useful for the remaining cases that are exclusivelly Russian speakers.

# Other similar types of classifiers (ie: DecisionTree with unlimited maximum depth, AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
# and KNeighborsClassifier) resulted in similar performance, which indicates that we need to further explore the rest of the non-categorical features.

# ## 3.3 Adding Websites

# Now, let's do some trick to include the `sites` that contains multiple-entries.

# In[29]:


df_ml = pd.DataFrame(pd.DataFrame(df['sites'].values.tolist()).stack().reset_index(level=1))
df_ml.columns = ['keys', 'values']

sites_keys = list(df['sites'].values[0][0].keys())
for new_feat in sites_keys:
    df_ml[new_feat] = [v.get(new_feat) for v in df_ml['values']]
    
df_ml = df_ml.join(df).drop(columns={'keys', 'values', 'sites'})

features += ['site']
df_ml = df_ml[features + ['joe',]]

df_ml.head()


# In[30]:


df_ml, le = categorize(df_ml, features + ['joe'])


X_train, X_test, y_train, y_test = split_data(
    df_ml, 'joe', test_size=.2, random_state=42)

print(len(y_test), 'samples in the test dataset.')


# In[31]:


# options: DecisionTreeClassifier AdaBoostClassifier BaggingClassifier RandomForestClassifier KNeighborsClassifier GradientBoostingClassifier
model = create_model(X_train, y_train, DecisionTreeClassifier())

y_pred = model.predict(X_test)
print_scores(y_pred, y_test)


# It seems that `sites` is an useful feature indeed.
# 
# The new classifier performance is much better than the previous one to the point that it can be deployed.

# In[32]:


model.get_depth()


# This time we are not plotting the nodes of the Decision Tree simply because it needs many more depths (36) than the previous one (3), which makes it harder to visualize. But the logic is the same: it is just a matter successively questioning the values of the features (nodes) and using the answer (branch) to follow to the next question (next node) until landing into a position without further questions (end node), which contains instead a classification (in our case, Joe or not-Joe).

# # 4 Save

# The following code is to convert the present Jupyter Notebook into Python script. The script is the one under version control since we do not want to keep track of JSON codes internal to the `.ipynb` files.

# In[33]:


# convert Notebook to Python for better version control
get_ipython().system(' jupyter nbconvert --to script "Toptal Final Project.ipynb" --output-dir="./code/diogo-dutra"')

