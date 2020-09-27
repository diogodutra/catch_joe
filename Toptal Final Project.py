#!/usr/bin/env python
# coding: utf-8

# # TODO:
# - [x] Split dataset in chronological order
# - [ ] Improve classifier performance
# - [ ] Handle unseen labels
# - [ ] Create standalone script
# - [ ] Add readme

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter, defaultdict

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
# 1. Save

# # 1 Load dataset

# In[2]:


file = 'data/dataset.json'

df = pd.read_json(file)

# create target label as a separate column
user_id_joe = 0
df['joe'] = df['user_id'] == user_id_joe

df.head()


# In[3]:


# split dataset into train and test
# in chronological order (not random)

test_ratio = 0.1
df_test = df.sort_values(by='date').tail(int(test_ratio * df.shape[0]))

df = df.drop(df_test.index) # train dataset

df.shape


# # 2 Data Exploration

# ## 2.1 Gender

# The first feature to be explored is `gender`. Let's explore the hypothesis that Joe never changed their gender.

# In[4]:


sns.catplot(x='joe', hue='gender', kind='count', data=df)


# It seems that Joe is a male, since there is no log on his name as female. This insight is useful to discard around 40% of the dataset.

# ## 2.2 Location

# In[5]:


locations = list(set(df['location']))
print(len(locations), 'locations in the dataset:', *locations)


# Let's now explore `location`. Unless Joe works in a cruising ship, probably he has limited variation of location around the globe so let's explore this hypothesis.

# In[6]:


sns.catplot(x='joe', hue='location', kind='count', data=df)


# In[7]:


Counter(df[df['joe']]['location'])


# Joe has access logs from Paris, Chicago and Toronto only. This is helpful to discard the other 18 locations.

# ## 2.3 Language

# Let's now explore `locale`. It is rare to find an active polyglot so let's explore this hypothesis.

# In[8]:


sns.catplot(x='joe', hue='locale', kind='count', data=df)


# In[9]:


Counter(df[df['joe']]['locale'])


# Despite the fact that Joe has many access from France, USA and Canada, his sessions are always in Russian language. Again, this eliminates all the other languages.

# ## 2.4 Operating System

# If Joe is not a geek than he is probably using only one or two different `os`.

# In[10]:


sns.catplot(x='joe', hue='os', kind='count', data=df)


# Indeed, Joe uses only Ubuntu and Windows 10. This rules out MacOS, Debian and the rest of the Microsoft's OS.

# ## 2.5 Browser

# For the same reason explained before for the OS, Joe is probably using only a couple of `browsers`.

# In[11]:


sns.catplot(x='joe', hue='browser', kind='count', data=df)


# Again, Joe uses only Firefox and Chrome, ruling out Internet Explorer and Safari.

# ## 2.6 Time of the day

# Let's now verify the hypothesis that Joe accesses internet only in some specific hours of the day.

# In[12]:


def extract_hour(df):
    return [int(time.split(':')[0]) for time in df['time']]


df['hour'] = extract_hour(df)
sns.catplot(x='joe', hue='hour', kind='count', data=df)


# In[13]:


print('Hours of the day with Joe\'s accesses:', *set(df[df['joe']]['hour']))


# It seems that Joe never accesses internet at some specific hours of the day. Therefore, this is yet another relevant information to be used by our classifier.

# ## 2.7 Day of the week

# Following the rationale from the previous subsection, let's verify the hypothesis that Joe accesses internet only in some specific days of the week.

# In[14]:


df['weekday'] = [date.day_name() for date in df['date']]

sns.catplot(x='joe', hue='weekday', kind='count', data=df)


# In[15]:


Counter(df[df['joe']]['weekday'])


# There is no particular day of the week that shows an unusual history of access from Joe, so let's drop this feature.

# ## 2.8 Day of the month

# Let's verify if Joe has different frequency of accesses along the days of the month.

# In[16]:


df['monthday'] = [date.day for date in df['date']]

sns.catplot(x='joe', hue='monthday', kind='count', data=df)


# There is no unusual pattern to be extracted out of the day of the month.

# ## 2.9 Month of the year

# Now let's check if there is any useful pattern along the months of the year.

# In[17]:


df['month'] = [date.month for date in df['date']]

sns.catplot(x='joe', hue='month', kind='count', data=df)


# Again, nothing useful from the month of the year.

# ## 2.10 Duration

# In[18]:


def extract_duration(df):
    return [sum(map(lambda x: x.get('length'), sites)) for sites in df['sites']]


df['duration'] = extract_duration(df)


# Plotting the duration would be a bit harder to analyse. Intead, let's compare the duration statistics of the population against Joe's.

# In[19]:


df['duration'].describe()


# In[20]:


df[df['joe']]['duration'].describe()


# Joe's duration of access is fit within the statistical boundaries of the population, which means that there is nothing unusual. Nonetheless, let's keep this feature since it is slightly off the population statistics so it might have some useful correlation with other features.

# In[21]:


sites_joe = {site.get('site') for sites in df[df['joe']]['sites'] for site in sites}
print(len(sites_joe), 'sites accessed by Joe.')


# In[22]:


def intersection_ratio(set_this, set_reference):
    return len(set(set_this) & set(set_reference)) / len(set(set_this)) if len(set(set_this)) > 0 else 0


def extract_sites_ratio(df):
    return [intersection_ratio([site.get('site') for site in sites], sites_joe)
                       for sites in df['sites']]


df['sites_ratio'] = extract_sites_ratio(df)
df[~df['joe']]['sites_ratio'].describe()


# In[23]:


df_test['sites_ratio'] = extract_sites_ratio(df_test)
df_test[df_test['joe']]['sites_ratio'].describe()


# In[24]:


def extract_site_old(df):
    return [not {site.get('site') for site in sites}.isdisjoint(sites_joe)
                       for sites in df['sites']]


df['site_old'] = extract_site_old(df)
df_test['site_old'] = extract_site_old(df_test)

sns.catplot(x='joe', hue='site_old', kind='count', data=df_test)


# In[25]:


sites_joe_list = list(sites_joe)

def create_dataframe_lengths_per_site(df, n_sites=20):
    sites_joe_length = np.zeros((df.shape[0], len(sites_joe)))
    for i_row, sites in enumerate(df['sites']):
        for site in sites:
            try:
                i_site = sites_joe_list.index(site.get('site'))
                sites_joe_length[i_row, i_site] = site.get('length')
            except:
                # site not found in Joe's history
                pass
           
    df_lengths = pd.DataFrame(sites_joe_length, columns=sites_joe_list)
#     df_lengths = df_lengths.reindex(df_lengths.mean().sort_values().index, axis=1)
#     df_lengths = df_lengths.loc
    
    return df_lengths
                
    
    
df_sites_joe_length = create_dataframe_lengths_per_site(df[df['joe']])
df_sites_joe_length.mean().sort_values(ascending=False)


# In[26]:


df_sites_all_length = create_dataframe_lengths_per_site(df)
df_sites_all_length.mean().sort_values(ascending=False)


# In[27]:


def extract_site_lengths(df):
    return df.join(create_dataframe_lengths_per_site(df))


df = extract_site_lengths(df)


# # 3 Predictive Model

# ## 3.1 Naïve Analysis

# The previously mentioned features are good enough to safely tell whenever is not Joe. However, how many logs by chance match exactly at the same time all these features? 

# In[28]:


df_like_joe = df.copy()

# extract set of single entries from Joe's logs
features = ['gender', 'os', 'browser', 'locale', 'hour', 'duration']
features += ['location', ]
# features += ['site_old']
# features += ['sites_ratio']
# features += sites_joe_list

filter_data = {feat: set(df[df['joe']][feat]) for feat in features}

for feature, valid_entries in filter_data.items():
    df_like_joe = df_like_joe[df_like_joe[feature].isin(valid_entries)]

    
# extract set of multiple website entries from Joe's logs
# sites_joe = {site.get('site') for sites in df_like_joe['sites'] for site in sites}
# df_like_joe = df_like_joe[list(map(lambda x:
#                 any(site.get('site') in sites_joe for site in x), df_like_joe['sites']))]
    
    
print('Original dataset contains', df.shape[0], 'logs.')
print('Like-Joe dataset contains', df_like_joe.shape[0], 'logs',
     "({0:.0%}).".format(df_like_joe.shape[0] / df.shape[0]))


# Filtering out those logs that do not match Joe's history is enough to discard a large piece of the dataset.
# 
# But how many of the left logs are our Joe indeed?

# In[29]:


# count = Counter(df_like_joe['joe'])
# print(count)

# is_joe = np.asarray(list(count.values()))
# is_joe = list(is_joe / is_joe.sum())

# print('False and True accesses ratio from Joe:', ', '.join('{0:.1%}'.format(i) for i in is_joe))


# In[30]:


# user_id_like_joe = set(df_like_joe['user_id'])
# print(len(user_id_like_joe), 'total of user_id with same logs than Joe:', *user_id_like_joe)


# Despite the fact that the filter previously mentioned has efficiently removed most of the dataset, there are yet some logs from a few people with enough occurencies to be a majority over Joe. This is yet something to be tackled, since we don't want these people being taken as Joe.

# ## 3.2 Decision Tree

# Maybe, the cross combination of restrictions across different features is enough to find Joe out of the other few people above. For instance, Joe might be the only one who uses Firefox (`browser`) on Windows 10 (`os`).
# 
# Let's create a simple Decision Tree, train it on the single-entries categorical features and check it's performance to detect Joe.

# In[31]:


def categorize(df, features):

    df[features] = df[features].astype('category')

    le = {feat: LabelEncoder().fit(df[feat]) for feat in features}
        
    return df, le


def encode_features(df, features, le):

    for feat in features:
        df[feat] = le[feat].transform(df[feat])
        
    return df
        

def encode_joe(is_joe_bool_list, encode_dict={True: user_id_joe, False: 1}):
    return [encode_dict[is_joe] for is_joe in is_joe_bool_list]
    

df_train = df[features + ['joe']].copy()
features_categorical = ['gender', 'os', 'browser', 'locale', 'hour']
features_categorical += ['location', ]
# features_categorical += ['site_old']

df_train, le = categorize(df_train, features_categorical)
df_train = encode_features(df_train, features_categorical, le)
df_train['joe'] = encode_joe(df_train['joe'])

df_train.head()


# In[32]:


X_train = df_train[features].values
y_train = df_train['joe']


# Before creating our predictive model, let's calculate the Naïve performance. We know that the majority of the data is not from Joe so the Naïve classifier always assume that the result is 1 (not Joe).

# In[33]:


def print_scores(y_pred, y_test):
    print('{0:.4%}'.format(accuracy_score(y_pred, y_test)), 'is the accuracy of the classifier.')
    print('{0:.0%}'.format(recall_score(y_pred, y_test, pos_label=user_id_joe)), 'of the Joe\'s accesses are detected.')
    print('{0:.0%}'.format(precision_score(y_pred, y_test, pos_label=user_id_joe)), 'of the detections are truly from Joe.')


y_pred = [1] * len(y_train)

print('Performance of Naïve on train dataset:')
print_scores(y_pred, y_train)


# As expected, the accuracy score is quite high because most of the data is not Joe (imbalanced). Moreover, precision and recall scores are obviously nulls because the Naïve blindly guessed it always as not Joe.

# In[34]:


# run preprocessing pipeline on test dataset

# def dataset(df, features, features_categorical):
df_test['duration'] = extract_duration(df_test)
df_test['hour'] = extract_hour(df_test)
df_test['sites_ratio'] = extract_sites_ratio(df_test)
df_test['site_old'] = extract_site_old(df_test)

df_test[features_categorical] = df_test[features_categorical].astype('category')
df_test = encode_features(df_test, features_categorical, le)
df_test = df_test[features + ['joe']]

X_test = df_test[features].values

y_test = encode_joe(df_test['joe'])
df_test['joe'] = y_test
    
#     return df, X, y


# df_train, X_train, y_train = dataset(df_test, features, features_categorical)
# df_test, X_test, y_test = dataset(df_test, features, features_categorical)


# In[35]:


model = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)

print('Performance on train dataset:')
y_pred = model.predict(X_train)
print_scores(y_pred, y_train)


# In[36]:


print('\nPerformance on test dataset:')
y_pred = model.predict(X_test)
print_scores(y_pred, y_test)


# The Decision Tree presents a slight increase of performance when compared to the Naïve. However, it is far from excellent since there are too many missed accesses from Joe.

# Before we move on to improve the performance, here is a question. How exactly does this Decision Tree above work in order to classify?
# 
# In order to help us answer this question, let's plot the nodes of the Decision Tree as a graph plot below.

# In[37]:


dtreeviz(model, X_train, y_train,
                target_name="target",
                feature_names=features,
                class_names=['Joe', 'not Joe'])


# In[38]:


stop


# In[ ]:


le['locale'].inverse_transform([18])


# The graph above shows that the Decision Tree queries the features in the following order:
# 1. If the language (`locale`) is less than 17.5 then is not Joe with 100% of certainty; else ...
# 1. If the language (`locale`) is more than 18.5 then is is not Joe with 100% of certainty; else ...
# 1. If the duration of access is less than 3.5 than it is Joe with 100% of certainty; else ...
# 1. We run out of questions so it guesses it is not Joe with roughly 10% of error.
# 
# Mind that `locale = 18` is the Russian language as coded by the `LabelEncoder`. Therefore, the first 2 questions above are mainly telling us that if the language of access is not Russian than it is not 
# Joe for sure.
# 
# Another observtion is that the `duration` alone feature was not useful, as explained in the previous section. However, it became useful for the remaining cases that are exclusivelly Russian speakers.

# Let's try the performance of other more sophisticted models.

# In[ ]:


models = [
    DecisionTreeClassifier(),
    AdaBoostClassifier(),
    BaggingClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    KNeighborsClassifier(n_neighbors=2),
]

print('Accuracy\t Model')

score_best = -np.Inf
i_best = -1
for i_model, model in enumerate(models):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_pred, y_test)
        
    print('{0:.4%}\t'.format(score), type(model).__name__)
    if score > score_best:
        score_best = score
        i_best = i_model
    


print()
print('Best model:', type(models[i_best]).__name__)
y_pred = models[i_best].predict(X_test)
print_scores(y_pred, y_test)


# Other more sophisticated classifiers resulted in similar performance, which indicates that we need to further explore the rest of the non-categorical features.

# The new classifier performance is much better than the previous one to the point that it can be deployed.

# # 4 Save

# ## 4.1 Saving the model

# In[ ]:


# load the input file
df_verify = pd.read_json('./data/verify.json')

# add some features
df_verify['hour'] = extract_hour(df_verify)
df_verify['duration'] = extract_duration(df_verify)
df_verify['sites_ratio'] = extract_sites_ratio(df_verify)

# remove some unused features
df_verify = df_verify[features]

# convert features into category type
df_verify = encode_features(df_verify, features_categorical, le)


df_verify.head()


# In[ ]:


y_infered = model.predict(df_verify.values)
count = Counter(y_infered)
print(count)
percentage = count[0] / (count[1] + count[0])
print('{0:.0%} of the Verification dataset is detected as Joe\'s access.'.format(percentage))


# ## 4.2 Exporting this Notebook

# The following code is to convert the present Jupyter Notebook into Python script. The script is the one under version control since we do not want to keep track of JSON codes internal to the `.ipynb` files.

# In[ ]:


# convert Notebook to Python for better version control
get_ipython().system(' jupyter nbconvert --to script "Toptal Final Project.ipynb" --output-dir="./code/diogo-dutra"')

