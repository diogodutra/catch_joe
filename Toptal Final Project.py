#!/usr/bin/env python
# coding: utf-8

# # TODO:
# - [x] Split dataset in chronological order
# - [ ] Improve classifier performance
#     - [x] Add top sites as feature
#     - [ ] Replace LabelEncode by OneHotEncode
#     - [x] Scale numeric features
#     - [x] Change metric
#     - [x] Add GridSearchCV
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn import tree
from dtreeviz.trees import dtreeviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/Diogo/anaconda3/Library/bin/graphviz'


# In[2]:


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import GridSearchCV


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

# In[3]:


file = 'data/dataset.json'

df = pd.read_json(file)

# create target label as a separate column
user_id_joe = 0
df['joe'] = df['user_id'] == user_id_joe

df.head()


# In[4]:


# split dataset into train and test
# in chronological order (not random)

test_ratio = 0.1
df_later = df.sort_values(by='date').tail(int(test_ratio * df.shape[0]))

df = df.drop(df_later.index) # train dataset

df.shape


# # 2 Data Exploration

# ## 2.1 Gender

# The first feature to be explored is `gender`. Let's explore the hypothesis that Joe never changed their gender.

# In[5]:


# sns.catplot(x='joe', hue='gender', kind='count', data=df)


# It seems that Joe is a male, since there is no log on his name as female. This insight is useful to discard around 40% of the dataset.

# ## 2.2 Location

# In[6]:


# locations = list(set(df['location']))
# print(len(locations), 'locations in the dataset:', *locations)


# Let's now explore `location`. Unless Joe works in a cruising ship, probably he has limited variation of location around the globe so let's explore this hypothesis.

# In[7]:


# sns.catplot(x='joe', hue='location', kind='count', data=df)


# In[8]:


# Counter(df[df['joe']]['location'])


# Joe has access logs from Paris, Chicago and Toronto only. This is helpful to discard the other 18 locations.

# ## 2.3 Language

# Let's now explore `locale`. It is rare to find an active polyglot so let's explore this hypothesis.

# In[9]:


# sns.catplot(x='joe', hue='locale', kind='count', data=df)


# In[10]:


# Counter(df[df['joe']]['locale'])


# Despite the fact that Joe has many access from France, USA and Canada, his sessions are always in Russian language. Again, this eliminates all the other languages.

# ## 2.4 Operating System

# If Joe is not a geek than he is probably using only one or two different `os`.

# In[11]:


# sns.catplot(x='joe', hue='os', kind='count', data=df)


# Indeed, Joe uses only Ubuntu and Windows 10. This rules out MacOS, Debian and the rest of the Microsoft's OS.

# ## 2.5 Browser

# For the same reason explained before for the OS, Joe is probably using only a couple of `browsers`.

# In[12]:


# sns.catplot(x='joe', hue='browser', kind='count', data=df)


# Again, Joe uses only Firefox and Chrome, ruling out Internet Explorer and Safari.

# ## 2.6 Time of the day

# Let's now verify the hypothesis that Joe accesses internet only in some specific hours of the day.

# In[13]:


def extract_hour(df):
    return [int(time.split(':')[0]) for time in df['time']]


# df['hour'] = extract_hour(df)
# sns.catplot(x='joe', hue='hour', kind='count', data=df)


# In[14]:


# print('Hours of the day with Joe\'s accesses:', *set(df[df['joe']]['hour']))


# It seems that Joe never accesses internet at some specific hours of the day. Therefore, this is yet another relevant information to be used by our classifier.

# ## 2.7 Day of the week

# Following the rationale from the previous subsection, let's verify the hypothesis that Joe accesses internet only in some specific days of the week.

# In[15]:


# df['weekday'] = [date.day_name() for date in df['date']]

# sns.catplot(x='joe', hue='weekday', kind='count', data=df)


# In[16]:


# Counter(df[df['joe']]['weekday'])


# There is no particular day of the week that shows an unusual history of access from Joe, so let's drop this feature.

# ## 2.8 Day of the month

# Let's verify if Joe has different frequency of accesses along the days of the month.

# In[17]:


# df['monthday'] = [date.day for date in df['date']]

# sns.catplot(x='joe', hue='monthday', kind='count', data=df)


# There is no unusual pattern to be extracted out of the day of the month.

# ## 2.9 Month of the year

# Now let's check if there is any useful pattern along the months of the year.

# In[18]:


# df['month'] = [date.month for date in df['date']]

# sns.catplot(x='joe', hue='month', kind='count', data=df)


# Again, nothing useful from the month of the year.

# ## 2.10 Duration

# In[19]:


def extract_duration(df):
    return [sum(map(lambda x: x.get('length'), sites)) for sites in df['sites']]


df['duration'] = extract_duration(df)


# Plotting the duration would be a bit harder to analyse. Intead, let's compare the duration statistics of the population against Joe's.

# In[20]:


# df['duration'].describe()


# In[21]:


# df[df['joe']]['duration'].describe()


# Joe's duration of access is fit within the statistical boundaries of the population, which means that there is nothing unusual. Nonetheless, let's keep this feature since it is slightly off the population statistics so it might have some useful correlation with other features.

# In[22]:


sites_joe = {site.get('site') for sites in df[df['joe']]['sites'] for site in sites}
print(len(sites_joe), 'sites accessed by Joe.')


# In[23]:


def intersection_ratio(set_this, set_reference):
    return len(set(set_this) & set(set_reference)) / len(set(set_this)) if len(set(set_this)) > 0 else 0


def extract_sites_ratio(df):
    return [intersection_ratio([site.get('site') for site in sites], sites_joe)
                       for sites in df['sites']]


df['sites_ratio'] = extract_sites_ratio(df)
df[~df['joe']]['sites_ratio'].describe()


# In[24]:


df_later['sites_ratio'] = extract_sites_ratio(df_later)
df_later[df_later['joe']]['sites_ratio'].describe()


# In[25]:


def extract_site_old(df):
    return [not {site.get('site') for site in sites}.isdisjoint(sites_joe)
                       for sites in df['sites']]


df['site_old'] = extract_site_old(df)
df_later['site_old'] = extract_site_old(df_later)

sns.catplot(x='joe', hue='site_old', kind='count', data=df_later)


# In[26]:


def extract_lengths(df, sites_joe_list):
    sites_joe_length = np.zeros((df.shape[0], len(sites_joe_list)))
    for i_row, sites in enumerate(df['sites']):
        for site in sites:
            try:
                i_site = sites_joe_list.index(site.get('site'))
                sites_joe_length[i_row, i_site] = site.get('length')
            except:
                # site not found in Joe's history
                pass
           
    df_lengths = pd.DataFrame(sites_joe_length, columns=sites_joe_list, index=df.index)
#     df_lengths = df_lengths.reindex(df_lengths.mean().sort_values().index, axis=1)
#     df_lengths = df_lengths.loc
    
    return df_lengths
                
    
sites_joe_list = list(sites_joe)    
df_sites_joe_length = extract_lengths(df[df['joe']], sites_joe_list)
df_sites_joe_length = df_sites_joe_length.mean().sort_values(ascending=False)
df_sites_joe_length


# In[27]:


df_sites_all_length = extract_lengths(df, sites_joe_list)
df_sites_all_length.mean().sort_values(ascending=False)


# In[28]:


top_sites = 100
joe_top_sites = list(df_sites_joe_length[:top_sites].index)
joe_top_sites


# In[29]:


df_lengths = extract_lengths(df[df['joe']], joe_top_sites)
df_lengths.head()


# In[30]:


df_lengths[df_lengths.columns[0]].hist()


# In[31]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer


# In[32]:


class NoScaler(StandardScaler):
    
    def __init__(self):
        pass
    
    def transform(self, x):
        return x
    
    def fit_transform(self, x):
        return self.transform(x)


# In[33]:


scaler = StandardScaler()

scaled = scaler.fit_transform(df_lengths)


# In[34]:


# def extract_site_lengths(df):
#     return df.join(create_dataframe_lengths_per_site(df))


# df = extract_site_lengths(df)


# # 3 Predictive Model

# ## 3.1 Naïve Analysis

# The previously mentioned features are good enough to safely tell whenever is not Joe. However, how many logs by chance match exactly at the same time all these features? 

# In[35]:


# define list of features to be used by the classifier

features = ['gender', 'os', 'browser', 'locale', 'hour', 'duration']
features += ['location', ]
# features += ['site_old']
# features += ['sites_ratio']
# features += sites_joe_list
features += joe_top_sites


features_categorical = ['gender', 'os', 'browser', 'locale', 'hour']
features_categorical += ['location', ]
# features_categorical += ['site_old']


# In[36]:


# df_like_joe = df.copy()

# filter_data = {feat: set(df[df['joe']][feat]) for feat in features}

# for feature, valid_entries in filter_data.items():
#     df_like_joe = df_like_joe[df_like_joe[feature].isin(valid_entries)]

    
# # extract set of multiple website entries from Joe's logs
# # sites_joe = {site.get('site') for sites in df_like_joe['sites'] for site in sites}
# # df_like_joe = df_like_joe[list(map(lambda x:
# #                 any(site.get('site') in sites_joe for site in x), df_like_joe['sites']))]
    
    
# print('Original dataset contains', df.shape[0], 'logs.')
# print('Like-Joe dataset contains', df_like_joe.shape[0], 'logs',
#      "({0:.0%}).".format(df_like_joe.shape[0] / df.shape[0]))


# Filtering out those logs that do not match Joe's history is enough to discard a large piece of the dataset.
# 
# But how many of the left logs are our Joe indeed?

# In[37]:


# count = Counter(df_like_joe['joe'])
# print(count)

# is_joe = np.asarray(list(count.values()))
# is_joe = list(is_joe / is_joe.sum())

# print('False and True accesses ratio from Joe:', ', '.join('{0:.1%}'.format(i) for i in is_joe))


# In[38]:


# user_id_like_joe = set(df_like_joe['user_id'])
# print(len(user_id_like_joe), 'total of user_id with same logs than Joe:', *user_id_like_joe)


# Despite the fact that the filter previously mentioned has efficiently removed most of the dataset, there are yet some logs from a few people with enough occurencies to be a majority over Joe. This is yet something to be tackled, since we don't want these people being taken as Joe.

# ## 3.2 Decision Tree

# Maybe, the cross combination of restrictions across different features is enough to find Joe out of the other few people above. For instance, Joe might be the only one who uses Firefox (`browser`) on Windows 10 (`os`).
# 
# Let's create a simple Decision Tree, train it on the single-entries categorical features and check it's performance to detect Joe.

# In[39]:


def categorize(df, features):

    df['hour'] = extract_hour(df)
    
    df[features] = df[features].astype('category')

    le = {feat: LabelEncoder().fit(df[feat]) for feat in features}
        
    return df, le


def encode_features(df, features, le):

    for feat in features:
        df[feat] = le[feat].transform(df[feat])
        
    return df
        

def encode_joe(is_joe_bool_list, encode_dict={True: user_id_joe, False: 1}):
    return [encode_dict[is_joe] for is_joe in is_joe_bool_list]


# In[40]:


def transform_features(df, features, features_categorical):
    
    df = df.copy()
    
    # add some features
    new_features = {
        'duration': extract_duration,
        'hour': extract_hour,
        'sites_ratio': extract_sites_ratio,
        'site_old': extract_site_old,
    }
    
    for feat_name, feat_func in new_features.items():
        if feat_name in features: df[feat_name] = feat_func(df)

    # convert features into category type
    df[features_categorical] = df[features_categorical].astype('category')
    df = encode_features(df, features_categorical, le)    
            
    # add Joe's top site lengths as features
    df = df.join(extract_lengths(df, joe_top_sites))
    df[joe_top_sites] = scaler.transform(df[joe_top_sites])
    
    
    return df[features]


# In[41]:


df_train = df.copy()

df_train, le = categorize(df_train, features_categorical)

y_train = encode_joe(df_train['user_id'] == user_id_joe)
df_train = transform_features(df_train, features, features_categorical)
X_train = df_train[features].values


# Before creating our predictive model, let's calculate the Naïve performance. We know that the majority of the data is not from Joe so the Naïve classifier always assume that the result is 1 (not Joe).

# In[42]:


def print_scores(y_pred, y_test):
    print('{0:.4%}'.format(accuracy_score(y_pred, y_test)), 'is the accuracy of the classifier.')
    print('{0:.2%}'.format(recall_score(y_pred, y_test, pos_label=user_id_joe)), 'of the Joe\'s accesses are detected.')
    print('{0:.2%}'.format(precision_score(y_pred, y_test, pos_label=user_id_joe)), 'of the detections are truly from Joe.')


y_pred = [1] * len(y_train)

print('Performance of Naïve on train dataset:')
print_scores(y_pred, y_train)


# As expected, the accuracy score is quite high because most of the data is not Joe (imbalanced). Moreover, precision and recall scores are obviously nulls because the Naïve blindly guessed it always as not Joe.

# In[43]:


# train model
model = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)

print('Performance on train dataset:')
y_pred = model.predict(X_train)
print_scores(y_pred, y_train)


# In[44]:


# prepare test dataset
df_test = df_later.copy()
y_test = encode_joe(df_test['user_id'] == user_id_joe)
df_test = transform_features(df_later, features, features_categorical)
X_test = df_test[features].values


# In[45]:


print('\nPerformance on test dataset:')
y_pred = model.predict(X_test)
print_scores(y_pred, y_test)


# The Decision Tree presents a slight increase of performance when compared to the Naïve. However, it is far from excellent since there are too many missed accesses from Joe.

# Before we move on to improve the performance, here is a question. How exactly does this Decision Tree above work in order to classify?
# 
# In order to help us answer this question, let's plot the nodes of the Decision Tree as a graph plot below.

# In[46]:


# dtreeviz(model, X_train, y_train,
#                 target_name="target",
#                 feature_names=features,
#                 class_names=['Joe', 'not Joe'])


# In[47]:


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

# In[54]:


# score_function = accuracy_score
score_function = f1_score

models = [
    DecisionTreeClassifier(),
    AdaBoostClassifier(),
    BaggingClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    
#     KNeighborsClassifier(n_neighbors=200),
    
#     QuadraticDiscriminantAnalysis(),
#     GaussianProcessClassifier(),
#     GaussianProcessClassifier(1.0 * RBF(1.0)),
#     SVC(),    
#     SVC(kernel="linear", C=0.025),
#     SVC(gamma=2, C=1),
    
#     MLPClassifier(),
]

print('Test Score\t Model')

score_best = -np.Inf
best_model = None
for i_model, model in enumerate(models):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = score_function(y_pred, y_test)
        
    print('{0:.4%}\t'.format(score), type(model).__name__)
    
    if score > score_best:
        score_best = score
        best_model = model
    


print('\nBest model:', type(best_model).__name__)

print('\nPerformance on train dataset:')
y_pred = best_model.predict(X_train)
print_scores(y_pred, y_train)

print('\nPerformance on test dataset:')
y_pred = best_model.predict(X_test)
print_scores(y_pred, y_test)


# In[56]:


# parameters = {
#     'n_estimators': [20, 40, 100],
# #     'max_features': ['auto', 'sqrt', 'log2']
#     'criterion': ['gini', 'entropy'],
# }

# base_model = RandomForestClassifier()
# final_model = GridSearchCV(base_model, parameters, scoring=make_scorer(f1_score))
# final_model.fit(X_train, y_train)

# print("Best parameters from gridsearch: {}".format(final_model.best_params_))

# print('\nPerformance on train dataset:')
# y_pred = final_model.predict(X_train)
# print_scores(y_pred, y_train)

# print('\nPerformance on test dataset:')
# y_pred = final_model.predict(X_test)
# print_scores(y_pred, y_test)


# The new classifier performance is much better than the previous one to the point that it can be deployed.

# # 4 Save

# ## 4.1 Running the model on verify dataset

# In[51]:


# load the input file
df_verify = pd.read_json('./data/verify.json')

df_verify = transform_features(df_verify, features, features_categorical)

y_pred = model.predict(df_verify.values)
count = Counter(y_pred)
print(count)
percentage = count[0] / (count[1] + count[0])
print('{0:.2%} of the Verification dataset is detected as Joe\'s access.'.format(percentage))


# ## 4.2 Exporting this Notebook

# The following code is to convert the present Jupyter Notebook into Python script. The script is the one under version control since we do not want to keep track of JSON codes internal to the `.ipynb` files.

# In[52]:


# convert Notebook to Python for better version control
# ! jupyter nbconvert --to script "Toptal Final Project.ipynb" --output-dir="./code/diogo-dutra"

