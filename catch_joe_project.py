#!/usr/bin/env python
# coding: utf-8

# # Project Catch Joe

# ## Index
# 
# 1. Introduction  
#     1.1 Import modules  
#     1.2 Load dataset  
#     
#     
# 2. Data Exploration  
#     2.1 Location  
#     2.2 Gender  
#     2.3 Language  
#     2.4 Operating System  
#     2.5 Browser  
#     2.6 Time of the day  
#     2.7 Day of the week  
#     2.8 Day of the month  
#     2.9 Month of the year  
#     2.10 Duration  
#     2.11 Sites  
#     
#     
# 3. Predictive Model  
#     3.1 Naïve Analysis  
#     3.2 Decision Tree  
#     3.3 Random Forest
#   
#   
# 4. Save  
#     4.1 Storage model parameters  
#     4.2 Run on Verify dataset  
#     4.3 Export this Notebook to Python code

# # &#x2615;  1 Introduction

# ## 1.1 Context

# The present Jupyter Notebook explains the process of creating a predictive model to identify an user access as **Joe** or **not-Joe** using this [dataset](https://drive.google.com/file/d/1nATkzOZUe6w5IWcFNE3AakzBl-6P-5Hw/view?usp=sharing).
# 
# The dataset contains data about user sessions that have been recorded over a period of time. The dataset consists of two parts: the training dataset where user ID’s are labeled, and the verification set without labels.
# 
# Each session is represented by a JSON object with the following fields:
# - `user_id` is the unique identifier of the user.
# - `browser`, `os`, `locale` contain info about the software on the user’s machine.
# - `gender`, `location` give analytics data about the user.
# - `date` and `time` is the moment when the session started (in GMT).
# - `sites` is a list of up to 15 sites visited during the session. For each site, the url and the length of visit in seconds are given.

# ## 1.1 Import modules

# In[1]:


import catch_joe
from catch_joe import         extract_duration, extract_hour_local, extract_lengths, extract_sites_ratio,         categorize, encode_features, encode_joe, transform_features, print_scores


# In[2]:


import os
from dtreeviz.trees import dtreeviz
os.environ["PATH"] += os.pathsep + 'C:/Users/Diogo/anaconda3/Library/bin/graphviz'

from collections import Counter
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer

import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# ## 1.2 Load dataset

# The dataframe below is what all the information available in the JSON file.
# 
# It contains all the information in a typical JSON to be classified, except that this one has an extra column with the label `user_id` for training purpose. We will use it to create another column with booleans to explicit if it is related to `user_id = 0 `, nicknamed as **Joe**.

# In[4]:


file = './data/dataset.json'

df = pd.read_json(file)

# create target label as a separate column
user_id_joe = 0
df['joe'] = df['user_id'] == user_id_joe

print(df.shape)
df.head()


# Before everything, let's split the data now in order to guarantee that there is no _Data Leakage_ in the following phase during our data exploration and analysis. Mind that this is a time-series data, so we will split it in the chronological order (instead of random sampling). The later dataset will be used to test and score our predictive model.

# In[5]:


# split dataset into train and test
# in chronological order (not random)

test_ratio = 0.1
df_later = df.sort_values(by='date').tail(int(test_ratio * df.shape[0]))

df = df.drop(df_later.index) # train dataset

df.shape


# # &#x1f4c8; 2 Data Exploration

# Let's go through a series of plots throughout the dataset in order to extract insights and patterns that will be used by our predictive model later on.

# ## 2.1 Gender

# The first feature to be explored is `gender`. Let's explore the hypothesis that Joe never changed their gender.

# In[6]:


_ = sns.catplot(x='joe', hue='gender', kind='count', data=df)


# It seems that Joe is consistently identified as a male, since there is no log on his name as female. This insight is useful to discard around 40% of the dataset.

# ## 2.2 Location

# Now, let's now explore `location`. Unless Joe lives in a cruising ship, he probably has little change of location around the globe.

# In[7]:


_ = sns.catplot(x='joe', hue='location', kind='count', data=df)


# In[8]:


Counter(df[df['joe']]['location'])


# Joe accessed from Paris, Chicago and Toronto only. This is helpful to discard the other 18 locations.

# ## 2.3 Language

# Let's now explore `locale`. It is rare to find an active polyglot so let's explore this hypothesis.

# In[9]:


_ = sns.catplot(x='joe', hue='locale', kind='count', data=df)


# In[10]:


Counter(df[df['joe']]['locale'])


# Despite the fact that Joe has many access from France, USA and Canada, his sessions are always in Russian language. Again, this eliminates all the other languages.

# ## 2.4 Operating System

# If Joe is not a geek than he is probably using only one or two different `os`.

# In[11]:


_ = sns.catplot(x='joe', hue='os', kind='count', data=df)


# Indeed, Joe uses only Ubuntu and Windows 10. This rules out MacOS, Debian and the rest of the Microsoft's OS.

# ## 2.5 Browser

# For the same reason explained before for the OS, Joe is probably using only a couple of `browsers`.

# In[12]:


_ = sns.catplot(x='joe', hue='browser', kind='count', data=df)


# Joe always uses Firefox and Chrome, ruling out Internet Explorer and Safari.

# ## 2.6 Time of the day

# Let's now verify the hypothesis that Joe accesses internet only in some specific hours of the day. Mind that there is a conversion from GMT to local timezone.

# In[13]:


df['hour'] = extract_hour_local(df)

# plot
sns.distplot(df['hour'],            norm_hist=True, kde=False, rug=False, bins=24)
sns.distplot(df[df['joe']]['hour'], norm_hist=True, kde=False, rug=False, bins=24)
plt.legend(['all', 'joe'])
plt.title('Histogram of accesses per hour of the day')
_ = plt.ylabel('density of occurrencies')


# Joe accesses internet only during lunch or dinner. Therefore, hour of access is yet another relevant information to be used by our classifier.

# ## 2.7 Day of the week

# Following the rationale from the previous subsection, let's verify the hypothesis that Joe accesses internet only in some specific days of the week.

# In[14]:


df['weekday'] = [date.day_name() for date in df['date']]

_ = sns.catplot(x='joe', hue='weekday', kind='count', data=df)


# In[15]:


Counter(df[df['joe']]['weekday'])


# Unfortunately, there is no particular day of the week that shows an unusual occurency of accesses from Joe, so let's drop this feature.

# ## 2.8 Day of the month

# Let's verify if Joe has different frequency of access along the days of the month.

# In[16]:


df['monthday'] = [date.day for date in df['date']]

_ = sns.catplot(x='joe', hue='monthday', kind='count', data=df)


# There is no unusual pattern to be extracted out of the day of the month.

# ## 2.9 Month of the year

# Now, let's check if there is any useful pattern along the months of the year.

# In[17]:


df['month'] = [date.month for date in df['date']]

_ = sns.catplot(x='joe', hue='month', kind='count', data=df)


# Again, nothing useful from the month of the year.

# ## 2.10 Duration

# In[18]:


df['duration'] = extract_duration(df)

sns.distplot(df           ['duration'])
sns.distplot(df[df['joe']]['duration'])
plt.xlabel('duration of each session')
plt.ylabel('probability density')
_ = plt.legend(['all', 'Joe'])


# Joe's duration of access fits well within the statistical boundaries of the whole population, which means that there is nothing unusual. Nonetheless, let's keep this feature since it is slightly off the population statistics so it might be useful when correlated with other features (ie: he might access longer during dinner than lunch).

# # 2.11 Sites

# The list of sites accessed per session and its respective lengths do sound like a pretty relevant information. We all have our own favorite sites that we access most often by far from the others. If such data is the user's "digital footprint" then their "digital gait" could be used to identify them, hopefully including Joe.

# In[19]:


sites_joe = {site.get('site') for sites in df[df['joe']]['sites'] for site in sites}
print(len(sites_joe), 'sites accessed by Joe.')


# In[20]:


sites_joe_list = list(sites_joe)    
df_sites_joe_length = extract_lengths(df[df['joe']], sites_joe_list)
df_sites_joe_length = df_sites_joe_length.mean().sort_values(ascending=False)

print("Most and least sites accessed by Joe:")
print('\nADDRESS \t AVERAGE LENGTH OF SESSION')
df_sites_joe_length


# In[21]:


df_sites_all_length = extract_lengths(df, sites_joe_list)

print("Most and least sites accessed by the whole population:")
print('\nADDRESS \t AVERAGE LENGTH OF SESSION')
df_sites_all_length.mean().sort_values(ascending=False)


# The sites most accessed by Joe have links and lengths different than the population. So, this information is useful as well.
# 
# However, let's extract as features for our predictive model only the top ones in order to avoid overfitting and heavy computational cost.

# In[22]:


top_sites = 50
joe_all_sites = list(df_sites_joe_length[:top_sites].index)
joe_top_sites = joe_all_sites[:top_sites]


# # &#128187; 3 Predictive Model

# ## 3.1 Naïve Bayes Analysis

# The previously mentioned features are good enough to safely tell whenever is not Joe.
# 
# What if they are good enough to be queried independentely as filters to discard parts of the data until we have only Joe left?
# 
# In orde to find this out, let's check how many logs match exactly the Joe's history from the categorical features. 

# In[23]:


# define list of features to be used by the classifier

features_categorical = ['gender', 'os', 'browser', 'location', 'locale', 'hour']

features = features_categorical + ['duration'] + joe_top_sites


# In[24]:


df_like_joe = df.copy()

filter_data = {feat: set(df[df['joe']][feat]) for feat in features_categorical}

for feature, valid_entries in filter_data.items():
    df_like_joe = df_like_joe[df_like_joe[feature].isin(valid_entries)]

    
# extract set of multiple website entries from Joe's logs
sites_joe = {site.get('site') for sites in df_like_joe['sites'] for site in sites}
df_like_joe = df_like_joe[list(map(lambda x:
                any(site.get('site') in sites_joe for site in x), df_like_joe['sites']))]
    
    
print('Original dataset contains', df.shape[0], 'logs.')
print('Like-Joe dataset contains', df_like_joe.shape[0], 'logs',
     "({0:.0%}).".format(df_like_joe.shape[0] / df.shape[0]))


# Filtering the sessions that do not match Joe's history is enough to discard a large piece of the dataset.
# 
# But how many of the left logs are our Joe indeed?

# In[25]:


count = Counter(df_like_joe['joe'])
print(count)

is_joe = np.asarray([count[False], count[True]])
is_joe = list(is_joe / is_joe.sum())

print('False and True sessions ratio from Joe:', ', '.join('{0:.1%}'.format(i) for i in is_joe))


# In[26]:


user_id_like_joe = set(df_like_joe['user_id'])
print(len(user_id_like_joe), 'total of user_id with same logs than Joe:', *user_id_like_joe)


# Despite the fact that the filter above efficiently removed most of the dataset, there are yet some sessions from a few people with enough occurencies to be a majority over Joe. This is yet something to be tackled, since we don't want these people being taken as Joe.

# Before creating our predictive model in the next subsection, let's calculate the Naïve performance. We know that the majority of the data is not from Joe so the Naïve classifier always assume that the result is 1 (not Joe).

# In[27]:


df_train = df.copy()

df_train, le = categorize(df_train, features_categorical)

y_train = encode_joe(df_train['user_id'] == user_id_joe)
df_train = transform_features(df_train, features, features_categorical, joe_top_sites, le)
X_train = df_train[features].values


# In[28]:


y_pred = [1] * len(y_train)

print('Performance of Naïve on train dataset:')
print_scores(y_pred, y_train)


# As expected, the accuracy score is quite high because the data is largely imbalanced with not-Joe sessions.
# 
# Moreover, precision and recall scores are obviously nulls because the Naïve blindly guesses as not Joe.
# 
# For these reasons, **our chosen metric will be the F1-score**, resulting in:
# - 0% F1-score is our reference as benchmarked by our Naïve guess; and
# - Both precision and recall scores are going to be equally considered.

# ## 3.2 Decision Tree

# Maybe, the correlation of every restriction is better enough to find Joe out of the other few people above. For instance, Joe might be the only one who uses Firefox (`browser`) on Windows 10 (`os`).
# 
# Let's create a simple Decision Tree, train it on the single-entries categorical features and check it's performance to detect Joe.

# In[29]:


# train model
n_features_categorical = len(features_categorical)
model = DecisionTreeClassifier(max_depth=3).fit(X_train[:, :n_features_categorical], y_train)

print('Performance on train dataset:')
y_pred = model.predict(X_train[:, :n_features_categorical])
print_scores(y_pred, y_train)


# The Decision Tree presents a slight increase of performance when compared to the Naïve guess. However, it is far from excellent since there are too wrong detections. Therefore, let's try other more sophisticated models aiming for a improvement in the performance (higher F1-score).

# Before we move on to improve the performance, here is a question. How exactly does this Decision Tree above work in order to classify?
# 
# In order to help us answer this question, let's plot the nodes of the Decision Tree as a graph plot below.

# In[30]:


dtreeviz(model, X_train[:, :n_features_categorical], np.asarray(y_train),
                class_names=['Joe', 'not Joe'],
                feature_names=features_categorical,
        )


# In[31]:


le['locale'].inverse_transform([18])


# The graph above shows that the Decision Tree queries the features in the following order:
# 1. If the language (`locale`) is less than 17.5 then is not Joe with 100% of certainty; else ...
# 1. If the language (`locale`) is more than 18.5 then is is not Joe with 100% of certainty; else ...
# 1. If the (encoded) location of access is less than 3.5 than it is Joe with 100% of certainty; else ...
# 1. We run out of questions so it guesses it is not Joe with roughly 10% of error.
# 
# Mind that `locale = 18` is the Russian language as coded by the `LabelEncoder`. Therefore, the first 2 questions above are mainly telling us that if the language of access is not Russian than it is not 
# Joe for sure.

# ## 3.3 Random Forest

# Finally, now we create our final predictive model.

# Let's start by preparing the test dataset following the same feature extraction pipeline as we did for the training dataset.

# In[32]:


# prepare test dataset
df_test = df_later.copy()
y_test = encode_joe(df_test['user_id'] == user_id_joe)
df_test = transform_features(df_later, features, features_categorical, sites_joe_list, le)
X_test = df_test[features].values


# Now, let us instantiate and try some more sophisticted models candidates in order to select the one with best performance.

# In[33]:


# score_function = accuracy_score
score_function = lambda x, y: f1_score(x, y, pos_label=user_id_joe)

models = [
    DecisionTreeClassifier(),
    AdaBoostClassifier(),
    BaggingClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    
    # these below were not included because
    # they take a long time and they
    # ended up not being selected anyway
    
#     KNeighborsClassifier(n_neighbors=200),    
#     QuadraticDiscriminantAnalysis(),
#     GaussianProcessClassifier(),
#     GaussianProcessClassifier(1.0 * RBF(1.0)),
#     SVC(),    
#     SVC(kernel="linear", C=0.025),
#     SVC(gamma=2, C=1),
#     MLPClassifier(),
]

print('Test F1-score\t Model')

score_best = -np.Inf
best_model = None
for i_model, model in enumerate(models):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = score_function(y_pred, y_test)
        
    print('{0:.3%}\t'.format(score), type(model).__name__)
    
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


# The best classifier performance trained above is much better than the previous Decision Tree to the point that it can be deployed.
# 
# The best classifier is the Random Forest, which is an ensemble learning method that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) of the individual trees. It is less prone to overfitting while often presenting a better accuracy.
# 
# It is utterly important to retrain the predictive model whenever Joe changes his session pattern. Otherwise the performance will dive. For instance, if Joe starts a new session with another language other than Russian, this will cause the classifier to miss him since it is an unprecedented behavior that was obviously not considered during the training above. This is a common limitation across all predictive models, since all future inferences are assumed to be a good extrapolation of past learned patterns.

# # &#x1f4be; 4 Save

# ## 4.1 Export the model parameters

# It is time to store the model parameters necesary to run a standalone script defined in the `catch_joe` module.

# In[34]:


catch_joe_model_parameters = {
    'model': model,
    'encoder': le,
    'joe_top_sites': joe_top_sites,
    'features': features,
    'features_categorical': features_categorical,
}

with open('./model/catch_joe.pickle', 'wb') as f:
    pickle.dump(catch_joe_model_parameters, f)


# ## 4.2 Run on Verify dataset

# Let's use the latest model saved above to run a standalone script.

# In[35]:


# below is how to run the script through terminal command in here
# !python catch_joe.py -j ./data/verify.json


# alternatively, let's run it from the Jupyter Notebook
# so we get y_pred to print some results below
y_pred = catch_joe.main(file_json = './data/verify.json')

count = Counter(y_pred)
print(count)
percentage = count[0] / (count[1] + count[0])
print('{0:.2%} of the predictions are detected as Joe\'s accesses.'.format(percentage))


# We do not have the correct answers (`user_id`) in the `verify.json` file. However, just for the sake of curiosity, we can observe from the results above that our predictive model predicts Joe's sessions as the minority of the times. Such figure is expected for a real dataset that is expected to contain hundreds of users.
# 
# There is a small drop in Joe's presence in the verify dataset compared to both the train and test. This might be an indication that the classifier is missing some of his sessions because of an unexpected change of Joe's behavior (problem explained in the subsection). If that is the case then it would demand a further training with the new data to fix this.

# 
# ## 4.3 Export to Python code

# The following code serves to convert the present Jupyter Notebook into Python code. This exported `.py` code is aimed to facilitate version control and tracking of "Python only" changes since it does not contain HTML nor JSON codes that rae typically present in the `.ipynb` files.

# In[62]:


# convert Notebook to Python for better version control
get_ipython().system(' jupyter nbconvert --to script "catch_joe_project.ipynb"')

