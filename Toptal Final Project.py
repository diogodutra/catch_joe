#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter


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


# Joe has access logs from Paris, Chicago and Toronto only. This is helpful to disregard the other 18 locations.

# ## Language

# Let's now explore `locale`. It is rare to find an active polyglot so let's explore this hypothesis.

# In[8]:


sns.catplot(x='joe', hue='locale', kind='count', data=df)


# In[9]:


Counter(df[df['joe']]['locale'])


# Despite the fact that Joe has many access from France, USA and Canada, his sessions are always in Russian language. Again, this eliminates all the other languages.

# In[10]:


sns.catplot(x='joe', hue='os', kind='count', data=df)


# Indeed, Joe uses only Ubuntu and Windows 10. This rules out MacOS, Debian and the rest of the Microsoft's OS.

# In[11]:


sns.catplot(x='joe', hue='browser', kind='count', data=df)


# Again, Joe uses only Firefox and Chrome, leaving out Internet Explorer and Safari.

# ## Browser

# For the same reason explained above for the OS, Joe is probably using only a couple of `browsers`.

# ## Operating System

# If Joe is not a geek than he is probably using only one or two different `os`.

# # Save

# In[7]:


# convert Notebook to Python for better version control
get_ipython().system(' jupyter nbconvert --to script "Toptal Final Project.ipynb" --output-dir="./code/diogo-dutra"')

