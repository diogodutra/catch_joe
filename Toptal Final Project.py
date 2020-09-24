#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[4]:


file = 'data/dataset.json'

df = pd.read_json(file)

df.head()


# In[23]:


# convert Notebook to Python for better version control
get_ipython().system(' jupyter nbconvert --to script "Toptal Final Project.ipynb" --output-dir="./code/diogo-dutra"')

