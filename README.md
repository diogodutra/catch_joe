![Hello 👋🏻](https://cdn.pixabay.com/photo/2017/06/04/16/32/hacker-2371490__340.jpg)

<p>&nbsp;</p>

# &#x2615; Welcome to Catch Joe project,

We want to identify which accesses on internet are from Joe (`user_id == 0`) by the following session information: browser, gender, date, time, addresses and length of the websites visited.

For this purpose, a predictive model was created from [this dataset](https://drive.google.com/file/d/1nATkzOZUe6w5IWcFNE3AakzBl-6P-5Hw/view?usp=sharing).


<p>&nbsp;</p>

# &#x1f4c8; Data Analysis and Predictive Model

The detailed walkthough about the exploratory data analysis (ie: plots, insights and feature analysis) and the predictive model creation (ie: definition of metrics, training, testing) are detailed on [this notebook](https://git.toptal.com/screening/diogo-dutra/blob/master/catch_joe_project.ipynb).


<p>&nbsp;</p>

# &#128187; Standalone script

The predictive model to identify Joe is [available here](https://git.toptal.com/screening/diogo-dutra/blob/master/catch_joe_project.py) to run as a standalone Python script over a JSON file. It is assumed that the environment is properly setup (Anaconda) with all necessary modules installed (NumPy, Pandas and SciKit-Learn). It is also necessary to download the model parameters [from here](https://git.toptal.com/screening/diogo-dutra/tree/master/model) to a local subfolder named `model`. Then, you can run it from the terminal:
```
python catch_joe.py -j ./data/verify.json 
```
Check its help function for more details on how to use the standalone script.
```
python catch_joe.py -h
```

<p>&nbsp;</p>

# 📬 Get in touch

Feel free to contact me at anytime should you need further information about this project or for any other Machine Learning and Data Scientist:
- &#128100; Personal Web: diogodutra.github.io
- ![](https://i.stack.imgur.com/gVE0j.png) LinkedIn: linkedin.com/in/diogodutra