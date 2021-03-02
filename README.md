# Evaluating models with Python
### 1. Starting out
Import pandas to read in data
```
import pandas as pd
import numpy as np
```
Import matplotlib for plotting
```
import matplotlib.pylab as plt
%matplotlib inline
```

Import decision trees and logistic regression
```
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
```

Import train, test, and evaluation functions
```
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
```
### 2. Data
We're going to use a mail response data set from a real direct marketing campaign located in files/mailing.csv.

You can download the files from http://bit.ly/mailingData.
