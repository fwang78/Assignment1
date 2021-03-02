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

You can download the files from repository

Our goal is to build a model to predict if people will give during the current campaign (this is the attribute called "class").

Read data using pandas
```
data = pd.read_csv("files/mailing.csv")
```
Split into X and Y
```
X = data.drop(['class'], 1)
Y = data['class']
data.head()
X.head()
Y.head()
```
### 3. Overfitting
Create an empty, unlearned tree
```
tree = DecisionTreeClassifier(criterion="entropy")
```
Fit/train the tree
```
tree.fit(X, Y)
```
Get a prediction
```
Y_predicted = tree.predict(X)
```
Get the accuracy of this prediction
```
accuracy = accuracy_score(Y_predicted, Y)
```
Print the accuracy
```
print("The accuracy is " + str(accuracy))
```
```
Y.mean()
```
95% of the people do not donate --> 99.5% accuracy is pretty good (much higher than the base rate of 95%).

However, we might be overfitting our data. The model might have "memorized" where all the points are. This does not lead to models that will generalize well.

We can create training and testing sets very easily. Here we will create train and test sets of X and Y where we assign 70% of our data to training.

Split X and Y into training and test
```
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.70)
```

Now, let's look at the same decision tree but fit it with our training data and test it on our testing data.


Create an empty, unlearned tree
```
tree = DecisionTreeClassifier(criterion="entropy")
```
Fit/train the tree on the training data
```
tree.fit(X_train, Y_train)
```
Get a prediction from the tree on the test data
```
Y_test_predicted = tree.predict(X_test)
```
Get the accuracy of this prediction
```
accuracy = accuracy_score(Y_test_predicted, Y_test)
```
Print the accuracy
```
print("The accuracy is " + str(accuracy))
```

Let's also use cross validation with 5 folds to see how well our model performs.
