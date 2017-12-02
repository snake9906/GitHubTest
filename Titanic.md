
# Titanic Classification Problem:
## Machine Learning, Randomized Tuning, Ensemble Voting, Pipelines, and Stacking Models
## Generative/ Discriminative Models, Nonparametric and Parametric Models, Dimensionality Reduction, Regularization

Author: Nick Brooks 

Date: November 2017

The goal of this notebook is to showcase the wide array of models through the Sklearn wrapper. Showcasing how to tune them using randomized search. Note: In practice, it is usually better to focus one’s energy on a high performance model, and thoroughly develop its hyperparameters, ensure quality feature selection and engineering.

Since such a range of models are explored, I also take the opportunity to explain the axis through which these models differ: **Parametric vs Non-Parametric**, and **Generative vs. Discriminative**

Next, I add more flavor to the modeling process through including **Ensemble Voting**, **Pipelining Dimensionality Reduction**, and **Stacking**.

- **UPDATE 1**: Fixing the best mode for underperformance, uploaded wrong code!
- **UPDATE 2**: Cannot reproduce optimal score, will have to gear notebook more towards reproducibility in the future.


```python
# General Packages
import pandas as pd
import numpy as np
import random as rnd
import os
import re
# import multiprocessing

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (16, 8)
import scikitplot as skplt

# Machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import feature_selection

# Unsupervised
from sklearn.decomposition import PCA

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

# Evalaluation
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Grid
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as st

# Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

# Esemble Voting
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score

# Stacking
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from matplotlib.colors import ListedColormap

# Warnings
import warnings
warnings.filterwarnings('ignore')

import time
import datetime
import platform
start = time.time()
```


```python
import platform

print('Version      :', platform.python_version())
print('Compiler     :', platform.python_compiler())
print('Build        :', platform.python_build())

print("\nCurrent date and time using isoformat:")
print(datetime.datetime.now().isoformat())
```


```python
# Master Parameters:
n_splits = 5 # Cross Validation Splits
n_iter = 80 # Randomized Search Iterations
scoring = 'accuracy' # Model Selection during Cross-Validation
rstate = 23 # Random State used 
```

## Loading and Pre-Processing
### Cleaning:
- Category to Numerical representation
- Dealing with Missing Values, and Filling NA


### Feature Selection:
- Extracting Titles from name variable to use as feature
- Rescaling continuous variables


```python
train_df = pd.read_csv("../input/train.csv", index_col='PassengerId')
test_df = pd.read_csv("../input/test.csv", index_col='PassengerId')
# train_df = pd.read_csv("Titanic/Data/train.csv", index_col='PassengerId')
# test_df = pd.read_csv("Titanic/Data/test.csv", index_col='PassengerId')

Survived = train_df['Survived'].copy()
train_df = train_df.drop('Survived', axis=1)
df = pd.concat([test_df, train_df])
traindex = train_df.index
testdex = test_df.index
del train_df
del test_df

# New Variables: Kaggle Source- https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
# Family Size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
# Name Length
df['Name_length'] = df['Name'].apply(len)
# Is Alone?
df['IsAlone'] = 0
df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

# Title: Kaggle Source- https://www.kaggle.com/ash316/eda-to-prediction-dietanic
df['Title']=0
df['Title']=df.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
df['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col',
                         'Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

df.loc[(df.Age.isnull())&(df.Title=='Mr'),'Age']= 33
df.loc[(df.Age.isnull())&(df.Title=='Mrs'),'Age']=36
df.loc[(df.Age.isnull())&(df.Title=='Master'),'Age']=5
df.loc[(df.Age.isnull())&(df.Title=='Miss'),'Age']=22
df.loc[(df.Age.isnull())&(df.Title=='Rare'),'Age']=46
df = df.drop('Name', axis=1)
# Fill NA
# Categoricals Variable
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0])
# Continuous Variable
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

## Assign Binary to Sex str
df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
# Title
#df['Title'] = df['Title'].map( {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master':3, 'Rare':4} ).astype(int)
# Embarked
df['Embarked'] = df['Embarked'].map( {'Q': 0, 'S': 1, 'C': 2} ).astype(int)

# Get Rid of Ticket and Cabin Variable
df= df.drop(['Ticket', 'Cabin'], axis=1)
```


```python
# Viz
# plt.rcParams['figure.figsize'] = (9, 9)
# Histogram
pd.concat([df.loc[testdex, :], Survived], axis=1).hist()
plt.show()
# Correlation Plot
sns.heatmap(pd.concat([df.loc[testdex, :], Survived], axis=1).corr(), annot=True, fmt=".2f")
```


```python
# Finish Pre-Processing
# Dummmy Variables
df = pd.get_dummies(df, columns=['Embarked','Title','Parch','SibSp','Pclass'], )

# Scaler
from sklearn import preprocessing
for col in ['Fare','Age','Name_length']:
    transf = df.Fare.reshape(-1,1)
    scaler = preprocessing.StandardScaler().fit(transf)
    df[col] = scaler.transform(transf)

# Split Data Up Again
# Train
train_df = df.loc[traindex, :]
train_df['Survived'] = Survived
# Test
test_df = df.loc[testdex, :]
```

## Take Your Position!
Declaring dependent and independent variables as well as helper functions that record the mean cross-validation average of model accuracy, and its standard deviation to understand the volatility of the models.


```python
X = train_df.drop(["Survived"] , axis=1)
y = train_df["Survived"]

print("X, Y, Test Shape:",X.shape, y.shape, test_df.shape)

results = pd.DataFrame(columns=['Model','Para','Test_Score','CV Mean','CV STDEV'])
ensemble_models= {}

# http://scikit-plot.readthedocs.io/en/stable/metrics.html
def eval_plot(model):
    y_probas = model.predict_proba(X_test)
    skplt.metrics.plot_roc_curve(y_test, y_probas)
    plt.show()
       
def save(model, modelname,proba=None):
    global results
    model.best_estimator_.fit(X, y)
    submission = model.predict(test_df)
    df = pd.DataFrame({'PassengerId':test_df.index, 
                           'Survived':submission})
    
    scores = cross_val_score(model.best_estimator_, X_train, y_train, cv=5,
                             scoring=scoring, verbose =0)
    CV_scores = scores.mean()
    STDev = scores.std()
    Test_scores = model.score(X_test, y_test)


    # CV and Save scores
    results = results.append({'Model': modelname,'Para': model.best_params_,'Test_Score': Test_scores,
                             'CV Mean':CV_scores, 'CV STDEV': STDev}, ignore_index=True)
    ensemble_models[modelname] = model.best_estimator_
    
    print("\nEvaluation Method={}".format(scoring))
    print("Optimal Model Parameters: {}".format(grid.best_params_))
    print("Train CV Accuracy: %0.2f (+/- %0.2f) [%s]" % (CV_scores, STDev, modelname))
    print('Test_Score:', Test_scores)
        
    #Scikit Confusion Matrix
    pred = model.predict(X_test)
    skplt.metrics.plot_confusion_matrix(y_test, pred, title="{} Confusion Matrix".format(modelname),
                normalize=True,figsize=(6,6),text_fontsize='large')
    plt.show()
    # Colors https://matplotlib.org/examples/color/colormaps_reference.html

def norm_save(model,score, modelname,proba=None):
    global results
    model.fit(X, y)
    submission = model.predict(test_df)
    df = pd.DataFrame({'PassengerId':test_df.index, 
                           'Survived':submission})
    
    CV_Score = score.mean()
    Test_scores = model.score(X_test, y_test)
    STDev = score.std()
    
    # CV and Save Scores
    Test_Score = model.score(X_test, y_test)
    results = results.append({'Model': modelname,'Para': model,'Test_Score': Test_scores,
                             'CV Mean': CV_Score, 'CV STDEV': STDev}, ignore_index=True)
    ensemble_models[modelname] = model
    
    print("\nEvaluation Method={}".format(scoring))
    print("Train CV Accuracy: %0.2f (+/- %0.2f) [%s]" % (CV_Score, STDev, modelname))  
    print('Test_Score:', Test_scores)
        
    #Scikit Confusion Matrix
    pred = model.predict(X_test)
    skplt.metrics.plot_confusion_matrix(y_test, pred, title="{} Confusion Matrix".format(modelname),
                normalize=True,figsize=(6,6),text_fontsize='large')
    plt.show()
```

## Imbalanced Dependent Variable
This signifies that there is an unequal occurrence of the dependent variable "Survived". This could potentially lead to a flawed model. I deal with this by stratifying the train/test groups, leading to equal representation of classes. Additional methods to handle imbalanced datasets include data augmentation and re-sampling methods.


```python
print(y.value_counts(normalize=True))
# 0 = Dead
# 1 = Survived
```

## Train/Test
Declaring dependent and independent variables, as well as preparing the training data into its train-validation-test sets for proper modelling. My validation step is executed during the model fitting process.

Note: Determining model quality through cross-validation is not correct if hyper-parameter tuning is applied, since this leads to a (hyper-parameter) overfitted evaluation of the model. Use an additional untouched test set.

Stratified cross validation splits are used. Number of splits is declared at the start of notebook.


```python
# Stratified Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,random_state=rstate)
X_train.shape, y_train.shape, X_test.shape, y_test.shape

# Stratified Cross-Validation
cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=rstate)
```

# Non-Parametric 
Counterpart to Parametric models, Parametric models does not make any assumptions about the data generating process’ distribution. For example, in statistical test, Non-Parametric models utilize rank and medians, instead of the mean and variance! On the other hand, they function in a infinite space of parameters, making their name counter-intuitive, but also highlighting their practical approach to representation; enabling them to increase their flexibility indefinitely.

# Discriminative Models
Discriminative models do not attempt to quantify how the data generating process operates, instead, its goal is to slice and dice the data to classify the data, effectively solving the problem by modeling p(y|x). 

## K-Nearest Neighbors
The simplest of all machine learning models, a nonparametric method that works well on short and narrow datasets, but seriously struggles in the high dimensional space. It works by using the K nearest point to the predicted point vote on its class (or continuous number when in the regression context). Note: Since this is a instance based learning algorithm, its function is applied locally, and the computation is deferred until the prediction stage. Furthermore, since the data serves as the “map” for new values, the model size may be clunkier than its counterparts. 


```python
param_grid ={'n_neighbors': st.randint(1,40),
            'weights':['uniform','distance']
            }

grid = RandomizedSearchCV(KNeighborsClassifier(),
                    param_grid,cv=cv, scoring=scoring,
                    verbose=1, n_iter=n_iter, random_state=rstate)

grid.fit(X, y)

save(grid, "KNN")
```

### Stochastic Gradient Descent

ERROR 404. Adding.. Beep Bop


```python
SGDClassifier().get_params().keys()
```


```python
# param_grid ={'loss':["hinge","log","modified_huber","epsilon_insensitive","squared_epsilon_insensitive"]
#             }

# grid = GridSearchCV(SGDClassifier(),
#                     param_grid,cv=cv, scoring=scoring,
#                     verbose=1)

# grid.fit(X, y)
# save(grid, "StochasticGradientDescent")
```

# Decision Trees
I like to think of decision trees as optimizing a series of “If/Then” statements, eventually assigning the value at the tree’s terminal node. Starting from one point at the top, features can then pass the various trials until being assigned its class at the end “leaf” nodes of the tree (technically, it's more like its root tip since the end nodes are visually represented at the bottom of the graph). Trees used here are binary trees, so nodes can only split into two ways.


```python
# Helper Function to visualize feature importance
predictors = [x for x in train_df.columns if x not in ['Survived']]
def feature_imp(model):
    MO = model.fit(X,y)
    feat_imp = pd.Series(MO.feature_importances_, predictors).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
```


```python
DecisionTreeClassifier().get_params().keys()
```


```python
# Baseline Decision Tree
tree = DecisionTreeClassifier()
print("Mean CV Accuracy:",cross_val_score(tree, X, y, cv=cv, scoring=scoring).mean())
feature_imp(tree)
```


```python
# f, ax = plt.subplots(figsize=(9, 6))
# s = sns.heatmap(pd.crosstab(train_df.Sex, train_df.Title),
#             annot=True, fmt="d", linewidths=.5, ax=ax,
#                 cbar_kws={'label': 'Count'})
# s.set_title('Title Count by Sex Crosstab Heatmap')
# s.set_xticklabels(["Mr","Mrs","Miss","Master","Rare"]);
```

# Ensemble Method for Decision Trees

Technique where multiple trees are created and then asked to come together and vote on the model’s outcome. Later in this notebook, this same idea is applied to bring together models of different types.

The sub categories **Boosting** and **Bootstrap Aggregating** are part of this framework, but differ in terms of how the group of trees are *trained*.

## General Hyper-Parameters for Decision Trees and their Ensembles:
 *Sklearn implementation, but universal theory/parameters*

HyperParameters for Tuning:
- max_features: This is the random subset of features to be considered for splitting operation, the lower the better to reduce variance. For Classification model, ideal max_features = sqr(n_var). 
- max_depth: Maximum depth of the tree. Alternate method of control model depth is *max_leaf_nodes*, which limits the number of terminal nodes, effectively limiting the depth of the tree.
- n_estimators: Number of trees built before average prediction is made.
- min_samples_split: Minimum number of samples required for a node to be split. Small minimum would potentially lead to a “Bushy Tree”, prone to overfitting. According to Analytic Vidhya, should be around 0.5~1% of the datasize.
- min_samples_leaf: Minimum number of samples required at the **Terminal Node** of the tree. In Sklearn, an alternative *min_weight_fraction_leaf* is available to use fraction of the data instead of a fixed integer.
- random_state: Ensuring consistent random generation, like seed(). Important to consider for comparing models of the same type to ensure a fair comparison. May cause overfitting if random generation is not representative.

### Quality of Life:
n_jobs: Computer processors utilized. -1 signifies use all processors
Verbose: Amount of tracking information printed. 0 = None, 1 = By Cross Validation, 1 < by tree.

## Bootstrap aggregating (AKA Bagging) Decision Trees

Creates a bunch of trees using a subset of the the data for each, while using sampling without replacement, which means that values may be sampled multiple times. When combined with cross-validation, the values not sampled, or “held-out”, can then be used as the validation set. This results in a better generalizing model: Less prone to overfitting while maintaining a high model capacity (synonymous with model complexity and flexibility).

This technique introduces a flavor of the Monte Carlo method, which hopes to achieve better accuracy through sampling.

Reading:
https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/


```python
# Parameter Tuning
param_grid ={'n_estimators': st.randint(20, 500)}

tree = DecisionTreeClassifier()
grid = RandomizedSearchCV(BaggingClassifier(tree),
                    param_grid, cv=cv, scoring=scoring,
                    verbose=1,n_iter=n_iter, random_state=rstate)

grid.fit(X, y)
save(grid, "Bagger_ensemble")
```

## Random Forest
Builds upon Leo Breiman's Bootstrap Aggregation method by adding a random feature selection dimension. 
More uncorrelated splits, less overemphasis on certain features. Similar to Neural Net’s dropout, since it forces the model to give a large role to less dominant features, leading to a better generalizing model. However, it differs from dropout in the sense that its effect is not repeated and passed over to future iterations of the training process.


```python
model = RandomForestClassifier()
print("Mean CV Accuracy:",cross_val_score(model, X, y, cv=cv, scoring=scoring).mean())
feature_imp(model)
```


```python
RandomForestClassifier().get_params().keys()
```


```python
param_grid ={'max_depth': st.randint(6, 11),
             'n_estimators':st.randint(300, 500),
             'max_features':np.arange(0.5,.81, 0.05),
            'max_leaf_nodes':st.randint(6, 10)}

model= RandomForestClassifier()

grid = RandomizedSearchCV(model,
                    param_grid, cv=cv,
                    scoring=scoring,
                    verbose=1,n_iter=n_iter, random_state=rstate)

grid.fit(X, y)
save(grid, "Random_Forest")
```


```python
feature_imp(grid.best_estimator_)
```

## Boosting Family
Simply put, boosting models convert weak models into strong models. Essentially, each weak model is trained on a different distribution of the data, enabling it to learn a narrow rule. When combined, it offers a stronger model. Iteratively, subsequent weak models focus on the source of prediction error from the vote of previous weak models, until the accuracy ceiling is reached.

Source:
https://www.analyticsvidhya.com/blog/2015/11/quick-introduction-boosting-algorithms-machine-learning/

## Adaptive Boosting

Applies weights to all data points and optimizes them using the loss function. Fixes mistakes by assigning high weights to them during iterative process.

Iterates through multiple models in order to determine the best boundaries. It relies on using weak models to determine the pattern, and eventually creates a strong combination of them.


```python
AdaBoostClassifier().get_params().keys()
```


```python
param_grid ={'n_estimators':st.randint(50, 400),
            'learning_rate':np.arange(.1, 4, .5)}

grid = RandomizedSearchCV(AdaBoostClassifier(),
                    param_grid,cv=cv, scoring=scoring,
                    verbose=1, n_iter=n_iter, random_state=rstate)

grid.fit(X, y);
save(grid, "AdaBoost_Ensemble")
```


```python
feature_imp(grid.best_estimator_)
```


## Gradient Boosting Classifier

### Additional Gradient Boosting Hyper-Parameters:
Learning_rate: How much parameters are updated after each iteration of gradient descent. Low mean smaller steps, most likely to reach the global minimum, although there are cases where this doesn’t always work as intended.
N_estimators: Note this is still the number of trees being built, but it within the GBC’s sequential methodology.
Subsample: Similar to the Bootstrap Aggregate method, controlling percentage of data utilized for a tree, although the standard is to sample *without* replacement. In this model, operates in similar ways to Stochastic Gradient Descent in Neural Networks, but with large batch sizes. Note this decision trees are still a *Shallow Model*, so it is still profoundly different from Neural Networks.
Loss: Function minimized by gradient descent.
Init: Initialization of the model internal parameters. May be used to build off another model's’ outcome.

As the name suggest, Gradient Boosting iteratively trains models sequentially to minimize Loss, a convex optimization method similar to that seen in Neural Networks.

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3885826/

This is a Greedy Algorithm, which makes the most favorable split when it can. This means it's short sighted and will also stop when the loss doesn’t improve.

More information:

https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3885826/


```python
GradientBoostingClassifier().get_params().keys()
```


```python
param_grid ={'n_estimators':st.randint(100, 400),
            'loss': ['deviance', 'exponential'],
            'learning_rate':np.arange(0.01, 0.32,.05),
            'max_depth': np.arange(2, 4.1, .5)}

grid = RandomizedSearchCV(GradientBoostingClassifier(),
                    param_grid,cv=cv,
                    scoring=scoring,
                    verbose=1, n_iter=n_iter, random_state=rstate)

grid.fit(X, y)
save(grid, "Gradient_Boosting")
```


```python
feature_imp(grid.best_estimator_)
```

## XGBoost - eXtreme Gradient Boosting
As the name suggest, Gradient Boosting on steroids. Infact, multiple steroids:
Regularization
Computation Speed and Parallel Processing
Handles Missing Values
Improves on Gradient Boosting ‘*Greedy*’ tendencies. It does this by considering reaching the max depth and retroactively pruning 
Interruptible and Resumable, as well as Hadoop capabilities.

XGBoost is able to approximate the Loss function more efficiently, thereby leading to faster computation and parallel processing. Inside its objective function, it has also incorporated a regularization term, ridge or lasso, to stay clear of unnecessary high dimensional spaces and complexity.

### Hyper-Parameters for Tree Classification Booster (Sklearn Wrapper)
Min_child_weight: Similar strand to *min_child_leaf*, which controls the minimum number of observation to split to a terminal node. This instead relates to defines the minimum sum of derivatives found in the Hessian (second-order)of all observations required in a child.
Gamma: Node is split only if it gives a positive decrease in the loss function. Higher performance regulator of complexity.
Max_delta_step: What the tree’s weights can be.
Colsample_bylevel: Random Forest characteristic, where it enables subfraction of features to be randomly selected for each tree.

### Regularization
In the context of GBMs, regularization through shrinkage is available, but applying it to the model with the tree base-learner is different from its traditional coefficient constriction. For GBM Trees, the shrinking decreases the influence of each additional tree in the iterative process, effectively applying a decay of impact over boosting rounds. As a result, it no longer searches as the feature selection method, since it cannot adjust the coefficients of each feature (and potential interaction/ polynomial terms).


### Learning Task Parameters:
Objective: binary:logistic = Outputs probability for two classes. Multi:softmax = outputs prediction for *num_class* classes. Multi:prob = probability for 3 or more classes.
The objective is merely a matter of catering to the data type, although additional protocols on the probabilities may be set up. For example, the highest probability below a certain threshold may be deemed an inadequate score, passed over for human processing. The optimal loss function should cater to the behavior of the data and goal at hand, and usually requires input from the domain knowledge base.

The Evaluation Metric (*eval_metric*) has important implications depending on the problem at hand. If the dataset is imbalanced and the minority class is of interest, it is better to use [AUC] Area Under the Curve than accuracy or rmse, since those metrics can get away with assigning the majority class to all, and still get away with high accuracy!


More Info:
https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/


Python Installation: https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_XGBoost_For_Anaconda_on_Windows?lang=en


```python
XGBClassifier().get_params().keys()
```


```python
one_to_left = st.beta(10, 1)  
from_zero_positive = st.expon(0, 50)

params = {  
    "n_estimators": st.randint(100, 500),
    "max_depth": st.randint(2, 8),
    "learning_rate": [0.01],
    #'max_features':'sqrt',
    #"learning_rate": st.uniform(0.001, 0.1),
    "colsample_bytree": one_to_left,
    "subsample": one_to_left,
    "gamma": st.uniform(0, 3),
    #'reg_alpha': from_zero_positive,
    #"min_child_weight": from_zero_positive,
    'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
}

xgbreg = XGBClassifier(objective= 'binary:logistic', eval_metric="auc",
                       nthreads=2)

grid = RandomizedSearchCV(xgbreg, params, n_jobs=1, verbose=1, n_iter=n_iter,
                          random_state=rstate, scoring=scoring)  
grid.fit(X_train,y_train, verbose=False)
save(grid, "Sci_kit XGB")
```


```python
feature_imp(grid.best_estimator_)
```


```python
with sns.axes_style(style='ticks'):
    g = sns.factorplot("Sex", "Age", "Survived", data=train_df, kind="box")
    g.set(ylim=(-1,5))
    g.set_axis_labels("Sex", "Age");
```


```python
num_rounds = 1000
model = XGBClassifier(n_estimators = num_rounds,
                        objective= 'binary:logistic',
                     learning_rate=0.01, random_state=rstate, scoring=scoring)

# use early_stopping_rounds to stop the cv when there is no score imporovement
model.fit(X_train,y_train, early_stopping_rounds=20, eval_set=[(X_test,
y_test)], verbose=False)
score = cross_val_score(model, X_train,y_train, cv=cv)
print(model)
print("\nxgBoost - CV Train : %.2f" % score.mean())
print("xgBoost - Train : %.2f" % metrics.accuracy_score(model.predict(X_train), y_train))
print("xgBoost - Test : %.2f" % metrics.accuracy_score(model.predict(X_test), y_test))
norm_save(model,score, "XGBsklearn")
```

## XGBoost Package Implementation


```python
num_rounds = 100
xgtrain = xgb.DMatrix(X_train, label=y_train)
xgtest = xgb.DMatrix(X_test, label=y_test)

# set xgboost params
param = {'max_depth': 3,  # the maximum depth of each tree
         'objective': 'binary:logistic'}

clf_xgb_cv = xgb.cv(param, xgtrain, num_rounds, 
                    stratified=True, 
                    nfold=n_splits, 
                    early_stopping_rounds=20)
print("Optimal number of trees/estimators is %i" % clf_xgb_cv.shape[0])

watchlist  = [(xgtest,'test'), (xgtrain,'train')]                
clf_xgb = xgb.train(param, xgtrain,clf_xgb_cv.shape[0], watchlist)

# predict function will produce the probability 
# so we'll use 0.5 cutoff to convert probability to class label
y_train_pred = (clf_xgb.predict(xgtrain, ntree_limit=clf_xgb.best_iteration) > 0.5).astype(int)
y_test_pred = (clf_xgb.predict(xgtest, ntree_limit=clf_xgb.best_iteration) > 0.5).astype(int)
score= metrics.accuracy_score(y_test_pred, y_test)
print("\nXGB - Train : %.2f" % score)
print("XGB - Test : %.2f" % metrics.accuracy_score(y_train_pred, y_train))
norm_save(model,score, "XGBstandard")
```

# Parametric Models
Family of models which makes assumption of the underlying distribution, and attempts to build models on top of it with a fixed number of parameters. Works great when they're assumption is correct and the data behaves itself.

# Parametric Discriminative Models
I want to take this opportunity to elaborate a bit more on the discriminative/generative dichotomy. An interesting observation by Andrew Ng is that while discriminative classifiers can reach a higher accuracy cap (asymptotic error), it achieves it at a slower rate than its generative counterpart. He does this by comparing *Naive Bayes* and *Logistic Regression*, two generative models. Therefore, for computational and goal solving reasons, it is best to solve for the conditional probability of p(x|y) directly, instead of trying to compute their joint distribution as seen in generative models.
 
Sources:
https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf

# Logistic Regression
The classification regression. In the binary classification problem, the classic linear regression is unable to bound itself between classes [0,1]. So the logistic regression uses the logit from the sigmoid function, which transformed the weighted input into a probability of class being “1”. Although not the best performer, if offers exploratory information and can potentially reveal causal information, if the experiment is setup correctly.


```python
model= LogisticRegression()
score = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
norm_save(LogisticRegression(),score, "Logistic_Regression")
```


```python
# import statsmodels.api as sm
# logit = sm.Logit(y, X) # fit the model
# result = logit.fit()
# result.summary()
```

# Neural Networks
The only “Deep Model” out of the mix. This is a characteristic of *Representation Learning*, which effectively grants the model its own feature processing and selection steps, catering to large, complex data with intertwining effects. In computer vision tasks, the convolutional neural networks is able to piece apart corners, colors, patterns and more. Recent research in Style Transfer even suggests that stylistic properties of a picture, such as art, can be extracted, and controlled.  (Source: https://arxiv.org/abs/1611.07865)

A common explanation for Neural Networks is that it is a whole bunch of Logistic Regressions. An important thing to remembers is that the hidden-layers are fully connected, meaning that each input variables has a weight to each node (hidden-unit) in the hidden layer, thereby resulting in a black box with a whole lot of parameters, and a whole lot of matrix multiplications. Finally it's, ironic that one of the most intelligible classifiers can be transformed into the least intelligible! In *Computer Age Statistical Inference* authors Bradley Efron and Trevor Hastie are hopeful for the next Ronald Fisher to come and provide statistical intelligibility to modern day machine learning models, many of which are highly developed computationally, but lacking inferential theory.

Model not ideal for such as small dataset.


```python
MLPClassifier().get_params().keys()
```


```python
# Start with a RandomSearchCV to efficiently Narrow the Ballpark
param_grid ={'max_iter': np.logspace(1, 5, 10).astype("int32"),
             'hidden_layer_sizes': np.logspace(2, 3, 4).astype("int32"),
             'activation':['identity', 'logistic', 'tanh', 'relu'],
             'learning_rate': ['adaptive'],
             'early_stopping': [True],
             'alpha': np.logspace(2, 3, 4).astype("int32")
            }

model = MLPClassifier()

grid = RandomizedSearchCV(model,
                    param_grid, cv=cv, scoring=scoring,
                    verbose=1, n_iter=n_iter, random_state=rstate)

grid.fit(X, y)
save(grid, "FFNeural_Net")
```

# Support Vector Classifier
Another model originating in Computer Science, the Support Vector Classifier creates a separation between point to determine class. Maximizing the accuracy of the discriminatory protocol.

### Hyperparameters:
- C: Rigidity and size of the separation line margin. Increasing C leads to a smaller, thus a more complex mode, and everything that comes with it (High variance, Low Bias)!

## Linear Classifier:
In this case, the separator is linear. Imagine a two dimensional plot with a straight line separating two classes of points, but efficiently scalable up to a high dimensional space (plane, cube… hypercube :-O ? )


```python
LinearSVC().get_params().keys()
```


```python
# # Define Model
# model = LinearSVC()
# #Fit Model
# scores= cross_val_score(model, X, y, cv=cv, scoring=scoring)
# norm_save(model, scores, "LinearSV")
```

## Radial Basis Function (RBF)
Here, not only are non-linear boundaries available, the kernel trick is aso introduced, which enables new representations of the data to be formulated, effectively granting the models new dimensions to find better separators in.

### Hyperparameter:
- Gamma: How far the influence of a single training example reaches, and sort of like a soft boundary with a gradient. 
- Low Gamma -> Distant fit, High Gamma = Close Fit, Inverse of the radius of influence of samples selected by the model as support vectors.
http://pages.cs.wisc.edu/~yliang/cs760/howtoSVM.pdf


```python
SVC().get_params().keys()
```


```python
svc = SVC(kernel= 'rbf', probability=True)

model = Pipeline(steps=[('svc', svc)])


param_grid = {'svc__C': st.randint(1,10000),
              'svc__gamma': np.logspace(1, -7, 10)}

grid = RandomizedSearchCV(model, param_grid,
                          cv=cv, verbose=1, scoring=scoring,
                         n_iter=n_iter, random_state=rstate)

grid.fit(X, y)
save(grid, "SVCrbf")
```

## Pipeline: Principle Components Analysis and Support Vector Classifier
Pipelines enable multiple models or processes to be chained up and hyper-parameter tuned together. Very powerful tool, especially when uncertain about a model's reaction to processed data, since it may reveal complex, high performing combinations.

### Dimensionality Reduction: Principal Component Analysis
Decreases the noise by condensing the features into the dimensions of most variance. 


```python
pca = PCA()
svc = SVC(kernel= 'rbf',probability=True)

model = Pipeline(steps=[('pca',pca),
                        ('svc', svc)])


param_grid = {'svc__C': st.randint(1,10000),
              'svc__gamma': np.logspace(1, -7, 10),
             'pca__n_components': st.randint(1,len(X.columns))}

grid = RandomizedSearchCV(model, param_grid,
                          cv=cv, verbose=1,
                         n_iter=n_iter, random_state=rstate, scoring=scoring)

grid.fit(X, y)
save(grid, "PCA_SVC")
```

# Parametric Generative Classification
Models the data generating process in order to assign which categorize the features. Process is represented in terms of joint probabilities. Advantageous since it can be used to generate features, but tends to struggle in terms of accuracy and doesn’t scale well into tall and wide datasets. Only Gaussian Bayes used in this category, but additional Generative Models can be explored, such as:

Hidden Markov model
Probabilistic context-free grammar
Averaged one-dependence estimators
Latent Dirichlet allocation
Restricted Boltzmann machine
Generative adversarial networks

## Gaussian Naive Bayes
Through Bayes rule, is able to use conditional probabilities to classify data. Predicts the class by finding the one with the highest Posterior.


```python
model = GaussianNB()

score = cross_val_score(model, X, y, cv=cv, scoring=scoring)
norm_save(model,score, "Gaussian")
```

## Results


```python
results = results.sort_values(by=["CV Mean"], ascending=False)
results
```

## Classification Voting: Model Ensemble
Like the system that enabled multiple trees to predict together, this system brings together models of all types. A broader implementation.

**Hard Voting:** Plurality voting over the classes.
**Soft Voting:** Selects classed based off aggregated probabilities over the models. Requires model with probabilistic capabilities, which is why I removed linear SCV from inclusion. This difference is tremendous, while hard votes may polarize a small difference between models, say the prediction of 51% Survived and 49% Dead, the soft voting is able to make a more nuanced decision, rewarding high confidence models, and penalizing low confidence.
**Weights:** This is an additional feature which enables manual assigning voting power to each voting model.

## Prepare and Observe Voting Models


```python
### Ensemble Voting
not_proba_list = ('LinearSV')
not_proba =  results[results.Model == not_proba_list]
hard_models = results # All Can Be
prob_models = results[results.Model != not_proba_list] # Not All
```


```python
# None Probabilistic
models = list(zip([ensemble_models[x] for x in not_proba.Model],
                  not_proba.Model))
clfs = []
print('5-fold cross validation:\n')
for clf, label in models:
    scores = cross_val_score(clf, X_train, y_train,cv=5, scoring=scoring, verbose=0)
    print("Train CV Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    md = clf.fit(X, y)    
    clfs.append(md)
    print("Test Accuracy: %0.2f \n" % (metrics.accuracy_score(clf.predict(X_test), y_test)))
    df.to_csv("{}.csv".format(label),header=True,index=False)

del clfs
```


```python
# Only Probabilistic
models = list(zip([ensemble_models[x] for x in prob_models.Model],
                  prob_models.Model))
plt.figure()

print('5-fold cross validation:\n')
clfs = []
#[ensemble_models[x] for x in prob_models.Model]
for clf, label in models:
    scores = cross_val_score(clf, X_train, y_train,cv=5, scoring=scoring, verbose=0)
    print("Train CV Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    md = clf.fit(X_train, y_train)
    # Add to Roc Curve
    fpr, tpr, _ = roc_curve(y_test, md.predict_proba(X_test)[:,1])
    roc_auc = auc(fpr, tpr)

    print('ROC AUC: %0.2f' % roc_auc)
    plt.plot(fpr, tpr, label='{} ROC curve (area = {:.2})'.format(label, roc_auc))
    
    clfs.append(md)
    print("Test Accuracy: %0.2f \n" % (metrics.accuracy_score(clf.predict(X_test), y_test)))
    df.to_csv("{}.csv".format(label),header=True,index=False)

# Plot
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
```

## Soft and Hard Voter of Difference Sizes


```python
# Play with Weights
plt.figure()
for x in [2,3,5,7,10]:
    ECH = EnsembleVoteClassifier([ensemble_models.get(key) for key in hard_models.Model[:x]], voting='hard')
    ECS = EnsembleVoteClassifier([ensemble_models.get(key) for key in prob_models.Model[:x]], voting='soft')
    print('\n')
    print('{}-Voting Models: 5-fold cross validation:\n'.format(x))
    
    for clf, label in zip([ECS, ECH], 
                          ['{}-VM-Ensemble Soft Voting'.format(x),
                           '{}-VM-Ensemble Hard Voting'.format(x)]):
        scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
        print("Train CV Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
        md = clf.fit(X_train, y_train)    
        clfs.append(md)

        if clf is ECS:
             # Add to Roc Curve
            fpr, tpr, _ = roc_curve(y_test, md.predict_proba(X_test)[:,1])
            roc_auc = auc(fpr, tpr)
            print('ROC AUC: %0.2f' % roc_auc)
            plt.plot(fpr, tpr, label='{} ROC curve (area = {:.2})'.format(label, roc_auc))
        
        
        Test_Score = metrics.accuracy_score(clf.predict(X_test), y_test)
        print("Test Accuracy: %0.2f " % Test_Score)
        
        CV_Score = scores.mean()
        STDev = scores.std()
        
        submission = md.predict(test_df)
        df = pd.DataFrame({'PassengerId':test_df.index, 
                               'Survived':submission})
        global results
        results = results.append({'Model': label,'Para': clf, 'CV Mean': CV_Score,
                'Test_Score':Test_Score,'CV STDEV': STDev}, ignore_index=True)
        ensemble_models[label] = model
        df.to_csv("{}.csv".format(label),header=True,index=False)
        
# Plot
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Soft Model ROC Curve')
plt.legend(loc="lower right")
plt.show()
```

## Stacking Models
“Stacking is a way of combining multiple models, that introduces the concept of a meta learner. It is less widely used than bagging and boosting. Unlike bagging and boosting, stacking may be (and normally is) used to combine models of different types.” - Anshul Joshi [1]

Big shoutout to Manohar Swamynathan, author of Mastering Machine Learning with Python in Six Steps, who eloquently explains and took me step by step through stacks in Python. [2]

Read more here:

[1] https://www.quora.com/What-is-stacking-in-machine-learning 

[2] https://github.com/Apress/mastering-ml-w-python-in-six-steps/blob/master/Chapter_4_Code/Code/Stacking.ipynb


```python
from sklearn import cross_validation

X = train_df.drop(["Survived"] , axis=1)
y = train_df["Survived"]

#test_df  = test_df.drop(["PassengerId"] , axis=1).copy()
print(X.shape, y.shape, test_df.shape)

#Normalize
# X = StandardScaler().fit_transform(X)

# ensemble_models.values()

# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rstate)
kfold = cross_validation.StratifiedKFold(y=y_train, n_folds=5, random_state=rstate)
num_trees = 10
verbose = True # to print the progress

clfs = [KNeighborsClassifier(),
        RandomForestClassifier(n_estimators=num_trees, random_state=rstate),
        GradientBoostingClassifier(n_estimators=num_trees, random_state=rstate)]

# Creating train and test sets for blending
dataset_blend_train = np.zeros((X_train.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_test.shape[0], len(clfs)))
dataset_blend_test_df = np.zeros((test_df.shape[0], len(clfs)))

print('5-fold cross validation:\n')
for i, clf in enumerate(clfs):   
    scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=kfold, scoring='accuracy')
    print("##### Base Model %0.0f #####" % i)
    print("Train CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    clf.fit(X_train, y_train)   
    print("Train Accuracy: %0.2f " % (metrics.accuracy_score(clf.predict(X_train), y_train)))
    dataset_blend_train[:,i] = clf.predict_proba(X_train)[:, 1]
    dataset_blend_test[:,i] = clf.predict_proba(X_test)[:, 1]
    dataset_blend_test_df[:,i] = clf.predict_proba(test_df)[:, 1]
    print("Test Accuracy: %0.2f " % (metrics.accuracy_score(clf.predict(X_test), y_test)))    

print("##### Meta Model #####")
clf = LogisticRegression()
scores = cross_validation.cross_val_score(clf, dataset_blend_train, y_train, cv=kfold, scoring=scoring)
clf.fit(dataset_blend_train, y_train)
print("Train CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
print("Train Accuracy: %0.2f " % (metrics.accuracy_score(clf.predict(dataset_blend_train), y_train)))
print("Test Accuracy: %0.2f " % (metrics.accuracy_score(clf.predict(dataset_blend_test), y_test)))
```


```python
eval_plot(clf)
```


```python
score = cross_val_score(clf, X, y, cv=cv, scoring=scoring)
norm_save(clf, score, "stacked")
```

## Best Model: Soft Voting Ensemble with Top Seven Models

After shoveling most of these models into Kaggle, I found that the **Soft Voting Ensemble with Top Seven Models** to be the best performing with a public score of **81.339**. I am curious to see whether model deviation, or CV Train and Test score difference are able to indicate submission set performance, since it may shed light on model consistency.


```python
results.sort_values(by=["CV Mean"], ascending=False)
```

## Classification Evaluation for Best Model


```python
# http://scikit-plot.readthedocs.io/en/stable/Quickstart.html
```


```python
# Reinstate Data, since it was meddled with during Stacked Models
X = train_df.drop(["Survived"] , axis=1)
y = train_df["Survived"]

# Stratified Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_train.shape, y_train.shape, X_test.shape, y_test.shape

evalmodel = EnsembleVoteClassifier([ensemble_models.get(key) for key in prob_models.Model[:7]], voting='soft')
evalmodel.fit(X_train, y_train)
y_pred = evalmodel.predict(X_test)
# Report
print("\n Report:")
print(classification_report(y_test, y_pred))
confusion_matrix(y_test, y_pred)
# Matrix
print("\n Matrix:")
skplt.metrics.plot_confusion_matrix(y_pred, y_test, normalize=True)
plt.show()
```


```python
# # Predict Submission Set and Output to CSV
# clf = EnsembleVoteClassifier([ensemble_models.get(key) for key in prob_models.Model[:7]], voting='soft')
# md = clf.fit(X, y)
# df = pd.DataFrame({'PassengerId':test_df.index, 'Survived':md.predict(test_df)})
# df.to_csv("Soft_Voting_7_TopModel", index=False)
```


```python
end = time.time()
print("Notebook took %0.2f minutes to Run"%((end - start)/60))
```

## Reflection
A recurrent theme that I have observed is that the accuracy on the testing set is always higher than that of the submission set. This suggests that there is a disconnected representation of the underlying data distributions. A likely contributor to this problem is the quality of pre-processing, whose features appeared to have redundant effects, such as Title and Sex. A quick fix could be dimensionality reduction, but ultimately if proper exploration of features is conducted, then the “garbage in” problem can be minimized.

Note: Deep Learning references are to be taken **LIGHTLY!**, currently reading Ian Goodfellow’s book, so this stuff is on my mind.

### Thank You for making it this far! Next, I am thinking of either pursuing a wide application Regression models on another dataset, or doubling down on the exploratory analysis and feature engineering for Titanic! 

### *What say you?* P.S, Looking for constructive feedback :)

## UpNext
- Better Confussion Matrix
- Missclassification by ID, and filter for "most diverse model". May create a matrix to visualize this.
- Reproducibility


```python

```
