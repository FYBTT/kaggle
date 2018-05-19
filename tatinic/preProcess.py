#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn import preprocessing
train_data = pd.read_csv("./data/train.csv")
train_data['Embarked'].fillna('S', inplace=True)
train_data['Cabin'] = train_data['Cabin'].fillna('0').apply(lambda x: x[0][0])
train_data['Pclass'] = train_data['Pclass'].apply(lambda x: str(x))
del train_data['Name'], train_data['Ticket']
test_data = pd.read_csv("./data/test.csv")
test_data['Fare'].fillna(test_data[(test_data['Pclass']==3) & (test_data['Embarked']=='S') & (test_data['Sex']=='male')]['Fare'].mean(), inplace=True)
test_data['Cabin'] = test_data['Cabin'].fillna('0').apply(lambda x: x[0][0])
test_data['Pclass'] = test_data['Pclass'].apply(lambda x: str(x))
del test_data['Name'], test_data['Ticket']


def age_map(x):
    if x<10:
        return '10-'
    if x<60:
        return '%d-%d'%(x//5*5, x//5*5+5)
    elif x>=60:
        return '60+'
    else:
        return 'Null'
train_data['Age_map'] = train_data['Age'].apply(lambda x: age_map(x))
test_data['Age_map'] = test_data['Age'].apply(lambda x: age_map(x))
train_data.Fare = preprocessing.scale(train_data.Fare)
test_data.Fare = preprocessing.scale(test_data.Fare)
train_x = pd.concat([train_data[['SibSp','Parch','Fare']], pd.get_dummies(train_data[['Pclass','Sex','Cabin','Embarked','Age_map']])],axis=1)
train_y = train_data['Survived']
test_x = pd.concat([test_data[['SibSp','Parch','Fare']], pd.get_dummies(test_data[['Pclass', 'Sex','Cabin','Embarked', 'Age_map']])],axis=1)
trainSet = set(train_x.columns)
testSet = set(test_x.columns)
if len(trainSet) > len(testSet):
    for col in list(trainSet - testSet):
        test_x[col] = 0
trainSort = train_x.columns.tolist()
test_x = test_x[trainSort]

from sklearn import linear_model
from sklearn import model_selection
import matplotlib.pyplot as plt  
from sklearn.learning_curve import learning_curve  
from sklearn.learning_curve import validation_curve  

base_line_model = linear_model.LogisticRegression(tol = 1e-6, max_iter = 500)
param = {'penalty':['l1','l2'], 
        'C':[0.1, 0.5, 1.0,5.0]}
grd = model_selection.GridSearchCV(estimator=base_line_model, param_grid=param, cv=5, n_jobs=4)
grd.fit(train_x, train_y)

def plot_learning_curve(clf, title, x, y, ylim=None, cv=None, n_jobs=3, train_sizes=np.linspace(.05, 1., 5)):
    train_sizes, train_scores, test_scores = learning_curve(
        clf, x, y, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    ax = plt.figure().add_subplot(111)
    ax.set_title(title)
    if ylim is not None:
        ax.ylim(*ylim)
    ax.set_xlabel(u"train_num_of_samples")
    ax.set_ylabel(u"score")

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                     alpha=0.1, color="b")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                     alpha=0.1, color="r")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"train score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"testCV score")

    ax.legend(loc="best")

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff

plot_learning_curve(grd, u"learning_rate", train_x, train_y)
gender_submission = pd.DataFrame({'PassengerId':test_data.iloc[:,0],'Survived':grd.predict(test_x)})
gender_submission.to_csv('./data/gender_submission.csv', index=None)