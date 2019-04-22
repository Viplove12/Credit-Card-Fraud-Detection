# -*- coding: utf-8 -*-
import pandas as pd

dataset=pd.read_csv('creditcard.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#splitting the dataset
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=0)  

#implementing logistic regression
from sklearn.linear_model import LogisticRegression
classifierLOG_woSMOTE=LogisticRegression(random_state=0)
classifierLOG_woSMOTE.fit(xtrain,ytrain)

#implementing naive bayes
from sklearn.naive_bayes import GaussianNB
classifierNB_woSMOTE=GaussianNB()
classifierNB_woSMOTE.fit(xtrain,ytrain)

#implementing random forest
from sklearn.ensemble import RandomForestClassifier
classifierRF_woSMOTE=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifierRF_woSMOTE.fit(xtrain,ytrain)

#predicting test results
ypredRF_woSMOTE=classifierRF_woSMOTE.predict(xtest)
ypredNB_woSMOTE=classifierNB_woSMOTE.predict(xtest)
ypredLOG_woSMOTE=classifierLOG_woSMOTE.predict(xtest)

#confusion matrix
from sklearn.metrics import confusion_matrix
cmRF_woSMOTE=confusion_matrix(ytest,ypredRF_woSMOTE)
cmNB_woSMOTE=confusion_matrix(ytest,ypredNB_woSMOTE)
cmLOG_woSMOTE=confusion_matrix(ytest,ypredLOG_woSMOTE)

#implementing f1 score
from sklearn.metrics import f1_score
scoreRF_woSMOTE = f1_score(ypredRF_woSMOTE, ytest)
scoreNB_woSMOTE = f1_score(ypredNB_woSMOTE, ytest)
scoreLOG_woSMOTE = f1_score(ypredLOG_woSMOTE, ytest)

#implementing SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
xtrain_SMOTE, ytrain_SMOTE = sm.fit_sample(xtrain, ytrain.ravel())

#implementing random forest
classifierRF_SMOTE=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifierRF_SMOTE.fit(xtrain_SMOTE,ytrain_SMOTE)

#implementing naive bayes
classifierNB_SMOTE=GaussianNB()
classifierNB_SMOTE.fit(xtrain_SMOTE,ytrain_SMOTE)

#implementing logistic regression
classifierLOG_SMOTE=LogisticRegression(random_state=0)
classifierLOG_SMOTE.fit(xtrain_SMOTE,ytrain_SMOTE)

#predicting test results
ypredRF_SMOTE=classifierRF_SMOTE.predict(xtest)
ypredNB_SMOTE=classifierNB_SMOTE.predict(xtest)
ypredLOG_SMOTE=classifierLOG_SMOTE.predict(xtest)

#confusion matrix
cmRF_SMOTE=confusion_matrix(ytest,ypredRF_SMOTE)
cmNB_SMOTE=confusion_matrix(ytest,ypredNB_SMOTE)
cmLOG_SMOTE=confusion_matrix(ytest,ypredLOG_SMOTE)

#implementing f1 score
scoreRF_SMOTE = f1_score(ypredRF_SMOTE, ytest)
scoreNB_SMOTE = f1_score(ypredNB_SMOTE, ytest)
scoreLOG_SMOTE = f1_score(ypredLOG_SMOTE, ytest)

#implementing undersampling
from imblearn.under_sampling import RandomUnderSampler
us = RandomUnderSampler(random_state=0)
xtrain_US, ytrain_US = us.fit_sample(xtrain, ytrain.ravel())

#implementing naive bayes
classifierNB_US=GaussianNB()
classifierNB_US.fit(xtrain_US,ytrain_US)

#implementing logistic regression
classifierLOG_US=LogisticRegression(random_state=0)
classifierLOG_US.fit(xtrain_US,ytrain_US)

#implementing random forest
classifierRF_US=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifierRF_US.fit(xtrain_US,ytrain_US)

#predicting test results
ypredRF_US=classifierRF_US.predict(xtest)
ypredNB_US=classifierNB_US.predict(xtest)
ypredLOG_US=classifierLOG_US.predict(xtest)

#confusion matrix
cmRF_US=confusion_matrix(ytest,ypredRF_US)
cmLOG_US=confusion_matrix(ytest,ypredLOG_US)
cmNB_US=confusion_matrix(ytest,ypredNB_US)


#implementing f1 score
scoreRF_US = f1_score(ypredRF_US, ytest)
scoreLOG_US = f1_score(ypredLOG_US, ytest)
scoreNB_US = f1_score(ypredNB_US, ytest)