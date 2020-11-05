# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#1-Data File and attach DataSets
train_df=pd.read_csv(r'D:\\Git_Projects\\Kaggle-ML-Projects\\Titanic Calssification\\train.csv')
test_df=pd.read_csv(r'D:\\Git_Projects\\Kaggle-ML-Projects\\Titanic Calssification\\test.csv')
combine = [train_df, test_df]
#print(train_df.columns.values)
#print(train_df.head(10))

#2-Data Cleaning
#Show all column and if have null value or not .
print(train_df.info)
#print some description
#print(train_df.describe())
#How to print all column and rows in pycharm
#print(train_df.head(300).to_string())


#mean between Features and Survived
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print("*"*50)
print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print("*"*50)
print(train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))

#This simple analysis confirms our assumptions as decisions for subsequent workflow stages. in age feature
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
plt.show()
