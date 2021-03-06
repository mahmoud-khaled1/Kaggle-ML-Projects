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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#1-Data File and attach DataSets
train_df=pd.read_csv(r'D:\\Git_Projects\\Kaggle-ML-Projects\\Titanic Calssification\\train.csv')
test_df=pd.read_csv(r'D:\\Git_Projects\\Kaggle-ML-Projects\\Titanic Calssification\\test.csv')
combine = [train_df, test_df]
#print(train_df.columns.values)
#print(train_df.head(10))

#2-Data Cleaning and learn from Correlating between Features.

#1-Correlating between Features
#Show all column and if have null value or not .
print(train_df.info)
#print some description in numerical  column
#print(train_df.describe())
#print some description in categorical  column
#print(train_df.describe(include=['O']))
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

#to show relationship between Pclass and Survived
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()

#correlation matrix ---> Relationship between every column and each
corrmat = train_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)


#2-Data Cleaning

#dropping features that have nearly all null and feature that doesn't affect on result of training .
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df] # train and test together
print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

#Creating new feature extracting from existing
#We want to analyze if Name feature can be engineered to extract titles
# and test correlation between titles and survival, before dropping Name and PassengerId features.
#In the following code we extract Title feature using regular expressions.
# The RegEx pattern (\w+\.) matches the first word which ends with a dot character
# within Name feature. The expand=False flag returns a DataFrame.

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False) #create new column and add it to dataset in train and test

#To show title
#print(pd.crosstab(train_df['Title'], train_df['Sex']))

#We can replace many titles with a more common name or classify them as Rare.
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

#to show mean of every title
#print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

#We can convert the categorical titles to ordinal.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()

#Now we can safely drop the Name feature from training and testing datasets.
# We also do not need the PassengerId feature in the training dataset.
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
print(train_df.shape, test_df.shape)

#Converting a categorical feature sex
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

#print(train_df.head())

#Let us start by preparing an empty array to contain guessed Age values based on Pclass x Gender combinations.
#Now we iterate over Sex (0 or 1) and Pclass (1, 2, 3) to calculate guessed values of Age for the six combinations.
guess_ages = np.zeros((2,3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                               (dataset['Pclass'] == j + 1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                        'Age'] = guess_ages[i, j]

    dataset['Age'] = dataset['Age'].astype(int)
#Let us create Age bands and determine correlations with Survived.
#train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
#train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

# replace Age with  based on range of people be(0,1,2,3) only
for dataset in combine:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']





#We can create a new feature for FamilySize which combines Parch and SibSp.
# This will enable us to drop Parch and SibSp from our datasets.
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

#We can create another feature called IsAlone.

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

#Let us drop Parch, SibSp,  features in favor of IsAlone.

train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp'], axis=1)
combine = [train_df, test_df]

#TO FILL Embarked COLUMN WITH VALUE IN CASE OF NULL
#LET'S GET MOST FREQUANCY
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',
                                                                                            ascending=False)
#Converting categorical feature to numeric¶
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#Fill null fare with mean values
#We can now complete the Fare feature for single missing value in test dataset using mode
# to get the value that occurs most frequently for this feature. We do this in a single line of code.

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
#Convert the Fare feature to ordinal values based on the FareBand.

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

#3-Feature selection
#I Need Every Feature

#4-Data Scaling
#No need To scale data

#5-Data Splitting
X = train_df.drop("Survived", axis=1)
Y = train_df["Survived"]
x_test  = test_df.drop("PassengerId", axis=1).copy()
#print(X_train.shape, Y_train.shape, X_test.shape)

#6-Choice best algorithm :
"""
Logistic Regression
KNN or k-Nearest Neighbors
Support Vector Machines
Naive Bayes classifier
Decision Tree
Random Forrest
Perceptron
Artificial neural network
RVM or Relevance Vector Machine
"""

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,shuffle=True,random_state=44)


def Logistaccc_Calssification():
    Logistic_Regression_model = LogisticRegression(penalty='l2', solver='liblinear', C=0.1, random_state=44, n_jobs=-1,
                                                   max_iter=10000, l1_ratio=None, tol=1e-4, dual=False)
    Logistic_Regression_model.fit(X_train, y_train)
    # Calculation Details
    print("Train Score :", Logistic_Regression_model.score(X_train, y_train))
    print("test Score :", Logistic_Regression_model.score(X_test, y_test))
    print("Classes are :", Logistic_Regression_model.classes_)
    # We have max_iter=10000 but here will found max_itr =4059 iteration that mean that algorithm after 4059 iteration not necessary
    print("Max Iteration :", Logistic_Regression_model.n_iter_)
    print("*" * 100)

    # Make Predicted
    y_pre = Logistic_Regression_model.predict(X_test)

    print(list(y_test[:15]))
    print(list(y_pre[:15]))

    # Calculted Coufusion matrix
    # [TP     FP
    # [FN     TN]
    ##We should make TP and TN biggest and smallest FP and FN
    CM = confusion_matrix(y_test, y_pre)
    print("Confusion Matrix :")
    print(CM)

    # show all correlation between output and all features
    coeff_df = pd.DataFrame(train_df.columns.delete(0))
    coeff_df.columns = ['Feature']
    coeff_df["Correlation"] = pd.Series(Logistic_Regression_model.coef_[0])
    coeff_df.sort_values(by='Correlation', ascending=False)
    return Logistic_Regression_model.score(X_train, y_train)


def SVM():
    SVC_model = SVC(C=1,kernel='linear',gamma='auto',max_iter=10000,random_state=44)
    SVC_model.fit(X_train, y_train)
    # Print Details
    print("Train Score :", SVC_model.score(X_train, y_train))
    print("Test Score :", SVC_model.score(X_test, y_test))
    print("No of Iteration :", SVC_model.max_iter)

    # make Predicted
    Y_pre = SVC_model.predict(X_test)

    print(list(Y_pre[:10]))
    print(list(y_test[:10]))
    from sklearn.metrics import confusion_matrix
    CM = confusion_matrix(y_true=y_test, y_pred=Y_pre)
    print(CM)
    return SVC_model.score(X_train, y_train)

def KNN():
    KNeighborsClassifierModel = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto')
    KNeighborsClassifierModel.fit(X_train, y_train)
    Y_pre = KNeighborsClassifierModel.predict(X_test)
    # Print Details
    print("Train Score :", KNeighborsClassifierModel.score(X_train, y_train))
    print("Test Score :", KNeighborsClassifierModel.score(X_test, y_test))

    print(list(Y_pre[:10]))
    print(list(y_test[:10]))
    from sklearn.metrics import confusion_matrix
    CM = confusion_matrix(y_true=y_test, y_pred=Y_pre)
    print(CM)
    return KNeighborsClassifierModel.score(X_train, y_train)

def Gaussian ():
    # priors:   القيم المفترضة السابقة
    GaussianNBModel = GaussianNB()
    GaussianNBModel.fit(X_train, y_train)

    # Print Score of algorithm
    print("Train Score :", GaussianNBModel.score(X_train, y_train))
    print("test Score :", GaussianNBModel.score(X_test, y_test))

    # Predected
    Y_pre = GaussianNBModel.predict(X_test)
    print(list(Y_pre[:20]))
    print(list(y_test[:20]))

    # confusion_matrix
    CM = confusion_matrix(y_true=y_test, y_pred=Y_pre)
    print(CM)
    return GaussianNBModel.score(X_train, y_train)

def Decision_Tree():
    # Applying algorithm
    Decision_Tree_model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=3, min_samples_split=2,
                                                 min_samples_leaf=2, max_features='auto')
    Decision_Tree_model.fit(X_train, y_train)

    # Show Details
    print("Train Score :", Decision_Tree_model.score(X_train, y_train))
    print("Test Score :", Decision_Tree_model.score(X_test, y_test))
    print("Calsses of model ", Decision_Tree_model.classes_)
    print("max Feature of model ", Decision_Tree_model.feature_importances_)  # important of every Feature
    # Calculation Prediction
    Y_pre = Decision_Tree_model.predict(X_test)
    print(list(Y_pre[:10]))
    print(list(y_test[:10]))

    from sklearn.metrics import confusion_matrix
    CM = confusion_matrix(y_true=y_test, y_pred=Y_pre)
    print(CM)
    return Decision_Tree_model.score(X_train, y_train)

def Random_Forest():
    RandomForestClassifierModel = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=3,
                                                         random_state=44)
    RandomForestClassifierModel.fit(X_train, y_train)

    print("Train Score :", RandomForestClassifierModel.score(X_train, y_train))
    print("test Score :", RandomForestClassifierModel.score(X_test, y_test))
    #print("no of features  :", RandomForestClassifierModel.n_features_)
    #print(" importace  features  :", RandomForestClassifierModel.feature_importances_)

    # Predected
    Y_pre = RandomForestClassifierModel.predict(X_test)
    print(list(y_test[:10]))
    print(list(Y_pre[:10]))

    from sklearn.metrics import confusion_matrix
    CM = confusion_matrix(y_true=y_test, y_pred=Y_pre)
    print(CM)
    return RandomForestClassifierModel.score(X_train, y_train)

LC_Score=Logistaccc_Calssification()
SVM_Score=SVM()
KNN_Score=KNN()
GAUSSIAN_Score=Gaussian()
DT_Score=Decision_Tree()
RF_Score=Random_Forest()

#Print all Score
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'GAUSSIAN ',
              'Decision Tree'],
    'Score': [SVM_Score, KNN_Score, LC_Score,
              RF_Score, GAUSSIAN_Score, DT_Score]})
models.sort_values(by='Score', ascending=False)
print(models)