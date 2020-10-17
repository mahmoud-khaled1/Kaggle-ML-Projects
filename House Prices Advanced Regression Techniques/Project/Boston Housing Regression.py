import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

#Attach Data Sets
Train=pd.read_csv(r'D:\Git_Projects\Kaggle(ML)Projects\House Prices Advanced Regression Techniques\train.csv')
Test=pd.read_csv(r'D:\Git_Projects\Kaggle(ML)Projects\House Prices Advanced Regression Techniques\test.csv')

"""
print("Price Details:")
print(Train['SalePrice'].describe())
"""
#histogram
"""
sns.distplot(Train['SalePrice'])
plt.show()
"""

#scatter plot grlivarea(Area Of House)/saleprice
data = pd.concat([Train['SalePrice'], Train['GrLivArea']], axis=1) #Concat 2 column together to plot them
data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000));


#scatter plot TotalBsmtSF(BadRoom)/saleprice
data = pd.concat([Train['SalePrice'], Train['TotalBsmtSF']], axis=1) #Concat 2 column together to plot them
data.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0,800000));



#box plot overallqual/saleprice #Min and max price of overallQuality of house
data = pd.concat([Train['SalePrice'], Train['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);

#box plot YearBuilt/saleprice #Min and max price of YearBuilt of house
data = pd.concat([Train['SalePrice'], Train['YearBuilt']], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x='YearBuilt', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)

#correlation matrix ---> Relationship between every column and each
corrmat = Train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)


#saleprice correlation matrix of  best specific column(Feature) with  precient number
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index # Get best 10 relationship Feature with Price .
cm = np.corrcoef(Train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)


#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(Train[cols], size = 2.5)


#missing data : dealing with null  in column because if we train algorithm with them,
#then miss leading will occured.

#We'll consider that when more than 15% of the data is missing, we should delete the corresponding variable
# and pretend it never existed. This means that we will not try any trick to fill the missing data in these cases.
# According to this, there is a set of variables (e.g. 'PoolQC', 'MiscFeature', 'Alley', etc.)
# that we should delete. The point is: will we miss this data? I don't think so.
# None of these variables seem to be very important, since most of them are not aspects in which
# we think about when buying a house (maybe that's the reason why data is missing?).
# Moreover, looking closer at the variables, we could say that variables like 'PoolQC',
# 'MiscFeature' and 'FireplaceQu' are strong candidates for outliers, so we'll be happy to delete them.
#In summary, to handle missing data, we'll delete all the variables with missing data, except the variable 'Electrical'.
# In 'Electrical' we'll just delete the observation with missing data.
#Print these missing  data column
total = Train.isnull().sum().sort_values(ascending=False)
percent = (Train.isnull().sum()/Train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))

#dealing with missing data
Train = Train.drop((missing_data[missing_data['Total'] > 1]).index,1)
Train = Train.drop(Train.loc[Train['Electrical'].isnull()].index)
print('*'*50)
print("No of Missing Data :",Train.isnull().sum().max()) #just checking that there's no missing data missing...

#Data Scaling
#standardizing data
saleprice_scaled = StandardScaler().fit_transform(Train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10] #rearrange Ascending
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:] #rearrange Descending
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

#bivariate analysis between saleprice/grlivarea after Data Scaling
var = 'GrLivArea'
data = pd.concat([Train['SalePrice'], Train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#The two values with bigger 'GrLivArea' seem strange and they are not following the crowd.
# We can speculate why this is happening. Maybe they refer to agricultural area and that could explain the low price.
# I'm not sure about this but I'm quite confident that these two points are not representative of the typical case.
# Therefore, we'll define them as outliers and delete them.

#deleting points
Train.sort_values(by = 'GrLivArea', ascending = False)[:2]
Train = Train.drop(Train[Train['Id'] == 1299].index)
Train = Train.drop(Train[Train['Id'] == 524].index)

#bivariate analysis between saleprice/grlivarea after Delete those strange point (anonymous point)
data = pd.concat([Train['SalePrice'], Train['TotalBsmtSF']], axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0,800000));


#bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
data = pd.concat([Train['SalePrice'], Train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#**We can feel tempted to eliminate some observations (e.g. TotalBsmtSF > 3000)
# but I suppose it's not worth it. We can live with that, so we'll not do anything.

#convert categorical(string input in column ) variable into dummy
#make new columns  and put integer value as classification target
Train = pd.get_dummies(Train)



#Applying algorithms of Regression ...........
