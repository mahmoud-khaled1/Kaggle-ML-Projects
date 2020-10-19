import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

#1-Data File and attach data
Train=pd.read_csv('D:\\train.csv')
Test=pd.read_csv('D:\\test.csv')

"""
print("Price Details:")
print(Train['SalePrice'].describe())

#histogram

sns.distplot(Train['SalePrice'])
plt.show()
"""
#scatter plot grlivarea(Area Of House)/saleprice
data = pd.concat([Train['SalePrice'], Train['GrLivArea']], axis=1) #Concat 2 column together to plot them
data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000));


#scatter plot TotalBsmtSF(BadRoom)/saleprice
data = pd.concat([Train['SalePrice'], Train['TotalBsmtSF']], axis=1) #Concat 2 column together to plot them
data.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0,800000));

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


#2-Data Cleaning

#missing data : dealing with null  in column because if we train algorithm with them,
#then miss leading will occured.
total = Train.isnull().sum().sort_values(ascending=False)
percent = (Train.isnull().sum()/Train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))

#dealing with missing data
Train = Train.drop((missing_data[missing_data['Total'] > 1]).index,1)
Train = Train.drop(Train.loc[Train['Electrical'].isnull()].index) #remove this row in electrical where have null value .
print('*'*50)
print("No of Missing Data :",Train.isnull().sum().max()) #just checking that there's no missing data missing...

#3-Data Scaling
#saleprice_scaled = StandardScaler().fit_transform(Train['SalePrice'][:,np.newaxis]);

#print(saleprice_scaled[:10])
#bivariate analysis between saleprice/grlivarea after Data Scaling
var = 'GrLivArea'
data = pd.concat([Train['SalePrice'], Train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#The two values with bigger 'GrLivArea' seem strange and they are not following the crowd.
# We can speculate why this is happening. Maybe they refer to agricultural area and that could explain the low price.
# I'm not sure about this but I'm quite confident that these two points are not representative of the typical case.
# Therefore, we'll define them as outliers and delete them.

#deleting points
#Train.sort_values(by = 'GrLivArea', ascending = False)[:2]
Train = Train.drop(Train[Train['Id'] == 1299].index)
Train = Train.drop(Train[Train['Id'] == 524].index)

#bivariate analysis between saleprice/grlivarea after Delete these points
var = 'GrLivArea'
data = pd.concat([Train['SalePrice'], Train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

Train2 = Train.drop("Id", axis=1)
Y=Train2.iloc[:,-1:]
Train2 = Train2.drop("SalePrice", axis=1)
#convert categorical(string input in column ) variable into dummy
#make new columns  and put integer value as classification target
Train2 = pd.get_dummies(Train2)

"""
from sklearn.preprocessing import StandardScaler
#Copy= True Just told algorithm don't change the data
Data_Scaling =StandardScaler(copy=True,with_mean=True,with_std=True)#.fit_transform(data)
Scaling_data=Data_Scaling.fit_transform(Train2)
"""
X=Train2.iloc[:,:]

#print(X.shape)
#print(Y.shape)

#4-Data Splitting
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.23,shuffle=True,random_state=44)

#Apply Linear Regression algorithm :
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error

# fit_intercept : هل نريد حساب نقطه التقاطع مع محور اكس ام لا
# normalize : make normalization to data or not
#copy_x : if you want to just copy only the data without any change of them
#n_jobs : is  specify speed of operation in your processor if =0 or none will be normal ,-1 is faster speed of processor
# and the more than 1 the more than speed
Linear_Reg_model=LinearRegression(fit_intercept=True,normalize=True ,copy_X=True,n_jobs=-1)
Linear_Reg_model.fit(X_train,y_train)

#Show Details
#Note the best Score is 1 so we should make it near to 1
print("Linear Regression Train Score :",Linear_Reg_model.score(X_train,y_train))
print("Linear Regression test Score :",Linear_Reg_model.score(X_test,y_test))

#Calculating Predication
Y_pred=Linear_Reg_model.predict(X_test)
print(Y_pred[:10])
print(y_test[:10])



