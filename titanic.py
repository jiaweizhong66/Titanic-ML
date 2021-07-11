import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, LogisticRegression 
from sklearn.metrics import classification_report, mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt
import seaborn as sb
from collection import counter

df = pd.read_csv("./train.csv")#reading in data from the csv file

# show dataframe
df.head()

# Checking the ratio 
sb.countplot(data=df_train,x="Survived",hue="Survived")
plt.title("Ratio of the classes")

#Checking the "Sex" feature and the corresponding Survival rate
sb.countplot(x="Sex",data=df_train,hue="Survived")

#Checking the "Pclass" feature and the corresponding Survival rate
sb.countplot(x="Pclass",data=df_train,hue="Survived")

#What is Pclass?
df_train.groupby(by = "Pclass")['Fare'].mean()

# Checking the "Age" feature and the corresponding Survival rate
plt.figure(figsize=(40,15))
sb.countplot(x="Age",data=df_train,hue="Survived")

# heatmap
plt.figure(figsize=(20,10))
sb.heatmap(df_train.corr())

# create new features
df_train[['male','female']] = pd.get_dummies(df_train['Sex'])
df_train[['C','Q','S']] = pd.get_dummies(df_train['Embarked'])
df_train.head()

# fill the null value
df_train.fillna(method="backfill",inplace=True)

#Drop the irrelevant features
drop_features = ['PassengerId', 'Name', 'Titles', 'Ticket','Name','Cabin','Embarked','Sex']
df_train.drop(drop_features,inplace=True,axis=1)

df_train.head()

# split the trainning and testing data
x_train, x_test, y_train, y_test = train_test_split(df_train.loc[:, df_train.columns != 'Survived'],df_train.Survived,\
test_size=0.1)

# initialize the linear regression algorithm
titanic_model_v1 = LogisticRegression()
titanic_model_v1.fit(x_train,y_train)

# making predictions
y_predictions = titanic_model_v1.predict(x_test)

# print reports
print(classification_report(y_test_2,y_predictions))