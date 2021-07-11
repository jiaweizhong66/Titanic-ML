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

# What is the most frequently occuring age for male and female? 
is_f = df['Sex'] == 'female'
df_f = df[is_f]
df_f['Age'].value_counts()

is_m = df['Sex'] == 'male'
df_m = df[is_m]
df_m['Age'].value_counts()

# Convert the 'Sex' feature of the Train dataset into categorical feature.
sex_category = []
for sex in df['Sex']:
  if sex == "female":
    sex_category.append(1)
  else:
    sex_category.append(0)
df['Sex_Category'] = sex_category


# Normalize 'Age' and 'Fare' to scale{0,1}
x = df['Age'].values.reshape(-1,1) #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df['Age_Normalized'] = x_scaled

fare = df['Fare'].values.reshape(-1,1) #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
fare_scaled = min_max_scaler.fit_transform(fare)
df['Fare_Normalized'] = fare_scaled

#check null values
df.isnull().sum()

# dropping null value
df.dropna(inplace=True)

# split the data into trainning and test parts
x_train,x_test,y_train,y_test = train_test_split(df[['Age_Normalized','Sex_Category','Fare_Normalized']],df['Survived'],
test_size=0.2)

print(x_train)
print(y_train)

# initialize ML model
log_reg = LogisticRegression()
log_reg.fit(x_train,y_train)

# Making prediction
survive_predictions = log_reg.predict(x_test)]

# reports
print(classification_report(y_test,survive_predictions))
