import pandas as pd
import numpy as np
from sklearn import preprocessing

df = pd.read_csv("./train.csv")#reading in data from the csv file
df.head()

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