import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

titanic_df = pd.read_csv('datasets/titanic/train.csv')
titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)
titanic_df[titanic_df.isnull().any(axis = 1)].count()
titanic_df = titanic_df.dropna()
titanic_df.describe()

fig, ax = plt.subplots(figsize = (12,8))
plt.scatter(titanic_df['Age'], titanic_df['Survived'])
plt.xlabel('Age')
plt.ylabel('Survived')

fig, ax = plt.subplots(figsize = (12,8))
plt.scatter(titanic_df['Fare'], titanic_df['Survived'])
plt.xlabel('Fare')
plt.ylabel('Survived')

pd.crosstab(titanic_df['Sex'], titanic_df['Survived'])
pd.crosstab(titanic_df['Pclass'], titanic_df['Survived'])

titanic_df_corr = titanic_df.corr()
titanic_df_corr

fig, ax = plt.subplots(figsize = (12,8))
sns.heatmap(titanic_df_corr, annot = True)

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
titanic_df['Sex'] = label_encoder.fit_transform(titanic_df['Sex'].astype(str))
label_encoder.classes_

titanic_df = pd.get_dummies(titanic_df, columns = ['Embarked'])

titanic_df = titanic_df.sample(frac = 1).reset_index(drop = True)

titanic_df.to_csv('datasets/titanic/train-processed.csv', index = False)

