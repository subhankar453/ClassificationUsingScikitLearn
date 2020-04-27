import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

titanic_df = pd.read_csv('datasets/titanic/train-processed.csv')

from sklearn.model_selection import train_test_split
X = titanic_df.drop('Survived', axis = 1)
Y = titanic_df['Survived']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)


#penalty is for regularization of the model to apply penalty for complex models
#C idicates inverse of regularization strength. A smaller value indicates stronger regularization
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(penalty = 'l2', C = 1.0, solver = 'liblinear').fit(x_train, y_train)
y_pred = logistic_model.predict(x_test)

df_pred_actual = pd.DataFrame({'predicted' : y_pred, 'actual' : y_test})
titanic_crosstab = pd.crosstab(y_pred, y_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

print("Accuracy : ", acc)
print("Precision : ", prec)
print("Recall : ", rec)

TP = titanic_crosstab[1][1]
TN = titanic_crosstab[0][0]
FP = titanic_crosstab[0][1]
FN = titanic_crosstab[1][0]

accuracy_verified = (TP + TN) / (TP + TN + FP + FN)
precision_verified = TP / (TP + FP)
recall_verified = TP / (TP + FN) 

accuracy_verified
precision_verified
recall_verified
