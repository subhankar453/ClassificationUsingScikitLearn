import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

fashion_df = pd.read_csv('datasets/fashion-mnist_train.csv')
fashion_df.shape

fashion_mnist_df = fashion_df.sample(frac = 0.3).reset_index(drop = True)
fashion_mnist_df.shape

lookup = {0 : 'T-shirt',
          1 : 'Trouser',
          2 : 'Pullover',
          3 : 'Dress',
          4 : 'Coat',
          5 : 'Sandal',
          6 : 'Shirt',
          7 : 'Sneaker',
          8 : 'Bag',
          9 : 'Ankle boot'}

def display_image(features, actual_label):
    print("Actual label: ",lookup[actual_label])
    plt.imshow(features.reshape(28,28))
    

X = fashion_mnist_df[fashion_mnist_df.columns[1:]]
Y = fashion_mnist_df['label']

X.loc[5].values[:100]
Y.loc[5]

display_image(X.loc[5].values, Y.loc[5])
X = X / 255
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

def summarize_classification(y_test, y_pred, avg_method = 'weighted'):
    acc = accuracy_score(y_test, y_pred, normalize = True)
    num_acc = accuracy_score(y_test, y_pred, normalize = False)
    prec = precision_score(y_test, y_pred, average = 'weighted')
    recall = recall_score(y_test, y_pred, average = 'weighted')
    print("Test data count : ", len(y_test))
    print("Accuracy count : ", num_acc)
    print("Accuracy score : ", acc)
    print("Precision score : ", prec)
    print("Recall score : ", recall)
    print()
    
logistic_model = LogisticRegression(solver = 'sag', multi_class = 'auto', max_iter = 10000)
logistic_model.fit(x_train, y_train)
y_pred = logistic_model.predict(x_test)
y_pred

summarize_classification(y_test, y_pred)
