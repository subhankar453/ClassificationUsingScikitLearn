import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

titanic_df = pd.read_csv('datasets/titanic/train-processed.csv')
features = list(titanic_df.columns[1:])
result_dict = {}

def summarize_classification(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred, normalize = True)
    num_acc = accuracy_score(y_test, y_pred, normalize = False)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return {'accuracy' : acc,
            'precision' : prec,
            'recall' : recall,
            'accuracy_count' : num_acc}

def build_model(classifier_fn,
                name_of_y_col,
                name_of_x_cols,
                dataset,
                test_frac = 0.2):
    X = dataset[name_of_x_cols]
    Y = dataset[name_of_y_col]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = test_frac)
    model = classifier_fn(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    test_summary = summarize_classification(y_test, y_pred)
    train_summary = summarize_classification(y_train, y_pred_train)
    pred_results = pd.DataFrame({'y_test' : y_test, 'y_pred' : y_pred})
    model_crosstab = pd.crosstab(pred_results.y_test, pred_results.y_pred)
    return {'training' : train_summary,
            'test' : test_summary,
            'confusion matrix' : model_crosstab}

def compare_results():
    for key in result_dict:
        print("Classification : ", key)
        print()
        print("Training data")
        for score in result_dict[key]['training']:
            print(score, result_dict[key]['training'][score])
        print()
        print("Test data")
        for score in result_dict[key]['test']:
            print(score, result_dict[key]['test'][score])
        print()
        
def logistic_fn(x_train, y_train):
    model = LogisticRegression(solver = 'liblinear')
    model.fit(x_train, y_train)
    return model

result_dict['survived - logistic'] = build_model(logistic_fn,
                                                 'Survived',
                                                 features,
                                                 titanic_df)
    
compare_results()

#svd = singular value decomposition
def linear_discriminant_fn(x_train, y_train, solver = 'svd'):
    model = LinearDiscriminantAnalysis(solver = solver)
    model.fit(x_train, y_train)
    return model

result_dict['survived - linear_discriminant_analysis'] = build_model(linear_discriminant_fn,
                                                                     'Survived',
                                                                     features[0:-1],
                                                                     titanic_df)

def quadratic_discriminant_fn(x_train, y_train):
    model = QuadraticDiscriminantAnalysis()
    model.fit(x_train, y_train)
    return model

result_dict['survived - quadratic_discriminant_analysis'] = build_model(quadratic_discriminant_fn,
                                                                              'Survived',
                                                                              features[0:-1],
                                                                              titanic_df)

def sgd_fn(x_train, y_train, max_iter = 10000, tol = 1e-3):
    model = SGDClassifier(max_iter = max_iter, tol = tol)
    model.fit(x_train, y_train)
    return model

result_dict['survived - sgd'] = build_model(sgd_fn,
                                            'Survived',
                                            features,
                                            titanic_df)

def linear_svc_fn(x_train, y_train, C = 1.0, max_iter = 10000, tol = 1e-3):
    model = SVC(kernel = 'linear', C = C, max_iter = max_iter, tol = tol)
    model.fit(x_train, y_train)
    return model

result_dict['survived - linear_svc'] = build_model(linear_svc_fn,
                                                   'Survived', 
                                                   features,
                                                   titanic_df)

def radius_neighbor_fn(x_train, y_train, radius = 40.0):
    model = RadiusNeighborsClassifier(radius = radius)
    model.fit(x_train, y_train)
    return model

result_dict['survived - radius_neighbors'] = build_model(radius_neighbor_fn,
                                                         'Survived',
                                                         features,
                                                         titanic_df)

def decision_tree_fn(x_train, y_train, max_depth = None, max_features = None):
    model = DecisionTreeClassifier(max_depth = max_depth, max_features = max_features)
    model.fit(x_train, y_train)
    return model

result_dict['survived - decision_tree'] = build_model(decision_tree_fn,
                                                      'Survived',
                                                      features,
                                                      titanic_df)

def naive_bayes_fn(x_train, y_train, priors = None):
    model = GaussianNB(priors = priors)
    model.fit(x_train, y_train)
    return model

result_dict['survived - naive_bayes'] = build_model(naive_bayes_fn,
                                                    'Survived',
                                                    features,
                                                    titanic_df)