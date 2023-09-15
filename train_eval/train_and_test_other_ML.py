import numpy as np
import os,xgboost
import pandas as pd
import pickle
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC  
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB

import load_dataset_onevisit
import utils


def multiclass_roc_auc_score(truth, pred, average="macro"):
    lb = sklearn.preprocessing.LabelBinarizer()
    lb.fit(truth)
    truth = lb.transform(truth)
    pred = lb.transform(pred)            
    return sklearn.metrics.roc_auc_score(truth, pred, average=average)


random_state = 1234


dataset = load_dataset_onevisit.load_NSF_dataset(missing_value_tolerance = 10,fill_missing=True)

row_name_list = []
row_list = []

data = np.array(dataset['variables'])
labels = np.array(dataset['response'])

split_method = dataset['split_method'] if 'split_method' in dataset else 'sample_wise'
category_info = dataset['category_info']
num_classes = dataset['num_classes']
rule_data = dataset['rule_data']
feature_names = dataset.get('feature_names')
max_steps = 8000


ss_train_test = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
X_train, y_train, X_test, y_test = utils.split_dataset(ss_train_test, data, labels, split_method, index=0)
X_train = utils.fill_in_missing_value(X_train,X_train)
X_test = utils.fill_in_missing_value(X_train,X_test)
print('*****************************************************************')
print('number of selected_features are :',len(feature_names))
print('train')
print(X_train.shape)
print(y_train.shape)
print('test')
print(X_test.shape)
print(y_test.shape)
print(rule_data)
print('*****************************************************************')
model_sets = ['LR', 'DT', 'XGB','RF','SVM','NB']
for m_name in model_sets:
    if m_name == 'LR':
        print('************* regu : {} *********'.format(m_name))
        classifier = LogisticRegression(solver='lbfgs', random_state=random_state)
    elif m_name == 'NB': # Naive bayes
        print('************* regu : {} *********'.format(m_name))
        classifier = GaussianNB()
    elif m_name == 'DT':
        print('************* regu : {} *********'.format(m_name))
        classifier = sklearn.tree.DecisionTreeClassifier(min_samples_split = 7, max_depth = 5)
    elif m_name == 'SVM':
        print('************* regu : {} *********'.format(m_name))
        classifier = SVC(probability=True, random_state=random_state,C = 2.70, gamma = 0.01, kernel = 'rbf')
    elif m_name == 'RF':
        print('************* regu : {} *********'.format(m_name))
        classifier = RandomForestClassifier(random_state=random_state,n_estimators = 600, min_samples_split = 4, max_features = 'sqrt', max_depth=9)
    elif m_name == 'XGB':
        print('************* regu : {} *********'.format(m_name))
        classifier = xgboost.XGBClassifier(n_estimators = 200, max_depth = 16, eta = 0.6)




    classifier.fit(X_train, y_train)
    evaluation_name = ['accuracy',  'recall', 'specificity', 'precision',
                    'f1', 'auc', 'aucpr','MCC','TP','TN','FP','FN']
    print(evaluation_name)
    _, train_metrics, _, _, _, _, _,_,_ = utils.cal_acc_full(classifier, X_train, y_train, num_classes>2)
    print(train_metrics)
    # Calculate test accuracy
    _, test_metrics, _, _, _, fpr_test, tpr_test,_,_ = utils.cal_acc_full(classifier, X_test, y_test, num_classes>2)
    print(test_metrics)
    print('****************************************')