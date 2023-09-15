import numpy as np
import os
import pandas as pd
import pickle
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


import sys
sys.path.append('../model')
sys.path.append('../data_load_UM')
sys.path.append('..')
import load_dataset_onevisit
import utils
from generalized_fuzzy_net import GeneralizedFuzzyClassifier


def multiclass_roc_auc_score(truth, pred, average="macro"):
    lb = sklearn.preprocessing.LabelBinarizer()
    lb.fit(truth)
    truth = lb.transform(truth)
    pred = lb.transform(pred)            
    return sklearn.metrics.roc_auc_score(truth, pred, average=average)

#============= Experiment Configurations============================
# Step Up Parameters
out_root = './train_test_results'
unique_id = 'Nov_12th' 
random_state = 1234
#============= Experiment Configurations============================
exp_save_path = os.path.join(out_root, unique_id)

if not os.path.isdir(out_root):
    os.mkdir(out_root)
if not os.path.isdir(exp_save_path):
    os.mkdir(exp_save_path)
print('######################################')
print('Experiment ID:', unique_id)
print('######################################')

dataset = load_dataset_onevisit.load_NSF_dataset(missing_value_tolerance = 10,fill_missing=True)

row_name_list = []
row_list = []

data = np.array(dataset['variables'])
labels = np.array(dataset['response'])

split_method = dataset['split_method'] if 'split_method' in dataset else 'sample_wise'
category_info = dataset['category_info']
num_classes = dataset['num_classes']
rule_data = [{'Relation': [[52, 0], [30, 0]], 'Out_weight': 1}, 
                 {'Relation': [[33, 0], [52, 0], [30, 0]], 'Out_weight': 1}, {'Relation': [[52, 0], [53, 3], [53, 4]], 'Out_weight': 1},
                 {'Relation': [[30, 0], [49, 2]], 'Out_weight': 1}, {'Relation': [[30, 0], [37, 0]], 'Out_weight': 1}]
feature_names = dataset.get('feature_names')
max_steps = 8000


ss_train_test = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
X_train, y_train, X_test, y_test = utils.split_dataset(ss_train_test, data, labels, split_method, index=0)
X_train = utils.fill_in_missing_value(X_train,X_train) # fill_in_missing_value_MI
X_test = utils.fill_in_missing_value(X_test,X_test)
print('*****************************************************************')
print('number of selected_features are :',len(feature_names))
print('train')
print(X_train.shape)
print(y_train.shape)
print('test')
print(X_test.shape)
print(y_test.shape)
print('*****************************************************************')
model_name = 'GFN'
regu_ls = [1e-05]
for regu in regu_ls:
    rule_data = rule_data
    init_rule_index =  'All'
    ws = 1.2
    classifier = GeneralizedFuzzyClassifier(
                n_classes=num_classes,
                max_steps=max_steps, #100000
                category_info=category_info,
                batch_size = 50,
                report_freq = 50, # 50
                patience_step = 500, # 500
                random_state=1995,
                epsilon_training=False,
                binary_pos_only=True,
                weighted_loss=[1.0, ws], #
                split_method=split_method,
                verbose=2,
                val_ratio=0.3,
                init_rule_index = init_rule_index, #'All'
                rule_data = rule_data,
                corr_regu = 4e-07,
                learning_rate = 0.0054,
                min_epsilon = 0.9,
                n_rules = 20,
                sparse_regu = regu
                )

    classifier.fit(X_train, y_train)
    # m_path = os.path.join(exp_save_path,'saved_model_GFN_trainandtest_regu_1e-05.mdl')
    # classifier = pickle.load(open(m_path, 'rb'))
    evaluation_name = ['accuracy',  'recall', 'specificity', 'precision',
                    'f1', 'auc', 'aucpr','MCC','TP','TN','FP','FN']
    print('************* regu : {} *********'.format(regu))
    print(evaluation_name)
    _, train_metrics, _, _, _, _, _,_,_ = utils.cal_acc_full(classifier, X_train, y_train, num_classes>2)
    print(train_metrics)
    # Calculate test accuracy
    _, test_metrics, _, _, _, fpr_test, tpr_test,_,_ = utils.cal_acc_full(classifier, X_test, y_test, num_classes>2)
    print(test_metrics)
    # pickle.dump(classifier, open(os.path.join(exp_save_path, f'saved_model_{model_name}_trainandtest_regu_{regu}.mdl'), 'wb'))
    print('****************************************')