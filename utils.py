# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 13:38:31 2020

@author: hemingy
"""
import numpy as np
import pandas as pd
import random
import sklearn
from sklearn.utils.fixes import loguniform
from itertools import repeat
import matplotlib.pyplot as plt
import miceforest as mf

from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report



def draw_ROC_curves(tprs_list, fprs_list, aucs_list, model_name, output_name):
    """ Rraw ROC curves with confidence interval.

    """
    mean_fpr = np.linspace(0, 1, 100)
    interp_tprs_list = []
    fig, ax = plt.subplots(figsize=(6,4))
    for i in range(len(aucs_list)):
        interp_tpr = np.interp(mean_fpr, fprs_list[i], tprs_list[i])
        interp_tpr[0] = 0.0
        interp_tprs_list.append(interp_tpr)
   
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
   
    mean_tpr = np.mean(interp_tprs_list, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs_list)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label='Mean ROC\n' + r'(AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
   
    std_tpr = np.std(interp_tprs_list, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
   
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title=f"ROC curve for {model_name}")
    ax.legend(loc="lower right")
    plt.savefig(output_name, dpi=300)


def show_metrics(metrics, show_value_list=None):
    """ Calculate the average and standard deviation from multiple repetitions and format them.
    """
    eval_m, eval_s = np.nanmean(metrics, 0), np.nanstd(metrics,0)
    show_results(eval_m, s=eval_s, show_full=False)  
    for i in range(eval_m.shape[0]):
        show_value_list.append('{:.3f} ({:.3f})'.format(eval_m[i], eval_s[i]))
    return show_value_list

   
def split_dataset(stratified_split, data, labels, split_method, index=0):        
    """ Split the dataset into the training set and test set.

    Parameters:
    ---------- 
    stratified_split : An instance of the object sklearn.model_selection.StratifiedShuffleSplit.
    data : np.ndarray. A np.array of features with a shape (the number of samples, the number of features).
    labels : np.dnarray. A np.ndarray of labels with a shape (the number of samples,).
    split_method : Str. It indicates how the train/val/test data split should be performed.
        Options include 'patient_wise', and 'sample_wise'. 'sample_wise' is the regular split. 
        For 'patient_wise', data samples from the same patient should be put into the same data set.
    index : Int. The index of the repetition.

    Returns:
    ---------- 
    X_train : np.ndarray with a shape (the number of training samples, the number of features). Training features.
    y_train : np.ndarray with a shape (the number of training samples,). Training labels.
    X_test : np.ndarray with a shape (the number of training samples, the number of features). Testing features.
    y_test : np.ndarray with a shape (the number of training samples,). Testing features.
    
    """
    if split_method == 'patient_wise':
        uids = data[:, 0].astype(np.uint64)        
        uids_HT_VAD_set = set(uids[labels==1])
        uids_too_well_set = set(uids[labels==0]).difference(uids_HT_VAD_set)
        uids_HT_VAD_arr = np.array(list(uids_HT_VAD_set))
        uids_too_well_arr = np.array(list(uids_too_well_set))
        
        uids = np.concatenate([uids_HT_VAD_arr, uids_too_well_arr], axis=0)
        uids_label = np.concatenate([np.ones(uids_HT_VAD_arr.shape[0]), 
                                     np.zeros(uids_too_well_arr.shape[0])], axis=0)
    
        index = list(stratified_split.split(uids, uids_label))[index]
        
        uids_train = np.take(uids, index[0], axis=0)
        uids_test = np.take(uids, index[1], axis=0)
    
        train_index = [i for i in range(data.shape[0]) if data[i,0] in uids_train]
        test_index = [i for i in range(data.shape[0]) if data[i,0] in uids_test]
        
    elif split_method == 'sample_wise':
        # Split the data for nested cross-validation on the training set
        index = list(stratified_split.split(data, labels))[index]
        train_index = index[0]
        test_index = index[1]
    
    else:
        raise NotImplementedError
    # print('The overlapping uids of training and testing is {}'.format(set(uids_train).intersection(set(uids_test))))
    # print('The uids of training is {}'.format(set(uids_train)))
    # print('The uids of validation is {}'.format(set(uids_test)))

        
    X_train, X_test = np.take(data, train_index, axis=0), np.take(data, test_index, axis=0)
    y_train, y_test = np.take(labels, train_index, axis=0), np.take(labels, test_index, axis=0)  
        
    return X_train.astype(np.float32), y_train.astype(np.int32), X_test.astype(np.float32), y_test.astype(np.int32)


def split_dataset_multi(stratified_split, data, labels, split_method, index=0):
    
    if split_method == 'patient_wise':
        uids = data[:, 0].astype(np.uint64)        
        uids_HT_VAD_set = set(uids[labels==1])
        uids_too_well_set = set(uids[labels==2])
        uids_contra_HT_set = set(uids[labels==3])
        
        uids_HT_VAD_arr = np.array(list(uids_HT_VAD_set))
        uids_too_well_arr = np.array(list(uids_too_well_set))
        uids_contra_HT_arr = np.array(list(uids_contra_HT_set))
        
        
        uids = np.concatenate([uids_HT_VAD_arr, uids_too_well_arr,uids_contra_HT_arr], axis=0)
        uids_label = np.concatenate([np.ones(uids_HT_VAD_arr.shape[0]), 
                                     np.ones(uids_too_well_arr.shape[0])+1,
                                     np.ones(uids_contra_HT_arr.shape[0])+2,
                                     ], axis=0)
        index = list(stratified_split.split(uids, uids_label))[index]
        
        uids_train = np.take(uids, index[0], axis=0)
        uids_test = np.take(uids, index[1], axis=0)
    
        train_index = [i for i in range(data.shape[0]) if data[i,0] in uids_train]
        test_index = [i for i in range(data.shape[0]) if data[i,0] in uids_test]

    X_train, X_test = np.take(data, train_index, axis=0), np.take(data, test_index, axis=0)
    y_train, y_test = np.take(labels, train_index, axis=0), np.take(labels, test_index, axis=0)  
        
    return X_train.astype(np.float32), y_train.astype(np.int32), X_test.astype(np.float32), y_test.astype(np.int32)

def split_and_downsample_multi(stratified_split, data, labels, split_method, index=0,downsample = False):
    
    if split_method == 'patient_wise':
        uids = data[:, 0].astype(np.uint64)        
        uids_HT_VAD_set = set(uids[labels==1])
        uids_too_well_set = set(uids[labels==2])
        uids_contra_HT_set = set(uids[labels==3])
        
        if downsample:
            random.seed(1234)
            uids_HT_VAD_set = random.sample(uids_HT_VAD_set, len(uids_contra_HT_set))
            uids_too_well_set = random.sample(uids_too_well_set, len(uids_contra_HT_set))


        uids_HT_VAD_arr = np.array(list(uids_HT_VAD_set))
        uids_too_well_arr = np.array(list(uids_too_well_set))
        uids_contra_HT_arr = np.array(list(uids_contra_HT_set))
        
        
        uids = np.concatenate([uids_HT_VAD_arr, uids_too_well_arr,uids_contra_HT_arr], axis=0)
        uids_label = np.concatenate([np.ones(uids_HT_VAD_arr.shape[0]), 
                                     np.ones(uids_too_well_arr.shape[0])+1,
                                     np.ones(uids_contra_HT_arr.shape[0])+2,
                                     ], axis=0)
        index = list(stratified_split.split(uids, uids_label))[index]
        
        uids_train = np.take(uids, index[0], axis=0)
        uids_test = np.take(uids, index[1], axis=0)
    
        train_index = [i for i in range(data.shape[0]) if data[i,0] in uids_train]
        test_index = [i for i in range(data.shape[0]) if data[i,0] in uids_test]

    X_train, X_test = np.take(data, train_index, axis=0), np.take(data, test_index, axis=0)
    y_train, y_test = np.take(labels, train_index, axis=0), np.take(labels, test_index, axis=0)  
        
    return X_train.astype(np.float32), y_train.astype(np.int32), X_test.astype(np.float32), y_test.astype(np.int32)








def split_dataset_REVIVAL(data, labels, split_method, index=0):       
    """ Split the heart failure dataset (REVIVAL+INTERMACS) into the training set and test set.
    The training set includes all samples from the INTERMACS registry and ~80% of the negative samples from the REVIVAL registry.
    The validation and test sets contain the remaining negative samples and all positive samples from the REVIVAL registry.
    Parameters:
    ---------- 
    data : np.ndarray. A np.array of features with a shape (the number of samples, the number of features).
    labels : np.dnarray. A np.ndarray of labels with a shape (the number of samples,).
    split_method : Str. It indicates how the train/val/test data split should be performed.
        Options include 'patient_wise', and 'sample_wise'. 'sample_wise' is the regular split. 
        For 'patient_wise', data samples from the same patient should be put into the same data set.
    index : Int. The index of the repetition.

    Returns:
    ---------- 
    X_train : np.ndarray with a shape (the number of training samples, the number of features). Training features.
    y_train : np.ndarray with a shape (the number of training samples,). Training labels.
    X_val : np.ndarray with a shape (the number of training samples, the number of features). Validation features.
    y_val : np.ndarray with a shape (the number of training samples,). Validation labels.
    X_test : np.ndarray with a shape (the number of training samples, the number of features). Testing features.
    y_test : np.ndarray with a shape (the number of training samples,). Testing features.
    
    """
    from sklearn.model_selection import train_test_split
    uids = data[:, 0]        
    
    uids_HT_VAD_revival = set(uids[np.all([labels==1, uids<40000], axis=0)])
    uids_too_well_revival = list(set(uids[np.all([labels==0, uids<40000], axis=0)]).difference(uids_HT_VAD_revival))
    uids_HT_VAD_revival = list(uids_HT_VAD_revival)
    
    print('The number of uids with positive samples in REVIVAL', len(uids_HT_VAD_revival))
    print('The number of positive samples in REVIVAL', uids[np.all([labels==1, uids<40000], axis=0)].shape)

    # Split positive samples from REVIVAL for validation and test only
    uids_HT_VAD_revival_test, uids_HT_VAD_revival_val = train_test_split(uids_HT_VAD_revival, test_size = 0.50, random_state=index)

    # 10% of negative samples from REVIVAL are put in the validation set
    # 10% of negative samples from REVIVAL are put in the test set
    uids_too_well_revival_rem, uids_too_well_revival_test = train_test_split(uids_too_well_revival, test_size = 0.10, random_state=index)
    uids_too_well_revival_val = train_test_split(uids_too_well_revival_rem, test_size = 0.10, random_state=index)[1]
    
    uids_test = set(uids_HT_VAD_revival_test).union(set(uids_too_well_revival_test))
    uids_val = set(uids_HT_VAD_revival_val).union(set(uids_too_well_revival_val))
    
    uids_train = set(uids) - uids_test - uids_val
    
    train_index = [i for i in range(data.shape[0]) if data[i,0] in uids_train]
    val_index = [i for i in range(data.shape[0]) if data[i,0] in uids_val]
    test_index = [i for i in range(data.shape[0]) if data[i,0] in uids_test]
        
    
    X_train, X_val, X_test = np.take(data, train_index, axis=0), np.take(data, val_index, axis=0), np.take(data, test_index, axis=0)
    y_train, y_val, y_test = np.take(labels, train_index, axis=0), np.take(labels, val_index, axis=0), np.take(labels, test_index, axis=0)  
        
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size = 0.2, random_state=index)
    
    return X_train.astype(np.float32), y_train.astype(np.int32), \
            X_val.astype(np.float32), y_val.astype(np.int32), \
             X_test.astype(np.float32), y_test.astype(np.int32)

             
def standardize(features, scaler, category_list):
    if scaler is not None:
        if category_list is not None:
            features_continous = features[:, category_list==0]
            scaled_features = features.copy()
            scaled_features[:, category_list==0] = scaler.transform(features_continous)
        else:
            scaled_features = scaler.transform(features)
    else:
        scaled_features = features
        
    return scaled_features


def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data
            
            
def show_results(m, s=None, show_full=False, name=None):
    """ Present the performance of the model.

    Parameters
    ----------
    m : np.ndarray. Mean value of the evaluation metrics including accuracy, sensitivity, specificity, precision, f1-score, auc, and aucpr.
    s : np.ndarry, optional. Standard deviation of the evaluation metrics including accuracy, sensitivity, specificity, precision, f1-score,
        auc, and aucpr.
    show_full : boolean, optional. Whether to show the full evaluation metrics The default is False.
    name: string.
    -------
    None.

    """
    # Show the percentage
    m = m*100
    if s is not None:
        s = s*100
        
    if name:
        print(f'{name} evaluation metrics:')
    if show_full:
        string = '\t Mean acc is {:.2f}, sen is {:.2f}, spe is {:.2f}, prec is {:.2f}, f1 is {:.2f}, auc is {:.2f}, aucpr is {:.2f}'.\
               format(m[0],m[1],m[2],m[3],m[4],m[5],m[6])
        print(string)
        
        if s is not None:            
            string = '\t Std acc is {:.2f}, sen is {:.2f}, spe is {:.2f}, prec is {:.2f}, f1 is {:.2f}, auc is {:.2f}, aucpr is {:.2f}'.\
                   format(s[0],s[1],s[2],s[3],s[4],s[5],s[6])
            print(string)
    else:
        if s is None:
            string = '\t auc is {:.2f}, aucpr is {:.2f}, F1-score is {:.2f}'.\
                   format(m[5],m[6],m[4])
        else:
            string = '\t auc is {:.2f} ({:.2f}), aucpr is {:.2f} ({:.2f}), F1-score is {:.2f} ({:.2f})'.\
                   format(m[5],s[5],m[6],s[6],m[4],s[4])
        print(string)
            
            
def indices_to_one_hot(data, n_classes=None):
    """ Convert an iterable of indices to one-hot encoded labels.
    

    Parameters
    ----------
    data : list. A list of integers.
    n_classes: int. The number of classes.

    Returns
    -------
    One-hot encoded labels.
    
    """
    if n_classes is None:
        n_classes = np.max(data) + 1
    targets = np.array(data).reshape(-1)
    return np.eye(n_classes)[targets]


def cal_acc(model, features, labels, multiple):
    """ Model prediction and evaluation metrics calculation (For classification only).
    
    Parameters
    ----------
    model : An estimator object. The trained model.
    features : np.ndarray. Normalized features.
    labels :  np.ndarray. Labels.
    multiple : boolean. Whether it is a multi-class classification task.

    Returns
    -------
    predictions : np.ndarray. Predictions from the traine model on the given features.
    metrics: A tuple of float. They are recall, specificity, precision, accuracy, f1, auc, aucpr, respectively.
        For multi-class classification, the values from individual classes are averaged.

    """
    if not isinstance(model, list):
        #predictions = model.predict(features)
        probs = model.predict_proba(features)
        predictions = np.argmax(probs, axis=-1)
        
    else:
        # A list of classifiers
        probs_list = []
        for voter in model:
            probs = voter.predict_proba(features)
            probs_list.append(probs)
        probs = np.mean(np.stack(probs_list, axis=-1), axis=-1)
        predictions = np.argmax(probs, axis=-1)
    
    # Calculate matrix
    if multiple:
        average_method = 'macro'
    else:
        average_method = 'binary'

    recall = sklearn.metrics.recall_score(labels, predictions, average=average_method)
    precision = sklearn.metrics.precision_score(labels, predictions, average=average_method, zero_division=0)
    specificity = sklearn.metrics.recall_score(1-labels, 1-predictions, average=average_method)
    accuracy = sklearn.metrics.accuracy_score(labels, predictions)
    #f1 = sklearn.metrics.f1_score(labels, predictions, average=average_method)
    f1 = 2*precision*recall/(precision + recall + 0.1)
    
    if multiple:
        aucpr = sklearn.metrics.average_precision_score(indices_to_one_hot(labels), 
                                                        probs, average=average_method)
        auc = sklearn.metrics.roc_auc_score(indices_to_one_hot(labels, np.max(labels)+1), probs,
                                            average=average_method, multi_class='ovr')
        fpr, tpr = None
    else:
        aucpr = sklearn.metrics.average_precision_score(labels, probs[:,1])
        auc = sklearn.metrics.roc_auc_score(indices_to_one_hot(labels), probs)
        fpr, tpr, _ = sklearn.metrics.roc_curve(labels, probs[:,1])

    metrics = np.array([accuracy, recall, specificity, precision, f1, auc, aucpr])
    metrics_name = ['accuracy',  'recall', 'specificity', 'precision',
                    'f1', 'auc', 'aucpr']
    return predictions, metrics, metrics_name, probs, labels, fpr, tpr


def create_RF_grid():
    """ Search grid for hyper-parameter tuning on random forest model.

    Returns
    -------
    param_grid: dict. Keys are the name of hyper-paramters for tuning. 
            Values are the search range.

    """
    # Create the grid
    param_grid = {'n_estimators': [int(x) for x in np.linspace(200, 1000, 5)],
                # Maximum number of levels in tree
                'max_depth': [int(x) for x in np.linspace(5, 25, 10)]+[None],
                # Number of features to consider at every split
                'min_samples_split': [int(x) for x in np.linspace(2, 10, 9)],
                # Minimum decrease in impurity required for split to happen
                'max_features': ['sqrt', 'log2'],
                }
    return param_grid


def create_DT_grid():
    """ Search grid for hyper-parameter tuning on random forest model.

    Returns
    -------
    param_grid: dict. Keys are the name of hyper-paramters for tuning. 
            Values are the search range.

    """
    # Create the grid
    param_grid = {
                'max_depth': [int(x) for x in np.linspace(5, 25, 10)] + [None],
                # Minimum number of samples required to split a node
                'min_samples_split': [int(x) for x in np.linspace(2, 10, 9)]
                # Minimum decrease in impurity required for split to happen
                }
    return param_grid


def create_EBM_grid(binary):
    """ Search grid for hyper-parameter tuning on Explainable Boosting Classifier.

    Returns
    -------
    param_grid: dict. Keys are the name of hyper-paramters for tuning. 
            Values are the search range.

    """
    # Create the grid
    if binary:
        param_grid = {'interactions': [0,1,2],
                   # Method to bin values for pre-processing.
                   # Learning rate for boosting
                   'learning_rate': loguniform(1e-3, 1),
                    }
    else:
        # Multiclass with interactions currently not supported in interpret.glassbox.
        param_grid = { # Method to bin values for pre-processing.
                   # Learning rate for boosting
                   'learning_rate': loguniform(1e-3, 1),
                    } 
    return param_grid


def create_SVM_grid():
    """ Search grid for hyper-parameter tuning on SVM with non-linear kernel.

    Returns
    -------
    param_grid: dict. Keys are the name of hyper-paramters for tuning. 
            Values are the search range.

    """
    param_grid = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                  # Regularization parameter.
                  'C': loguniform(1e-1, 1e3), 
                  # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
                  'gamma': loguniform(1e-4, 1e0)
                  }
    return param_grid


def create_XGBoost_grid():
    """ Search grid for hyper-parameter tuning on XGBoost classifier.

    Returns
    -------
    param_grid: dict. Keys are the name of hyper-paramters for tuning. 
            Values are the search range.

    """
    param_grid = {'n_estimators': [int(x) for x in np.linspace(200, 2000, 10)],
                # Maximum tree depth for base learners
                'max_depth': [int(x) for x in np.linspace(2, 20, 10)],
                # Learning rate
                'eta': [x for x in np.linspace(0.1, 1, 10)],
               }
    return param_grid


def create_GFN_grid():
    """ Search grid for hyper-parameter tuning on generalized fuzzy network.

    Returns
    -------
    param_grid: dict. Keys are the name of hyper-paramters for tuning. 
            Values are the search range.

    """
    # Used for categorical dataset
    # param_grid = {'min_epsilon': [x for x in np.linspace(0.4,0.9,6)],
    #               'n_rules': [20,25,30],
    #               'learning_rate': loguniform(5e-3, 1e-1),
    #               'sparse_regu': [0],
    #               'corr_regu': loguniform(1e-5, 1e-2),
    #             }

    # # Used for complicated functions
    # param_grid = {'min_epsilon': [x for x in np.linspace(0.4,0.9,6)], #0.2
    #               'n_rules': [20,25,30],
    #               'learning_rate': loguniform(5e-3, 1e-1),
    #               'sparse_regu': loguniform(1e-5, 1e-2),
    #               'corr_regu': loguniform(1e-5, 1e-2),
    #             }
                
    ## Used for REVIVAL dataset 
    param_grid = {'min_epsilon': [x for x in np.linspace(0.4,0.9,6)],
                  'n_rules': [20,25,30],
                  'learning_rate': loguniform(5e-3, 1e-1),
                  'sparse_regu': [0],
                  'corr_regu': loguniform(1e-8, 1e-4),
                }
    
    return param_grid

def calculate_MCC(TP,TN,FP,FN):
      denom = (TP+FP) * (TP +FN) * (TN+FN) * (TN+FP)
      if denom == 0:
          denom += 0.01
      return (TP*TN-FP*FN)/np.sqrt(denom)

def cal_acc_full(model, features, labels, multiple):
    """ Model prediction and evaluation metrics calculation (For classification only).
    
    Parameters
    ----------
    model : An estimator object. The trained model.
    features : np.ndarray. Normalized features.
    labels :  np.ndarray. Labels.
    multiple : boolean. Whether it is a multi-class classification task.

    Returns
    -------
    predictions : np.ndarray. Predictions from the traine model on the given features.
    metrics: A tuple of float. They are recall, specificity, precision, accuracy, f1, auc, aucpr, respectively.
        For multi-class classification, the values from individual classes are averaged.

    """
    if not isinstance(model, list):
        #predictions = model.predict(features)
        probs = model.predict_proba(features)
        predictions = np.argmax(probs, axis=-1)
        
    else:
        # A list of classifiers
        probs_list = []
        for voter in model:
            probs = voter.predict_proba(features)
            probs_list.append(probs)
        probs = np.mean(np.stack(probs_list, axis=-1), axis=-1)
        predictions = np.argmax(probs, axis=-1)
    
    # Calculate matrix
    if multiple:
        average_method = 'macro'
    else:
        average_method = 'binary'

    recall = sklearn.metrics.recall_score(labels, predictions, average=average_method)
    precision = sklearn.metrics.precision_score(labels, predictions, average=average_method, zero_division=0)
    specificity = sklearn.metrics.recall_score(1-labels, 1-predictions, average=average_method)
    accuracy = sklearn.metrics.accuracy_score(labels, predictions)
    #f1 = sklearn.metrics.f1_score(labels, predictions, average=average_method)
    f1 = 2*precision*recall/(precision + recall + 0.1)
    
    
    # TP, FP, TN, FN
    TP = np.sum((predictions == 1) & (labels == 1))
    FP = np.sum((predictions == 1) & (labels == 0))
    TN = np.sum((predictions == 0) & (labels == 0))
    FN = np.sum((predictions == 0) & (labels == 1))
    MCC = calculate_MCC(TP,TN,FP,FN)
    
    if multiple:
        aucpr = sklearn.metrics.average_precision_score(indices_to_one_hot(labels), 
                                                        probs, average=average_method)
        auc = sklearn.metrics.roc_auc_score(indices_to_one_hot(labels, np.max(labels)+1), probs,
                                            average=average_method, multi_class='ovr')
        fpr, tpr = None
    else:
        aucpr = sklearn.metrics.average_precision_score(labels, probs[:,1])
        auc = sklearn.metrics.roc_auc_score(indices_to_one_hot(labels), probs)
        fpr, tpr, _ = sklearn.metrics.roc_curve(labels, probs[:,1],drop_intermediate = False)
        precision_p, recall_p, _ = sklearn.metrics.precision_recall_curve(labels, probs[:,1])

    metrics = np.array([accuracy, recall, specificity, precision, f1, auc, aucpr,MCC, TP,TN,FP,FN])
    metrics_name = ['accuracy',  'recall', 'specificity', 'precision',
                    'f1', 'auc', 'aucpr','MCC','TP','TN','FP','FN']
    return predictions, metrics, metrics_name, probs, labels, fpr, tpr,precision_p, recall_p

def cal_acc_multi(model, features, labels):
    

    probs = model.predict_proba(features)
    fpr, tpr, thresholds = roc_curve(labels[:,1], probs[:,1],pos_label=1)
    

    spec_and_sens_ls = []
    accuracy_scores = []
    
    average_method = 'macro'
    for thresh in thresholds:
        pred = np.array([m > thresh for m in probs[:,1]])
        specificity = sklearn.metrics.recall_score(1-labels[:,1], 1-pred)
        recall = sklearn.metrics.recall_score(labels[:,1], pred)
        spec_and_sens = specificity + recall
        spec_and_sens_ls.append(spec_and_sens)
        # accuracy_scores.append(accuracy_score(labels, pred))
        # accuracy_scores.append(accuracy_score(labels, [m > thresh for m in probs]))
    spec_and_sens_ls = np.array(spec_and_sens_ls)
    # accuracy_scores = np.array(accuracy_scores)
    optim_threshold =  thresholds[spec_and_sens_ls.argmax()]
    print('The optimal threshold is {:.3f}'.format(optim_threshold,3))

    
    predictions = probs >= optim_threshold

    recall = sklearn.metrics.recall_score(labels, predictions, average=average_method)
    precision = sklearn.metrics.precision_score(labels, predictions, average=average_method, zero_division=0)
    specificity = sklearn.metrics.recall_score(1-labels, 1-predictions, average=average_method)
    accuracy = sklearn.metrics.accuracy_score(labels, predictions)
    auc = roc_auc_score(labels, probs, average=average_method)
    aucpr = average_precision_score(labels, probs, average=average_method)
    f1 = sklearn.metrics.f1_score(labels, predictions, average=average_method)
    
    metrics = np.round(np.array([accuracy, recall, precision, specificity, f1, auc, aucpr]),3)
    metrics_name = ['accuracy', 'recall','precision','specificity','f1', 'auc', 'aucpr']
    np.savetxt('pred.out', predictions, delimiter=',')
    np.savetxt('probs.out', probs, delimiter=',')
    np.savetxt('labels.out', labels, delimiter=',')

    label_names = ['Have AT','AT but have contradictions']
    print(classification_report(labels, predictions,target_names=label_names,zero_division=0))

    return predictions, metrics, metrics_name, probs, labels




def most_common(lst):
    if len(lst) >= 2:
        lst = [x for x in lst if str(x) != 'nan']
        if len(lst) != 0:
            v = max(set(lst), key=lst.count)
        else:
            v = np.nan
    elif len(lst) == 1:
        v = lst[0]
    elif len(lst) == 0:
        v = np.nan
    return v

def fill_in_missing_value(features_A,features_B):
    """ Fill in missing values using the average of the feature values.
    
    features: np.ndarray. A np.array of features with a shape (the number of samples, the number of features)
    """
    m_list = np.array([most_common(features_A[:,k]) for k in range(features_A.shape[1])])
    m_mat = np.stack([m_list]*features_B.shape[0],axis=0)
    features_B[pd.isnull(features_B)] = m_mat[pd.isnull(features_B)]
    return features_B


def fill_in_missing_value_MI(features_A):
    """ Fill in missing values using the average of the feature values.
    
    features: np.ndarray. A np.array of features with a shape (the number of samples, the number of features)
    """
    
    kds = mf.ImputationKernel(features_A,save_all_iterations=True,random_state=1991)
    kds.mice(2)
    features_A = kds.complete_data()
    return features_A

def clamp_ALT(ls):
    alt = np.where(ls > 60, 100,ls)
    return alt