import os,sys
import numpy as np
import xgboost, os
import pandas as pd
import pickle
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC  
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
# import mod_fylearn.frr

import sys
sys.path.append('../model')

import load_dataset_onevisit
import utils
from generalized_fuzzy_net import GeneralizedFuzzyClassifier


def multiclass_roc_auc_score(truth, pred, average="macro"):
    lb = sklearn.preprocessing.LabelBinarizer()
    lb.fit(truth)
    truth = lb.transform(truth)
    pred = lb.transform(pred)            
    return sklearn.metrics.roc_auc_score(truth, pred, average=average)


def nested_cross_validation(X, y, num_classes, category_info, feature_names, rule_data, model_name, 
                            n_folds, n_folds_hyper_tuning, search_iters, max_steps, split_method, 
                            random_state):
    """
    Nested cross-validation.

    Parameters
    ---------- 
    X : np.ndarray. A np.array of features with a shape (the number of samples, the number of features)
    y : np.dnarray. A np.ndarray of labels with a shape (the number of samples,)
    num_classes : Int. The number of classes
    category_info : A np.array with a shape (the number of features).  entry category_info[i] = 0 means the i-th variable 
        is a continous variable. entry category_info[i] > 0 means the i-th variable is a categorical variable, and 
        the value is the number of levels. This information is only used in the proposed machine learning technique.
    feature_names : A list of feature names.
    rule_data : A list of existing rules
    model_name : Str. A string of model name. Options: 'RF', 'XGB', 'SVM', 'EBM', 'GFN_cv', 'DT', 'LR', 'NB', 'FLN', 'GFN'
    n_folds : Int. The number of folds in the outer cross-validation.
    n_folds_hyper_tuning : Int. The number of folds in the inner cross-validation.
    search_iters : Int. The number of search interations in the hyper-parameter tuning.
    max_steps : Int. The maximal steps in optimizing GFN algorithm.
    split_method : Str. It indicates how the train/val/test data split should be performed.
        Options include 'patient_wise', and 'sample_wise'. 'sample_wise' is the regular split. 
        For 'patient_wise', data samples from the same patient should be put into the same data set.
    random_state : Int. Random state.
    
    Returns
    -------
    fold_classifiers : A list of trained classifier objects from the outer cross-validation.
    eval_series : A pd.Series contains evaluation metrics from the outer cross-validation.
    roc_values : A dictionary with keys: 'fpr_test', 'tpr_test', 'auc_test' from the outer cross-validation. 
        It can help draw the ROC curve and its confidence interval from the outer cross-validation.

    """
    fold_train = np.zeros([n_folds, 12])
    fold_test = np.zeros([n_folds, 12])
    fold_classifiers = []
    roc_values = {'fpr_test': [],
                  'tpr_test': [],
                  'auc_test': []}
    aucpr_values = {'recall_test': [],
                  'precision_test': [],
                  'aucpr_test': []}
    
    # Split the dataset for the outer cross-validation
    ss = StratifiedShuffleSplit(n_splits=n_folds, test_size=0.20, random_state=random_state)
    # Specify models that requires hyper-parameter tuning (using the random searches and inner cross-validation)
    cv_network = ['RF', 'XGB', 'SVM', 'EBM', 'GFN', 'DT']
    
    for index in range(n_folds):
        X_train, y_train, X_val, y_val = utils.split_dataset(ss, X, y, split_method, index=index)

        # Train the model on training data 
        if not model_name.startswith('GFN'):
            scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)   
            X_val = scaler.transform(X_val)  
        
        # Instantiate model
        if model_name == 'SVM':
            base = SVC(probability=True, random_state=random_state)
            grid = utils.create_SVM_grid()
        elif model_name == 'RF': # random forest
            base = RandomForestClassifier(random_state=random_state)
            grid = utils.create_RF_grid()
        elif model_name == 'XGB': # XGBoost
            base = xgboost.XGBClassifier()#random_state=random_state
            grid = utils.create_XGBoost_grid()
        elif model_name == 'LR': # Logistic regression
            classifier = LogisticRegression(solver='lbfgs', random_state=random_state)
        elif model_name == 'NB': # Naive bayes
            classifier = GaussianNB()
        elif model_name == 'DT':
            base = sklearn.tree.DecisionTreeClassifier()
            grid = utils.create_DT_grid()
        elif model_name == 'FLN':  # Fuzzy Logic network
            classifier = mod_fylearn.frr.FuzzyReductionRuleClassifier()
        elif model_name == 'GFN': # Generalized fuzzy network
            rule_data = rule_data
            init_rule_index = 'All'
            ws = 1.2
            sparse_regu = 0
            sys.stdout.write('ws: %f\r' % ws)
            sys.stdout.flush()
            print(rule_data)
            print(init_rule_index)
            print(sparse_regu)
            base = GeneralizedFuzzyClassifier(
             n_classes=num_classes,
             max_steps=max_steps, #100000
             category_info=category_info,
             batch_size = 50,
             report_freq = 50, # 50
             patience_step = 500, # 500
             random_state=random_state,
             epsilon_training=False,
             binary_pos_only=True,
             weighted_loss=[1.0, ws], #
             split_method=split_method,
             sparse_regu = sparse_regu,
             verbose=0,
             val_ratio=0.3,
             init_rule_index = init_rule_index, # '0_1_2_3_4_5_6_7', None
             rule_data = rule_data
             ) 
            grid = utils.create_GFN_grid()

        # Hyper-parameter tuning
        if model_name in cv_network: #
            import platform
            if platform.system() == 'Windows':
                n_jobs = 1
            else:
                n_jobs = -1
            
            # Inner cross-validation
            classifier = RandomizedSearchCV(estimator=base, param_distributions=grid, 
                                           n_iter=search_iters, cv=n_folds_hyper_tuning, 
                                           verbose=1, random_state=random_state,
                                           n_jobs=n_jobs, scoring=sklearn.metrics.make_scorer(multiclass_roc_auc_score))

        # Fit the random search model
        classifier.fit(X_train, y_train)
        if model_name in cv_network: 
            # Print hyper-parameters selected from the inner cross-validation
            print(classifier.best_params_)
            best_classifier = classifier.best_estimator_
        else:
            best_classifier = classifier
            
        # Add the classifier with optical combination of hyper-parmaeter from each fold
        best_classifier = loaded_models[index]
        fold_classifiers.append(best_classifier)
        
        # Calculate training accuracy
        _, metrics, _, _, _, _, _,_,_ = utils.cal_acc_full(best_classifier, X_train, y_train, num_classes>2)
        fold_train[index,:] = np.array(metrics)
        # Calculate test accuracy
        _, metrics, _, _, _, fpr_test, tpr_test, precision_test, recall_test = utils.cal_acc_full(best_classifier, X_val, y_val, num_classes>2)
        fold_test[index,:] = np.array(metrics)

        roc_values['fpr_test'].append(fpr_test)
        roc_values['tpr_test'].append(tpr_test)
        roc_values['auc_test'].append(metrics[5])

        aucpr_values['recall_test'].append(recall_test)
        aucpr_values['precision_test'].append(precision_test)
        aucpr_values['aucpr_test'].append(metrics[-1])

    # Show the results from the nested cross-validation
    evaluation_name = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'F1', 'AUC', 'AUCPR','MCC','TP','TN','FP','FN']
    colnames = ['{}_{}'.format(set_name, eval_name) for set_name in ['Test', 'Train'] for eval_name in evaluation_name]
    show_value_list = []
    
    print('Test')
    show_value_list = utils.show_metrics(fold_test, show_value_list)  
    print('Training')
    show_value_list = utils.show_metrics(fold_train, show_value_list)
    # Summarize the evaluation measurements into a pd.Series.
    eval_series = pd.Series(show_value_list, index=colnames)

    return fold_classifiers, eval_series, roc_values, aucpr_values
    
def generate_mean_metrics(tprs_list, fprs_list, aucs_list, type):
    mean_fpr = np.linspace(0, 1, 100)
    interp_tprs_list = []
    if type == 'auc':
        for i in range(len(aucs_list)):
            interp_tpr = np.interp(mean_fpr, fprs_list[i], tprs_list[i])
            interp_tpr[0] = 0.0
            interp_tprs_list.append(interp_tpr)
        mean_tpr = np.mean(interp_tprs_list, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
    if type == 'aucpr':
        for i in range(len(aucs_list)):
            interp_tpr = np.interp(mean_fpr, fprs_list[i], tprs_list[i])
            interp_tpr[0] = 1.0
            interp_tprs_list.append(interp_tpr)
        mean_tpr = np.mean(interp_tprs_list, axis=0)
        mean_tpr[-1] = 0.0
        mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
    
    return mean_fpr, mean_tpr, mean_auc
#============= Experiment Configurations============================
# Step Up Parameters
out_root = './cv_results'
unique_id = 'Sep_13th_new_rule_cv_1.2_MI_onlyGFN'
random_state = 1234
model_set = ['GFN']
debug =  False # Set to True if only want to debug the code.
#============= Experiment Configurations============================

exp_save_path = os.path.join(out_root, unique_id)

if not os.path.isdir(out_root):
    os.mkdir(out_root)
if not os.path.isdir(exp_save_path):
    os.mkdir(exp_save_path)

if debug:
    # The number of folds in the outer cross-validation.
    n_folds = 1
    # The number of search interations in the hyper-parameter tuning.
    search_iters = 1 
    # The number of folds in the inner cross-validation.
    n_folds_hyper_tuning = 2
    # Maximal training steps for the proposed GFN model.
    max_steps = 100
else:
    n_folds = 5
    search_iters = 100 #300
    n_folds_hyper_tuning = 3
    max_steps = 8000

print('######################################')
print('Experiment ID:', unique_id)
print('######################################')

dataset = load_dataset_onevisit.load_NSF_dataset(missing_value_tolerance = 10,fill_missing=True)
m_path = './cv_results/Nov_8th_new_rule_2_cv_1.2_MI/saved_model_GFN.mdl'
loaded_models = pickle.load(open(m_path, 'rb'))

row_name_list = []
row_list = []

auc_dict = {}
aucpr_dict= {}

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
color_dict = mcolors.TABLEAU_COLORS
color_items = list(color_dict.items())
color_items.pop(3)

for model_name in model_set:
    print('*************** model is : {}***************'.format(model_name))
    data = np.array(dataset['variables'])
    labels = np.array(dataset['response'])
    split_method = dataset['split_method'] if 'split_method' in dataset else 'sample_wise'
    category_info = dataset['category_info']
    num_classes = dataset['num_classes']
    rule_data = [{'Relation': [[52, 0], [30, 0]], 'Out_weight': 1}, 
                 {'Relation': [[33, 0], [52, 0], [30, 0]], 'Out_weight': 1}, {'Relation': [[52, 0], [53, 3], [53, 4]], 'Out_weight': 1},
                 {'Relation': [[30, 0], [49, 2]], 'Out_weight': 1}, {'Relation': [[30, 0], [37, 0]], 'Out_weight': 1}]
    feature_names = dataset.get('feature_names')

    # Split the dataset into training and indepedent test set if needed.
    ss_train_test = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
    X_train, y_train, X_test, y_test = utils.split_dataset(ss_train_test, data, labels, split_method, index=0)
    X_train = utils.fill_in_missing_value_MI(X_train)
    X_test = utils.fill_in_missing_value_MI(X_test)
    
    # X_train = data
    # y_train = labels

    # Split the data for nested cross-validation on the training set
    fold_classifiers, eval_series, roc_values,aucpr_values = nested_cross_validation(X_train, y_train, num_classes, category_info, feature_names, rule_data, model_name, 
                    n_folds, n_folds_hyper_tuning, search_iters, max_steps, 
                    split_method, random_state)

    # Save the models
    # pickle.dump(fold_classifiers, open(os.path.join(exp_save_path, f'saved_model_{model_name}.mdl'), 'wb'))
    auc_dict[model_name] = roc_values
    aucpr_dict[model_name] = aucpr_values
    row_name_list.append(model_name)
    row_list.append(eval_series)
eval_table = pd.concat(row_list, axis=1).transpose()
eval_table.index = row_name_list
eval_table.to_csv(os.path.join(exp_save_path, 'eval_table_{}.csv'.format(random_state)))
np.save(os.path.join(exp_save_path, 'auc_dict_{}.npy'.format(random_state)), auc_dict)
np.save(os.path.join(exp_save_path, 'aucpr_dict_{}.npy'.format(random_state)), aucpr_dict)

fig, ax = plt.subplots(figsize=(6,4))
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
for i, key in enumerate(auc_dict):
    model_name = key
    roc_values = auc_dict[key]

    mean_fpr, mean_tpr, mean_auc = generate_mean_metrics(roc_values['tpr_test'], roc_values['fpr_test'], roc_values['auc_test'],type = 'auc')
    ax.plot(mean_fpr, mean_tpr, color=color_items[i][-1],
            label=model_name,
            lw=2, alpha=.8)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title=f"ROC curve")
    ax.legend(loc="lower right")
plt.savefig(os.path.join(exp_save_path, 'ROC_all_{}.png'.format(random_state)), dpi=300)

output_name = 'test_aucpr.png'
fig, ax = plt.subplots(figsize=(6,4))
ax.plot([0, 1], [1, 0], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
for i, key in enumerate(auc_dict):
    model_name = key
    aucpr_values = aucpr_dict[key]


    mean_recall, mean_precision, _ = generate_mean_metrics(aucpr_values['recall_test'], aucpr_values['precision_test'], aucpr_values['aucpr_test'],type = 'aucpr')
    ax.plot(mean_recall, mean_precision, color=color_items[i][-1],
            label=model_name,
            lw=2, alpha=.8)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title=f"AUPRC curve")
    ax.legend(loc="lower right")
plt.savefig(os.path.join(exp_save_path, 'AUCPR_all_{}.png'.format(random_state)), dpi=300)

