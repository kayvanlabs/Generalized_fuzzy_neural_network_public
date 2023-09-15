import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import copy
import torch.utils.data
import torch.nn.functional as F
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn import preprocessing
from sklearn.utils.multiclass import unique_labels
import utils
import network
import sys


dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_dataset_loader(features, labels=None, batch_size=1, 
                         scaler=None, infinite=False, category_list=None):
    """
    Standalize featuers and build data loader for model training and evaluation.

    """
    
    if scaler is not None:
        scaled_features = utils.standardize(features, scaler, category_list)
    else:
        scaled_features = features    

    if labels is not None:
        tensor_features = torch.from_numpy(scaled_features)
        tensor_labels = torch.from_numpy(labels.astype(np.int32))
        dataset = torch.utils.data.TensorDataset(tensor_features.to(device), tensor_labels.to(device))
    else:
        tensor_features = torch.from_numpy(scaled_features)
        dataset = torch.utils.data.TensorDataset(tensor_features.to(device))
    
    if infinite:
        data_loader = utils.repeater(torch.utils.data.DataLoader(dataset, int(batch_size), shuffle=True))
    else:
        data_loader = torch.utils.data.DataLoader(dataset, int(batch_size), shuffle=False)
        
    return scaled_features, data_loader



class GeneralizedFuzzyClassifier(BaseEstimator, ClassifierMixin):
    """Generalizaed Fuzzy Network.
    Consists of encoding layer, rule layer, inference layer.

    Parameters
    ----------
    weighted_loss: list. A list of weights for each class.
    
    epsilon_training: boolean. Whether to make the epsilon trainable or not.
    
    init_rule_index: None or str or list. None: do not use existing knowledge to initiate part of network. Str: the index of the ground
        truth rule that will be used to initiate the network. For example, '0_1' means the first two rules will be used (0-based). 
        List: customized rule that will be used. For example: [{'Relation': [[1, 0], [1, 2], [2, 1], [2, 2]], 'Out_weight': 1}]
        
    binary_pos_only: boolean. Whether to only build rules for positive class in a binary classification task
    
    n_rules: int. The number of hidden neurons.
    
    n_classes: int. The number of classses
    
    report_freq: Int. Calculate the performance on the training and validation dataset every {report_freq} steps.
    
    patientce_step: int. If the val acc doesn't improved for {patience_step} step, the training will be early stopped.
    
    max_steps: int. Maximal training steps.
    
    learning_rate: float. Learning rate
    
    batch_size: int. Batch size
    
    epsilon_decay: int. Epsilon decay steps. After epsilon_decay steps, the epsilon will reduce by 0.1
    
    min_epsilon: float. The minimal epsilon value.
    
    sparse_regu: float. Magnitude of the sparse regularization term.
    
    corr_regu: float. Magnitude of the correlation regularization term.
    
    category_info: np.ndarray of Int with a shape of (self.n_variables,). Each entry indicates that whether the corresponding 
        variable is categorical. If the variable is continous, then the entry is 0; otherwise the entry gives the number of categorical levels. 
        
    split_method: str. The method to split the dataset into training and validation sets. 'sample_wise' or 'patient_wise' (Not implemented yet).
        
    random_state: int. Random seed.

    verbose: int. Output level during the training. 0: No output; 1: print the best step;  2: print the training process.
        
    val_ratio: The ratio of data samples in the training set that are used as validation set.
    
    rule_data: A list of existing rules from domain knowledge.
    """
    def __init__(self, 
                 weighted_loss=None,
                 epsilon_training=False,
                 init_rule_index=None,
                 binary_pos_only=False,
                 n_rules=40,
                 n_classes=2,
                 report_freq=50,
                 patience_step=500,
                 max_steps=10000,
                 learning_rate=0.01,
                 batch_size=150,
                 min_epsilon=0.1,
                 sparse_regu=0,
                 corr_regu=0,
                 category_info=None,
                 split_method='sample_wise',
                 random_state=None,
                 verbose=0,
                 val_ratio=0.2,
                 rule_data=None):
        
        self.weighted_loss = weighted_loss
        self.epsilon_training = epsilon_training
        self.init_rule_index = init_rule_index
        self.binary_pos_only = binary_pos_only
        self.n_rules = n_rules
        self.n_classes = n_classes
        self.report_freq = report_freq
        self.patience_step = patience_step
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.min_epsilon = min_epsilon
        self.sparse_regu = sparse_regu
        self.corr_regu = corr_regu
        self.category_info = category_info
        self.split_method = split_method
        self.random_state = random_state
        self.verbose = verbose
        self.val_ratio = val_ratio
        self.rule_data = rule_data
        
        if self.n_classes>2 and self.binary_pos_only:
            raise ValueError('binary_pos_only should be set to true only in a binary classification task.')
        
        if self.init_rule_index is not None and self.rule_data is None:
            raise ValueError('init_rule_index is given but rule_data is None.')
            
        if self.n_classes>2 and self.rule_data is not None:
            raise ValueError('the current design and implementation of rule_data only support binary classification.')
            
            
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit the model. 

        Parameters
        ----------
        X : np.ndarray. Features in the training set. 
        y : np.ndarray. Labels in the training set. 
        X_val : np.ndarray, optional. Features in the validation set. The default is None. 
            If it is None, the validation set will be randomly sampled from the training set.
        y_val : np.ndarray, optional. Features in the validation set. The default is None. 
            If it is None, the validation set will be randomly sampled from the training set.

        Raises
        ------
        ValueError if the init_rule_index is not in the required format.

        """
        torch.manual_seed(self.random_state)
        
        if X_val is None:
            #If the validation set is not given, the validation set will be randomly sampled from the training set.
            ss_train_val = StratifiedShuffleSplit(n_splits=1, test_size=self.val_ratio, random_state=self.random_state)
            X_train, y_train, X_val, y_val = utils.split_dataset(ss_train_val, X, y, self.split_method, index=0)
        else:
            X_train, y_train = X, y
        
        if self.split_method == 'patient_wise':
            # If the split_method is 'patient_wise', the first column of the features is the patient id
            # The patient id can facilitate the data split, but it should not be used in the following
            # model training and evalutation.
            X_train = X_train[:, 1:]
            X_val = X_val[:, 1:]
        
        if self.category_info is None:
            # If the category information is not given, then we will assume all variables are continuous.
            self.category_info = np.zeros(X_train.shape[1])

        self.scaler = preprocessing.StandardScaler().fit(X_train[:, self.category_info==0])
        
        # Build the training data loader. It will generate training batch in an infinite way. 
        _, train_loader = build_dataset_loader(X_train, y_train, self.batch_size, 
                                            scaler=self.scaler, infinite=True,
                                            category_list=self.category_info)
        
        # Build the data loader that used to evaluate model's performance on the training data, so it is not infinite.
        scaled_train_X, train_loader_for_eval = build_dataset_loader(X_train, y_train, self.batch_size,
                                            scaler=self.scaler, infinite=False,
                                            category_list=self.category_info)
        
        # Build the data loader that used to evaluate model's performance on the validation data, and it is not infinite.
        _, val_loader = build_dataset_loader(X_val, y_val, self.batch_size, 
                                          scaler=self.scaler, infinite=False,
                                          category_list=self.category_info)
        
        # Initialize the model using existing rules. 
        if self.init_rule_index is not None:
            if isinstance(self.init_rule_index, str):
                if self.rule_data is not None:
                    if self.init_rule_index == 'All':
                        init_data = self.rule_data
                    else:
                        init_rule_index = self.init_rule_index.split('_')
                        init_rule_index = [int(x) for x in init_rule_index]
                        # print(init_rule_index)
                        init_data = [self.rule_data[index] for index in init_rule_index]
                else:
                    raise ValueError
            else:
                init_data = self.init_rule_index
        else:
            init_data = None
        
        # Build network
        net = network.Net(self.n_rules, self.n_classes, self.category_info,
                     self.binary_pos_only, self.epsilon_training)
        net.reset_parameters(torch.from_numpy(scaled_train_X), y_train, init_data)
        net.to(device)
        
        self.initial_weights = copy.deepcopy(net.state_dict())
        # Build loss function and optimizer

        if self.weighted_loss is not None:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(self.weighted_loss, dtype=dtype, device=device))
        else:
            criterion = nn.CrossEntropyLoss()


            
        
        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)
        
        
        train_aucpr_list = []
        val_aucpr_list = []
            
        start_epsilon = 0.99
        
        patience = 0
        best_value = 0
        best_step = 0
        best_net = copy.deepcopy(net.state_dict())
        best_epsilon = start_epsilon
        delay_step = -1
        
        for global_step, (inputs, labels) in enumerate(train_loader, 0):
            labels = labels.type(torch.long)
            # Optimize
            optimizer.zero_grad()
            
            # Exponentially reduce the epsilon value
            epsilon = max(start_epsilon*(0.999**((global_step-delay_step)//2)), self.min_epsilon)
            if global_step>100:
                for g in optimizer.param_groups:
                    g['lr'] = self.learning_rate*0.5

        
            out, connection_mask, attention_mask_continous, \
                attention_mask_categorical, _, _ = net(inputs, epsilon) # Forward pass
            
            # Add regularization term
            regu_1, regu_2 = self._regularization_calculation(connection_mask,
                                                        attention_mask_continous,
                                                        attention_mask_categorical)

            cross_entropy =  criterion(out, labels)
            loss = cross_entropy + regu_1 + regu_2
            loss.backward()        
            
            for param in net.parameters():
                if param.grad is not None:
                    param.grad[torch.isnan(param.grad)] = 0
            
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()

            if (global_step+1) % self.report_freq == 0:
                if self.epsilon_training:
                    print(net.layer2.epsilon.detach().numpy())
                    print(net.layer3.epsilon.detach().numpy())
                    
                #print(net.layer3.bias.detach().numpy())
                _, _, _, _, f1_train, auc_train, aucpr_train, _, _ = self._model_testing(net, train_loader_for_eval, epsilon)
                _, _, _, _, f1_val, auc_val, aucpr_val, _, _ = self._model_testing(net, val_loader, epsilon)
                train_aucpr_list.append(aucpr_train)
                val_aucpr_list.append(aucpr_val) 
                
                if epsilon == self.min_epsilon:
                    if f1_val>best_value:
                        best_value = f1_val
                        best_step = global_step
                        best_net = copy.deepcopy(net.state_dict())
                        patience = 0
        
                    else:
                        patience += 1
        
                    if patience > self.patience_step//self.report_freq:
                        break
                
                if self.verbose==2:
                    print(epsilon)
                    print(f'Step {global_step}, train_auc: {auc_train:.3f}, train_aucpr: {aucpr_train:.3f}, train_f1: {f1_train:.3f}, val_auc: {auc_val:.3f}, val_aucpr: {aucpr_val:.3f}, val_f1: {f1_val:.3f}.')
            if global_step > self.max_steps:
                break

        if self.verbose>=1:
            print(f'The best value is {best_value:.2f} at step {best_step} with epsilon {best_epsilon:.3f}.')
        
        net.load_state_dict(best_net)
        self.estimator = net
        
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        self.epsilon = best_epsilon

        return self
    

    def predict(self, X):
        """ Predict the class of the given data samples.
        """
        if self.split_method == 'patient_wise':
            X = X[:, 1:]
            
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        
        _, test_loader = build_dataset_loader(X, batch_size=self.batch_size, scaler=self.scaler,
                                           infinite=False, category_list=self.category_info)
            
        pred_list = []
        variable_contrib_list = []
        rule_contrib_list = []
        for i, inputs in enumerate(test_loader, 0):
            x, _, _, _, variable_contrib, rule_contrib = self.estimator(inputs[0], self.epsilon)
            pred = torch.argmax(x, dim=1)
            pred_list.append(pred)
            variable_contrib_list.append(variable_contrib.detach().numpy())
            rule_contrib_list.append(rule_contrib.detach().numpy())
            
        pred_list = np.concatenate(pred_list, axis=0)
        
        # The following two are the firing strength of individual variables to individual rules
        # and the firing strength of individual rules to individual class. 
        # They are used for rule clustering and summarization.
        self.variable_contrib = np.concatenate(variable_contrib_list, axis=0)
        self.rule_contrib = np.concatenate(rule_contrib_list, axis=0)
        return pred_list
    
    
    def predict_proba(self, X):
        """ Predict the probabilites of belonging to individaul classes of the given data samples.
        """
        if self.split_method == 'patient_wise':
            X = X[:, 1:]
            
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        _, test_loader = build_dataset_loader(X, batch_size=self.batch_size, scaler=self.scaler,
                                           infinite=False, category_list=self.category_info)
            
        prob_list = []
        for i, inputs in enumerate(test_loader, 0):
            x, _, _, _, _, _ = self.estimator(inputs[0], self.epsilon)
            prob = F.softmax(x, dim=1)
            prob_list.append(prob.detach())        
            
        prob_list = np.concatenate(prob_list, axis=0)
        prob_list = np.round(prob_list, 3)
        return prob_list
    
    
    def _regularization_calculation(self, connection_mask, attention_mask_continous, attention_mask_categorical):
        """ Calculate sparse regularization and correlation regularization.
        Sparse regularization is calculated as the magnitude of parameters in the network.
        Correlation regularization is calcualted as the correlation of each pair of rules encoded in the network. 
        """
        category_info = self.category_info.astype(np.int8)
                
        n_continous_variables = np.sum(category_info==0)
        n_category_variables = np.sum(category_info>0)
        
        connection_regu = torch.norm(connection_mask.view(-1), 1)
        attention_regu_continous = torch.norm(attention_mask_continous.view(-1), 1)
         
        if n_category_variables>0:
            attention_regu_categorical = torch.norm(attention_mask_categorical.view(-1), 1)
            attention_regu = attention_regu_continous + attention_regu_categorical
        else:
            attention_regu = attention_regu_continous
        
        # Sparse regularization
        regu_1 = self.sparse_regu * (connection_regu + attention_regu)
        
        # Build the rule matrix
        mat = attention_mask_continous * torch.stack([connection_mask[0:n_continous_variables, :]]*3, axis=1)
        mat = mat.reshape(-1, self.n_rules)
        
        if n_category_variables>0:
            n_category = category_info[category_info>0]
            attention_category_list = torch.split(attention_mask_categorical, list(n_category))
            mat_category_list = []
            for i in range(n_category_variables):
                temp = attention_category_list[i]*torch.stack([connection_mask[n_continous_variables+i]]*n_category[i], axis=0)
                mat_category_list.append(temp)
            
            mat = torch.cat([mat, torch.cat(mat_category_list, dim=0)],  dim=0)
        
        # Correlation regularization
        regu_2 = 0
        for i in range(self.n_rules):
            for j in range(i, self.n_rules):
                regu_2 += torch.sum(mat[:, i]*mat[:, j])/(
                        torch.norm(mat[:, i], 2)*torch.norm(mat[:, j], 2)+0.0001)
        regu_2 = self.corr_regu * regu_2
        return regu_1, regu_2
    
    
    def _model_testing(self, net, test_loader, epsilon):
        """ Model test.
        
        Parameters
        ----------
        net: A Net object. The network with the best validation performance
        test_loader: finite data_loader for evaluation.
        epsilon: A float. The current epsilon value.
        
        Returns
        -------
        Evaluation metrics includeing accuracy, sensitivity, specificity, precision, f1-score, auc, and aucpr.
        """
        pred_list = []
        label_list = []
        prob_list = []
        for i, (inputs, labels) in enumerate(test_loader, 0):
            #inputs, labels = data
            labels = labels.type(torch.long)
            x, _, _, _, _, _ = net(inputs, epsilon)
            
            if torch.sum(torch.isnan(x))>0 or torch.sum(torch.isinf(x))>0:
                import time
                filename = time.strftime("%b_%d_%H_%M_%S", time.localtime())
                params = np.array([self.n_rules, self.batch_size, self.learning_rate, 
                                   self.sparse_regu, self.corr_regu])
                np.save(filename, params)
                                
            prob = F.softmax(x, dim=1)
            pred = torch.argmax(x, dim=1)
            pred_list.append(pred)
            label_list.append(labels)
            prob_list.append(prob.detach())
        
        pred_list = torch.cat(pred_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        prob_list = torch.cat(prob_list, dim=0)
            
        pred = pred_list.numpy()
        labels = label_list.numpy()
        probs = prob_list.numpy()
        probs = np.round(probs, 3)
        
        acc = np.sum(pred == labels)/len(labels)
        sen = np.sum(pred[labels==1])/np.sum(labels)
        spe = np.sum(1-pred[labels==0])/np.sum(1-labels)
        pre = np.sum(pred[labels==1])/(np.sum(pred)+1) # avoid warnings when all samples are classified as negative
        f1 = sklearn.metrics.f1_score(labels, pred)
        
        fpr, tpr, _ = sklearn.metrics.roc_curve(labels+1, probs[:,1], pos_label=2)
        auc = sklearn.metrics.auc(fpr, tpr)
        aucpr = sklearn.metrics.average_precision_score(labels, probs[:,1])
        return acc, sen, spe, pre, f1, auc, aucpr, probs, labels
