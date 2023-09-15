import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
import numpy as np

dtype = torch.float32

    
def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    labels =y[labels]
    return labels

def reject_outliers(data, m=2):
    return data[abs(data - torch.mean(data)) < m * torch.std(data)]


class Net(nn.Module):
    """
    Build the network.
    """
    def __init__(self, n_rules, n_classes, category_info, binary_pos_only, epsilon_training):
        """ Initialize the network.

        Parameters
        ----------
        n_rules : Int. The number of rules
        n_classes : Int. The number of classes of the target classification task
        category_info : np.ndarray of Int with a shape of (self.n_variables,). Each entry indicates that whether the corresponding 
            variable is categorical. If the variable is continuous, then the corresponding entry is 0; 
            otherwise the corresponding entry gives the number of categorical levels.
        binary_pos_only : Boolean. Whether to only build rules for positive class in a binary classification task
        epsilon_training : Boolean. Whether to make epsilon trainable.

        Returns
        -------
        None.

        """
        super(Net, self).__init__()
        self.n_rules = n_rules
        self.n_classes = n_classes
        self.category_info = category_info.astype(np.int8)
        self.binary_pos_only = binary_pos_only
        self.epsilon_training = epsilon_training
        
        # The number of concepts to encode individual continuous variables
        # In this study, the number of concepts is 3: low, medium, high
        self.n_concepts = 3 
        # Feature encoding layer
        self.layer1 = InputLayer(self.category_info)
        # Rule Layer 
        total_rules = n_rules
        self.layer2 = RuleLayer(self.n_concepts, total_rules, self.category_info, epsilon_training=epsilon_training)
        # Output Layer
        if binary_pos_only: 
            # This should only be used in a binary classification task.
            self.layer3 = InferenceLayerPos(n_rules, n_classes, epsilon_training=epsilon_training)
        else:
            self.layer3 = InferenceLayer(n_rules, n_classes, bias=False, epsilon_training=epsilon_training)
        
        
    def forward(self, x, epsilon):
        """ Forward pass.
        
        x: torch.Tensor. input features with the shape of (batch size, the number of features)
        epsilon: float, range: (0, 1).
        """
        # Encoding layer
        x_continuous, x_category_list = self.layer1(x, epsilon)
        # Rule layer
        x2, connection_mask, attention_masks_continuous, attention_masks_categorical, variable_contrib = self.layer2(
                                                                x_continuous, x_category_list, epsilon)
        # Inference layer
        x3, rule_contrib = self.layer3(x2, epsilon)
        return x3, connection_mask, attention_masks_continuous, attention_masks_categorical, variable_contrib, rule_contrib
    
    
    def _initiate_weights_with_given_rules(self, attention_continuous, attention_categorical, 
                                           connection, weight, init_data, init_value=0.5):
        """
        Initialize the network with collected existing rules.

        Parameters
        ----------
        attention_continuous : torch.Tensor with a shape of (n_continuous_variables, n_concepts, n_rules). 
                Attention matrix for continuous variables in the rule layer.
        attention_masks_categorical : torch.Tensor with a shape of (n_categorical_variables, n_rules). 
                Attention matrix for categorical variables  in the rule layer.
        connection : torch.Tensor with a shape of (n_variables, n_rules). Connection matrix in the rule layer.
        weight : torch.Tensor with a shape of (n_rules, n_classes). Weight matrix in the inference layer.
        init_data : A list of existing rules. The following is an example of rules:
                [{'Relation': [[0,0], [3,2], [5,0]],
                                'Out_weight': 1},
                 {'Relation': [[2,2], [4,2], [5,0]],
                                'Out_weight': 1}, ]
                Each element in the list is one rule. The above example has two existing rules.
                For the first rule, from the value of the 'Relation': [[0,0], [3,2], [5,0]], we can conclude the rule is:
                    if x0 is low AND x3 is high AND x5 is low, then the data sample is positive. 
                
                TODO: The current design and implementation of init_data structure can only be used in binary classification
                task and when binary_pos_only is True. 
                    
        init_value : Float, optional. Initial value for the corresponding entries in the attention and connect matrix.
                The default is 1.

        Returns
        -------
        Updated attention, connection, and weigth matrics.

        """
        
        n_continuous_variables = np.sum(self.category_info==0)
        category_levels = self.category_info[self.category_info>0]
        
        # Build the delta vector from category_info
        delta = []
        d_continuous = 0
        d_categorical = 0
        for i in range(len(self.category_info)):
            if self.category_info[i] == 0:
                d_categorical += 1
                delta.append(d_continuous)
            else:
                d_continuous += 1
                delta.append(d_categorical)
              
        for rule_index, rule in enumerate(init_data):
            temp_attention_continuous = -torch.ones_like(attention_continuous[:,:,0])
            temp_attention_categorical = -torch.ones_like(attention_categorical[:,0])
            temp_connection = -torch.ones_like(connection[:,0])
            
            for concept in rule['Relation']:
                variable_index = concept[0] - delta[concept[0]]
                concept_index = concept[1]
        
                if self.category_info[concept[0]] == 0:
                    temp_attention_continuous[variable_index, concept_index] = init_value
                    temp_connection[variable_index] = init_value
                else:
                    temp_attention_categorical[np.sum(
                        category_levels[:variable_index])+concept_index] = init_value
                    temp_connection[n_continuous_variables+variable_index] = init_value
            attention_continuous[:,:,rule_index] = temp_attention_continuous
            attention_categorical[:,rule_index] = temp_attention_categorical
            connection[:,rule_index] = temp_connection
            
            # Output layer
            if self.binary_pos_only:
                weight[rule_index] = rule['Out_weight']
            else:
                if rule['Out_weight'] > 0:
                    weight[rule_index, 1] = rule['Out_weight']
                else:
                    weight[rule_index, 0] = rule['Out_weight']
        return attention_continuous, attention_categorical, connection, weight
    
    
    def reset_parameters(self, train_features, train_y, init_data=None):
        # Take the use of training samples to initialize parameters in the Input layer
        features_continuous = train_features[:, self.category_info==0]
        
        # m1_list = []
        # m2_list = []
        # s1_list = []
        # s2_list = []

        # for i in range(features_continuous.shape[1]):
        #     tmp = features_continuous[:,i]
        #     tmp_pos = reject_outliers(tmp[train_y == 1])
        #     tmp_neg = reject_outliers(tmp[train_y == 0])
        #     mean_ls = torch.stack([torch.mean(tmp_pos),torch.mean(tmp_neg)],axis = 0)
        #     std_ls = torch.stack([torch.std(tmp_pos),torch.std(tmp_neg)],axis = 0)

        #     mean_ls, indices = torch.sort(mean_ls)
        #     std_ls = std_ls[indices]

        #     m1_list.append(mean_ls[0])
        #     m2_list.append(mean_ls[1])
        #     s1_list.append(std_ls[0])
        #     s2_list.append(std_ls[1])
        
        # m1_list = torch.FloatTensor(m1_list)
        # m2_list = torch.FloatTensor(m2_list)
        # s1_list = torch.FloatTensor(s1_list)
        # s2_list = torch.FloatTensor(s2_list)
            
        
        
        
        
        # m_list = []
        # s_list = []
        # for i in range(features_continuous.shape[1]):
        #     tmp = reject_outliers(features_continuous[:,i])
        #     m_list.append(torch.mean(tmp))
        #     s_list.append(2*torch.std(tmp))
        # m_list = torch.FloatTensor(m_list)
        # s_list = torch.FloatTensor(s_list)
        
        
        m_list = torch.mean(features_continuous, dim=0) 
        s_list = torch.std(features_continuous, dim=0)
        
        self.layer1.reset_parameters(m_list, s_list)
        # self.layer1.reset_parameters(m1_list,s1_list,m2_list,s2_list)
        self.layer2.reset_parameters()
        self.layer3.reset_parameters()
        
        if init_data is not None:
            attention_continuous = self.layer2.attention_continuous.data
            attention_categorical = self.layer2.attention_categorical.data
            connection = self.layer2.connection.data
            weight = self.layer3.weight.data
            
            attention_continuous, attention_categorical, connection, out = self._initiate_weights_with_given_rules(
                attention_continuous, attention_categorical, connection, weight, init_data, init_value=0.5)
            
            self.layer2.attention_continuous.data = attention_continuous
            self.layer2.attention_categorical.data = attention_categorical
            self.layer2.connection.data = connection
            self.layer3.weight.data = out
        
        
class InputLayer(nn.Module):
    """
    Encode the input variables into concepts.
    For continuous variables, they will be encoded using fuzzified concetps.
    For categorical variables, they will be encoded by one-hoting coding.
    """
    def __init__(self, category_info):
        """
        Parameters
        ----------
        n_category_list (int): 1D tensor with length of num_variables. The number of categories if the 
            variable is categorical or zero.
        """
        super(InputLayer, self).__init__()
        self.category_info = category_info
        self.n_continuous_variables = np.sum(category_info==0)
        self.n_categorical_variables = np.sum(category_info>0)
        # Initiate parameters
        # The first dimension of self.weight is 4.
        # For each concepts, the membership function is defined by 4 trainable parameters.
        self.weight = Parameter(torch.Tensor(4, self.n_continuous_variables))
        
    
    def f(self, value):
        epsilon = self.epsilon
        # find indices where value >0 and <0
        i1 = value>=0
        i2 = value<0
        
        out = torch.zeros_like(value)
        out[i1] = value[i1] + epsilon*torch.log(torch.exp(-value[i1]/epsilon)+1)
        out[i2] = epsilon*torch.log(1+torch.exp(value[i2]/epsilon))
        return out

    
    def forward(self, variables, epsilon):
        """
        Parameters
        ----------
        variables: torch.Tensor. the original feature matrix with a shape of (batch_size, n_variables)
        
        Returns
        -------
        x_continuous: torch.Tensor. Encoded continuous variables in a shape of (batch_size, n_continuous_variables, n_concepts), 
                where n_concepts=3.
        x_categorical_list:  A list of encoded categorical variables. 
        """
        input_continuous = variables[:, self.category_info==0]
        input_categorical = variables[:, self.category_info>0]
        
        category_levels = self.category_info[self.category_info>0]

        self.epsilon = epsilon
        # Calculate a1, a2, b1, b2 lists, which define membership functions for individual continuous variables
        self.a2_list = self.weight[0,:] + 0.01 + self.weight[1,:].pow(2)
        self.b1_list = self.weight[0,:] - 0.01 - self.weight[1,:].pow(2)
        self.a1_list = self.a2_list - 0.01 - self.weight[2,:].pow(2)
        self.b2_list = self.b1_list + 0.01 + self.weight[3,:].pow(2)


        
        batch_size = variables.shape[0]
        a1_batch = torch.unsqueeze(self.a1_list, 0).repeat(batch_size, 1)
        a2_batch = torch.unsqueeze(self.a2_list, 0).repeat(batch_size, 1)
        b1_batch = torch.unsqueeze(self.b1_list, 0).repeat(batch_size, 1)
        b2_batch = torch.unsqueeze(self.b2_list, 0).repeat(batch_size, 1)


        # Calculate membership values of indiviudal continuous variables to low, medium, high concepts, respectively
        lx =  self.f((a2_batch-input_continuous)/(a2_batch-a1_batch)) - self.f((a1_batch-input_continuous)/(a2_batch-a1_batch))
        mx = self.f((input_continuous-a1_batch)/(a2_batch-a1_batch)) - self.f((input_continuous-a2_batch)/(a2_batch-a1_batch)) \
            + self.f((b2_batch-input_continuous)/(b2_batch-b1_batch)) - self.f((b1_batch-input_continuous)/(b2_batch-b1_batch)) - 1
        hx = self.f((input_continuous-b1_batch)/(b2_batch-b1_batch)) - self.f((input_continuous-b2_batch)/(b2_batch-b1_batch))
        
        x_continuous = torch.stack([lx, mx, hx], axis=-1)
        x_continuous = F.relu(x_continuous)
        x_categorical_list = []

        # Categorical variables are encoded using regular one-hot embedding.
        for i in range(input_categorical.shape[1]):
            x = input_categorical[:,i]
            x = x.type(torch.long)
            out = one_hot_embedding(x, int(category_levels[i]))
            out = out.type(dtype)
            x_categorical_list.append(out)
        return x_continuous, x_categorical_list
            
        
    # def reset_parameters(self, m1_list, s1_list,m2_list, s2_list):
    #     # Initialize the parameters of membership function using the mean and standard deviation of the training data. 
    #     # weight = torch.stack([m_list, torch.sqrt(s_list), torch.sqrt(s_list), torch.sqrt(s_list)], dim=0)
    #     weight = torch.stack([m1_list, torch.sqrt(s1_list), m2_list, torch.sqrt(s2_list)], dim=0)
    #     self.weight.data = weight

    def reset_parameters(self, m_list, s_list):
        # Initialize the parameters of membership function using the mean and standard deviation of the training data. 
        weight = torch.stack([m_list, torch.sqrt(s_list), torch.sqrt(s_list), torch.sqrt(s_list)], dim=0)
        self.weight.data = weight
        
        
class RuleLayer(nn.Module):
    """
    Calculate rules.
    """
    def __init__(self, n_concepts, n_rules, category_info, epsilon_training=False):
        super(RuleLayer, self).__init__()
        self.n_concepts = n_concepts
        self.n_variables = len(category_info)
        self.n_continuous_variables = np.sum(category_info==0)
        self.n_categorical_variables = np.sum(category_info>0)
        self.category_levels = category_info[category_info>0]
        self.n_rules = n_rules
        self.epsilon_training = epsilon_training
        
        # Initiate parameters
        self.connection = Parameter(torch.Tensor(self.n_variables, n_rules))
        self.attention_continuous = Parameter(torch.Tensor(self.n_continuous_variables, n_concepts, n_rules))
        self.attention_categorical = Parameter(torch.Tensor(int(np.sum(self.category_levels)), n_rules))
        
        if self.epsilon_training:
            self.epsilon = Parameter(torch.Tensor([0.1]))
    
    
    def forward(self, x_continuous, x_categorical_list, epsilon):
        """
        

        Parameters
        ----------
        x_continuous : torch.Tensor. Encoded continuous variables in a shape of (n_concepts, n_variables), where n_concepts=3,
            which is generated from the encoding layer.
        x_categorical_list : List. A list of encoded categorical variables from the encoding layer.
        epsilon : Float. 

        Returns
        -------
        out : torch.Tensor with a shape of (batch_size, n_rules). Firing strength of individual rules.
        connection_mask : torch.Tensor with a shape of (n_variables, n_rules). Connection matrix.
        attention_masks_continuous : torch.Tensor with a shape of (n_continuous_variables, n_concepts, n_rules). 
                Attention matrix for continuous variables.
        attention_masks_categorical : torch.Tensor with a shape of (n_categorical_variables, n_rules). 
                Attention matrix for categorical variables.
        varible_contrib : torch.Tensor with a shape of (batch_size, n_variables, n_rules). Firing strength of 
                individual variables to individual rules. This information will be used for the final rule
                clustering and summarization.

        """
        n_samples = x_continuous.shape[0]
        if self.epsilon_training:
            epsilon = max(0.1, 1/(self.epsilon.pow(2)+1)) #.pow(0.5)
        connection_mask = (torch.tanh(self.connection) + 1)/2
        attention_masks_continuous = (torch.tanh(self.attention_continuous) + 1)/2
        attention_masks_categorical = (torch.tanh(self.attention_categorical) + 1)/2
        
        out = []
        
        x_continuous_stack = torch.unsqueeze(x_continuous, -1).repeat(1, 1, 1, self.n_rules)
        amask_batch = torch.unsqueeze(attention_masks_continuous, 0).repeat(n_samples, 1, 1, 1)

        hidden = torch.mul(x_continuous_stack, amask_batch)
        hidden = torch.sum(hidden, dim=-2)
            
        out_category = []
        for i in range(self.n_rules):
            # For category variables
            if self.n_categorical_variables>0:
                hidden_category= []
                category_mask_list = torch.split(attention_masks_categorical, list(self.category_levels))
                for j in range(self.n_categorical_variables):
                    hidden_category.append(torch.matmul(x_categorical_list[j], category_mask_list[j][:, i]))
                hidden_category = torch.stack(hidden_category, dim=1)
                out_category.append(hidden_category)
        
        if len(out_category)>0:
            hidden = torch.cat([hidden, torch.stack(out_category, axis=-1)], axis=1)
            
        variable_contrib = hidden
        hidden = 1-F.relu(1-hidden) + 0.1**5
        connection_batch = torch.unsqueeze(connection_mask, 0).repeat(n_samples, 1, 1)
        
        temp = hidden.pow(connection_batch*(epsilon-1)/epsilon)   
        temp = torch.sum(temp, dim=1) - (self.n_variables-1)
        out = temp.pow(epsilon/(epsilon-1))
        
        if torch.sum(torch.isnan(out))>0 or torch.sum(torch.isinf(out))>0:
            print('rule layer error')
        return out, connection_mask, attention_masks_continuous, attention_masks_categorical, variable_contrib
    
    
    def reset_parameters(self):       
        # value1 = 0
        # value2 = 0
        # nn.init.uniform_(self.connection, a=value2-0.01, b=value2)
        # nn.init.uniform_(self.attention_continuous, a=value1-0.001, b=value1)
        # nn.init.uniform_(self.attention_categorical, a=value1-0.001, b=value1)

        value1 = 0
        value2 = 0
        nn.init.uniform_(self.connection, a=value2-1, b=value2)
        nn.init.uniform_(self.attention_continuous, a=value1-1, b=value1)
        nn.init.uniform_(self.attention_categorical, a=value1-1, b=value1)
        
        if self.epsilon_training:
            self.epsilon.data = torch.Tensor([0.1])
        
    
class InferenceLayer(nn.Module):
    """
    Calculate rules.
    """
    def __init__(self, n_rules, n_classes, bias=True, epsilon_training=False):
        super(InferenceLayer, self).__init__()
        self.n_rules = n_rules
        self.n_classes = n_classes
        self.epsilon_training = epsilon_training
        self.add_bias = bias
        
        # Initiate parameters
        # Whether to use trainable bias or fixed bias
        if self.add_bias:
            self.bias = Parameter(torch.Tensor(n_classes))
        if self.epsilon_training:
            self.epsilon = Parameter(torch.Tensor([0.1]))
        
        self.weight = Parameter(torch.Tensor(n_rules*n_classes, n_classes))
        
        
    def forward(self, x, epsilon):
        """
        Calculate the firing strength of individual rules. 

        Parameters
        ----------
        x : torch.Tensor with a shape of (batch_size, n_rules). Firing strength of individual rules from the rule layer.
        epsilon : Float.

        Returns
        -------
        out : torch.Tensor with a shape of (batch_size, n_classes).

        """
        n_samples = x.shape[0]
        if self.epsilon_training:
            epsilon = max(0.1, 1/(self.epsilon.pow(2)+1))

        out_list = []
        for i in range(self.n_classes):
            input_part = x[:,self.n_rules*i:self.n_rules*(i+1)]
            weight_part = self.weight[self.n_rules*i:self.n_rules*(i+1), 0].pow(2)
            out = torch.mul(input_part, torch.stack([weight_part]*n_samples, axis=0)) + 0.1**5 
            temp = torch.sum(out.pow(1/epsilon), axis=-1) + 0.1**5
            temp = temp.pow(epsilon)
            out_list.append(temp)        
            
        if self.add_bias:
            out = torch.stack(out_list, dim=-1) + self.bias
        else:
            out = torch.stack(out_list, dim=-1)

        if torch.sum(torch.isnan(out))>0 or torch.sum(torch.isinf(out))>0:
            print('inference layer error')
            
        return out
    
    def reset_parameters(self):
        nn.init.uniform_(self.weight, a=1, b=1.001)
        if self.add_bias:
            nn.init.uniform_(self.bias, a=-0.1, b=-0.1)
        if self.epsilon_training:
            self.epsilon.data = torch.Tensor([0.1])
        

class InferenceLayerPos(nn.Module):
    """
    Calculate rules.
    With InferenceLayerPos, all rules encoded in the network will only contribute the positive class.
    This will only be used in a binary classification task.
    """
    def __init__(self, n_rules, n_classes, epsilon_training=False):
        super(InferenceLayerPos, self).__init__()
        self.n_rules = n_rules
        self.n_classes = n_classes
        self.epsilon_training = epsilon_training
        
        # Initiate parameters
        self.weight = Parameter(torch.Tensor(n_rules))
        # Use a fixed bias
        self.bias = -2 #-1, -2
        if self.epsilon_training:
            self.epsilon = Parameter(torch.Tensor([0.1]))

    def forward(self, x, epsilon):
        """
        Calculate the firing strength of individual rules. 

        Parameters
        ----------
        x : torch.Tensor with a shape of (batch_size, n_rules). Firing strength of individual rules from the rule layer.
        epsilon : Float.

        Returns
        -------
        out : torch.Tensor with a shape of (batch_size, 2).

        """
        n_samples = x.shape[0]
        if self.epsilon_training:
            epsilon = max(0.1, 1/(self.epsilon.pow(2)+1)) #.pow(0.5)
            
        out = torch.mul(x, torch.stack([self.weight.pow(2).pow(0.5)]*n_samples, axis=0))  
        rule_contrib = out
        temp = torch.sum(out.pow(1/epsilon), axis=-1)

        out = torch.stack([torch.zeros(temp.shape), temp + self.bias], dim=-1)
        return out, rule_contrib
    
    def reset_parameters(self):
        nn.init.uniform_(self.weight, a=0.1, b=0.1) #1
        if self.epsilon_training:
            self.epsilon.data = torch.Tensor([1])


class InferenceLayerMultiLabel(nn.Module):
    """
    Calculate rules.
    With InferenceLayerPos, all rules encoded in the network will only contribute the positive class.
    This will only be used in a binary classification task.
    """
    def __init__(self, n_rules, n_classes, epsilon_training=False):
        super(InferenceLayerMultiLabel, self).__init__()
        self.n_rules = n_rules
        self.n_classes = n_classes
        self.epsilon_training = epsilon_training
        self.add_bias = bias
        
        # Initiate parameters
        # Whether to use trainable bias or fixed bias
        if self.add_bias:
            self.bias = Parameter(torch.Tensor(n_classes))
        if self.epsilon_training:
            self.epsilon = Parameter(torch.Tensor([0.1]))
        
        self.weight = Parameter(torch.Tensor(n_rules*n_classes, n_classes))

    def forward(self, x, epsilon):
        """
        Calculate the firing strength of individual rules. 

        Parameters
        ----------
        x : torch.Tensor with a shape of (batch_size, n_rules). Firing strength of individual rules from the rule layer.
        epsilon : Float.

        Returns
        -------
        out : torch.Tensor with a shape of (batch_size, 2).

        """
        n_samples = x.shape[0]
        if self.epsilon_training:
            epsilon = max(0.1, 1/(self.epsilon.pow(2)+1))

        out_list = []
        rule_contrib = []
        for i in range(self.n_classes):
            weight_part = self.weight[self.n_rules*i:self.n_rules*(i+1), 0].pow(2)
            out = torch.mul(x, torch.stack([weight_part]*n_samples, axis=0)) + 0.1**5 
            rule_contrib.append(out)
            temp = torch.sum(out.pow(1/epsilon), axis=-1) + 0.1**5
            temp = temp.pow(epsilon)
            out_list.append(temp)        
            
        if self.add_bias:
            out = torch.stack(out_list, dim=-1) + self.bias
        else:
            out = torch.stack(out_list, dim=-1)

        if torch.sum(torch.isnan(out))>0 or torch.sum(torch.isinf(out))>0:
            print('inference layer error')

        out = F.sigmoid(out)
        rule_contrib = torch.stack(rule_contrib, dim=-1)
            
        return out,rule_contrib
    
    def reset_parameters(self):
        nn.init.uniform_(self.weight, a=1, b=1.001)
        if self.add_bias:
            nn.init.uniform_(self.bias, a=-0.1, b=-0.1)
        if self.epsilon_training:
            self.epsilon.data = torch.Tensor([0.1])


