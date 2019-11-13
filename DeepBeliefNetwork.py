'''
Deep Belief Network
'''


import torch
import numpy as np
from RestrictedBoltzmannMachine import RBM


class DBN(torch.nn.Module):
    
    
    
    def __init__(self, num_hidden_layers, num_visible, num_hidden, num_output, num_gibbs_samplings, learning_rate = 1e-3):
        super(DBN,self).__init__()
        self.desc = "DBN"
        
        
        self.num_hidden_layers = num_hidden_layers
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.num_gibbs_samplings = num_gibbs_samplings
        self.learning_rate = learning_rate
        
        self.weights = []
        self.RBMS = []
        
        
    def __desc__(self):
        return self.desc
        
    
    def train(self, training_set, num_epochs):
        
        new_training_set = training_set.copy()
        
        for i in range(self.num_hidden_layers):
            curr_rbm = None
            if i == 0:
                
                print("Current RMB numeber", i)
                curr_rbm = RBM(self.num_visible, self.num_hidden[i], self.num_gibbs_samplings, self.learning_rate)
                
                curr_rbm.contrastive_divergence(new_training_set, num_epochs)
                
                self.weights.append(curr_rbm.return_weights)  
                self.RBMS.append(curr_rbm) 
                
                #after training a new RBM, we want to take its output as the new input for the next layer
                for k in range(len(new_training_set)):
                    #visible_vector = new_training_set[k]
                    #need to convert current training set to a ndarray
                    visible_vector = np.asarray(new_training_set[k]).reshape((len(new_training_set[k]), 1))
                    _, sampled_hidden_vector = curr_rbm.sample_hidden(visible_vector)
                        
                    new_training_set[k] = list(sampled_hidden_vector)
                
            else:
                print("Current RMB numeber", i)
                curr_rbm = RBM(self.num_hidden[i-1], self.num_hidden[i], self.num_gibbs_samplings, self.learning_rate)
                
                #the training set is a list of input vectors
                #we need to compute another list of vectors for additional hidden layers
                curr_rbm.contrastive_divergence(new_training_set, num_epochs)
                
                self.weights.append(curr_rbm.return_weights)  
                self.RBMS.append(curr_rbm) 
                
                #after training a new RBM, we want to take its output as the new input for the next layer
                for k in range(len(new_training_set)):
                    #visible_vector = new_training_set[k]
                    #need to convert current training set to a ndarray
                    visible_vector = np.asarray(new_training_set[k]).reshape((len(new_training_set[k]), 1))
                    _, sampled_hidden_vector = curr_rbm.sample_hidden(visible_vector)
                        
                    new_training_set[k] = sampled_hidden_vector
                    
                    
            #after training, create new weight matrix representing
            #the connection between the last RBM and the output layer
            last_weights = np.random.rand((curr_rbm.num_hidden, self.num_output)) * 0.1
            self.weights.append(last_weights)
            
    def backpropagation(self, training_set, training_labels, num_epochs):
        '''
        The training set consists of an array like object of vectors
        with each vector representing a feature vectors of a phone, word or sentence
        Each of thoes feature vectors is associated with a training label
        A training label is an array like object of vectors, with each element
        being an encopding of a single phone. For example, the phone 'a' is represented as (0, 0, 0, 0, 0, 0)
        the phone 'b' as (0,0,0,0,0,1) etc. 
        '''
        
        '''
        We will try with only backpropagating the last layer of the network
        '''
        
        for epoch in range(num_epcohs):
            for k in range(len(training_set)):
                curr_elem = training_set[k]
                curr_label = training_labels[k]
                curr_label_len = len(training_labels[k])
                #now, we divide the training_set[k] into curr_label_len parts
                #and feed each separate part into the network 
                
                for i in range(0, curr_label_len):
                    curr_training_elem = training_set[0:curr_label_len]
                    
                    
                    '''
                    Bring back the previous code here
                    '''
                    
                    
    
                
                
            