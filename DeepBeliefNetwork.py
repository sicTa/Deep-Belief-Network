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
                    
    
                
                
            