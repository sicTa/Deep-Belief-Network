'''
Deep Belief Network
'''


import torch
import numpy
import RestrictedBoltzmannMachine.py


class DBN(torch.nn.Module):
    super(DBN,self).__init__()
    self.desc = "DBN"
    
    
    def __init__(self, num_hidden_layers, num_visible, num_hidden, num_gibbs_samplings, learning_rate = 1e-3):
        self.num_hidden_layers = num_hidden_layers
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_gibbs_samplings = num_gibbs_samplings
        self.learning_rate = learning_rate
        
        self.weights = []
        self.RBMS = []
        
        
        
        
    
    def train(self, training_set, num_epochs):
        
        new_training_set = training_set.copy()
        
        for i in range(num_hidden_layers):
            curr_rbm = Null
            if i == 0:
                curr_rbm = RBM(self.num_visible, self.num_hidden[i], self.num_gibbs_samplings, self.learning_rate)
                
                curr_rbm.train(new_training_set, num_epochs)
                
                self.weights.append(curr_rbm.return_weights)  
                self.RBMS.append(curr_rbm) 
                
                #after training a new RBM, we want to take its output as the new input for the next layer
                for k in range(len(new_training_set)):
                    visible_vector = new_training_set[k]
                    _, sampled_hidden_vector = curr_rbm.sample_hidden(visible_vector)
                        
                    new_training_set[k] = sampled_hidden_vector
                
            else:
                curr_rbm = RBM(self.num_hidden[i-1], self.num_hidden[i], self.num_gibbs_samplings, self.learning_rate)
                
                #the training set is a list of input vectors
                #we need to compute another list of vectors for additional hidden layers
                curr_rbm.train(new_training_set, num_epochs)
                
                self.weights.append(curr_rbm.return_weights)  
                self.RBMS.append(curr_rbm) 
                
                #after training a new RBM, we want to take its output as the new input for the next layer
                for k in range(len(new_training_set)):
                    visible_vector = new_training_set[k]
                    _, sampled_hidden_vector = curr_rbm.sample_hidden(visible_vector)
                        
                    new_training_set[k] = sampled_hidden_vector
                
                
            
            