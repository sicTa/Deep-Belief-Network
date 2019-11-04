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
        
        
        
        
    
    def train(self, training_data, num_epochs):
        
        
        for i in range(num_hidden_layers):
            curr_rbm = Null
            if i == 0:
                curr_rbm = RBM(self.num_visible, self.num_hidden[i], self.num_gibbs_samplings, self.learning_rate)
            else:
                curr_rbm = RBM(self.num_hidden[i-1], self.num_hidden[i], self.num_gibbs_samplings, self.learning_rate)
            