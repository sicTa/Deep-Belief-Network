'''
Deep Belief Network
'''


import torch
import numpy as np
from RestrictedBoltzmannMachine import RBM


class DBN(torch.nn.Module):
    
    
    
    def __init__(self, num_layers, layers, num_output):
        super(DBN,self).__init__()
        self.desc = "DBN"
        
        
        self.num_layers = num_layers
        self.layers = layers
        self.num_output = num_output
        
        
        self.weights = []
        self.RBMS = []
        
        
    def __desc__(self):
        return self.desc
        
    
    def train(self, training_set, num_epochs):
        
        new_training_set = training_set.copy()
        
        for i in range(self.num_layers):
            curr_rbm = None
            if i == 0:
                curr_rbm = RBM(self.layers[i], self.layers[i + 1])
                
                curr_rbm.contrastive_divergence(new_training_set, num_epochs)
                
                self.weights.append(curr_rbm.return_weights)  
                self.RBMS.append(curr_rbm) 
                
                
                #after training a new RBM, we want to take its output as the new input for the next layer
                new_training_set = curr_rbm.run_visible_probabilities(new_training_set)
                    
               
                
            else:
                print("Current RMB numeber", i)

                curr_rbm = RBM(self.layers[i], self.layers[i + 1])
                #the training set is a list of input vectors
                #we need to compute another list of vectors for additional hidden layers
                curr_rbm.contrastive_divergence(new_training_set, num_epochs)
                
                self.weights.append(curr_rbm.return_weights)  
                self.RBMS.append(curr_rbm) 
                
                
                #after training a new RBM, we want to take its output as the new input for the next layer
                new_training_set = curr_rbm.run_visible_probabilities(new_training_set)
                    
                    
            ''' #after training, create new weight matrix representing
            #the connection between the last RBM and the output layer
            last_weights = np.random.rand((curr_rbm.num_hidden, self.num_output)) * 0.1
            self.weights.append(last_weights)'''
            
    def propagate_forward(self, input_vector):
        
        first = np.asarray(input_vector).reshape((len(input_vector), 1))
        
        for i in range(self.num_layers):
            if i == 0:
                x = self.RBMS[i].run_visible(first)
            else:
                x = self.RBMS[i].run_visible(x)
            
    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    
    def export_rbms(self):
        for i in range(self.num_layers):
            self.RBMS[i].export_rbm("RBM num " + str(i) + ".txt")
            
    def add_RBM_to_network(self, rbm):
        self.RBMS.append(rbm)
        self.weights.append(rbm.return_weights)
        
            

            