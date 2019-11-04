#Restricted Boltzmann Machine
import torch
import numpy as np


class RBM(torch.nn.Module):
    '''
    A class containing the model of a single restricted Boltzmann machine.
    
    '''
    
    def __init__(self, num_visible, num_hidden, num_gibbs_samplings = 1, learning_rate=1e-3):
        super(RBM,self).__init__()
        self.desc = "RBM"
          
        self.num_visible = num_visible                                #number of visible nodes
        self.num_hidden = num_hidden                                  #number of hidden nodes
        self.num_gibbs_samplings = num_gibbs_samplings                #number of Gibbs samplings
        self.learning_rate = learning_rate

        self.weights = torch.randn(num_visible, num_hidden) * 0.1     #initialize weight to random value
        self.visible_bias = torch.ones(num_visible) * 0.5             #initialize the visible bias to 0.5
        self.hidden_bias = torch.zeros(num_hidden)                    #initialize hidden bias to 0

    def sample_hidden(self, visible_probabilities):
        '''
        The function return an array of probabilites sampled at the hidden layer. 
        For more information refer to paper "Deep Belief Networks for phone recognition"
        page 2.
        '''
        hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.hidden_bias
        hidden_probabilities = self._sigmoid(hidden_activations)

        hidden_probabilities_numpy = hidden_probabilities.numpy()
        h1_sample_numpy = np.binomial(size=hidden_probabilities_numpy.shape,   # discrete: binomial
                                       n=1,
                                       p=hidden_probabilities_numpy)

        hidden_sample = tensor.from_numpy(h1_sample_numpy)
        
        return hidden_probabilities, hidden_sample
    
    def hidden_activation(self, visible_probabilities):
        '''
        Returns the activation of hidden variables
        '''
        hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.hidden_bias
        return hidden_activations

    def sample_visible(self, hidden_probabilities):
        '''
        The function return an array of probabilites sampled at the visible layer. 
        For more information refer to paper "Deep Belief Networks for phone recognition"
        page 2.
        '''
        visible_activations = torch.matmul(hidden_probabilities, self.weights.t()) + self.visible_bias
        visible_probabilities = self._sigmoid(visible_activations)

        visible_probabilities_numpy = visible_probabilities.numpy()
        v1_sample_numpy  = np.binomial(size=visible_probabilities_numpy.shape,   # discrete: binomial
                                       n=1,
                                       p=visible_probabilities_numpy)

        visible_sample = tensor.from_numpy(v1_sample_numpy)
        
        return visible_probabilities, visible_sample
    
    def visible_activation(self, hidden_probabilities):
        '''
        Returns the activation of visible variables
        '''
        visible_activations = torch.matmul(hidden_probabilities, self.weights.t()) + self.visible_bias
        return visible_activations

    def propup(self, v):
        '''
        For a visible vector v return another vector of probabilities.
        For more information refer to paper "Deep Belief Networks for phone recognition"
        page 2.

        This is, in fact, the result the RBM sees at the hidden layer when a signal
        is coming from the visible layer.
        '''

        propup_pre_sigma = torch.matmul(v, self.weights.t()) + self.hidden_bias
        return self._sigmoid(propup_pre_sigma)


    def propdown(self, h):
        '''
        For a hidden vector h return another vector of probabilities.
        For more information refer to paper "Deep Belief Networks for phone recognition"
        page 2.

        This is, in fact, the result the RBM sees at the visible layer when a signal
        is coming from the hidden layer.
        '''

        propdown_pre_sigma = torch.matmul(h, self.weights.t()) + self.visible_bias
        return self._sigmoid(propdown_pre_sigma)

    def reconstruct(self, v):
        '''
        For a given input vector v, we raturn a vector
        reconstructed by the network. 
        '''

        hidden_res = self.propup(v)
        reconstructed_v = self.propdown(hidden_res)
        return reconstructed_v

    def gibbs_hvh(h_sample):
        '''
        This method reconstruct one gibbs iteration. A signal is
        propagated from the hidden layer, the result is memorized at the visible
        layer and then propagated back to the hidden layer. 
        '''

        v1_prob, v1_sample = self.sample_visible(h_sample)
        h1_prob, h1_sample = self.sample_hidden(v1_sample)

        return [v1_prob, v1_sample,
                h1_prob, h1_sample]

    def gibbs_vhv(v_sample):
        '''
        This method reconstruct one gibbs iteration. A signal is
        propagated from the visible layer, the result is memorized at the hidden
        layer and then propagated back to the visible layer. 
        '''

        h1_prob, h1_sample = self.sample_hidden(v_sample)
        v1_prob, v1_sample = self.sample_visible(h1_sample)

        return [v1_prob, v1_sample,
                h1_prob, h1_sample]

    def free_energy(self, v_sample, W):
        num = len(v_sample)
        Wv = np.clip(torch.matmul(v_sample,W) + self.hidden_bias,-80,80)    #we restrict the result to the range -80, 80
        hidden = np.log(1+np.exp(Wv)).sum(1)
        vbias = torch.matmul(v_sample, self.visible_bias).view(len(hidden))
        return -hidden.view(num)-vbias.view(num)


    def contrastive_divergence(self, input_vector, training_data):
        '''
        An implementation of Contrastive Divergence algorithm with
        k Gibbs' samplings (CD-k)
        '''
        
        weight_matrix = torch.zeros(num_visible, num_hidden)
        visible_bias_vector = torch.zeros(num_visible)
        hidden_bias_vector = torch.zeros(num_hidden)
        
        for input_vector in training_data:
            #ph_mean, ph_sample = self.sample_hidden(input_vector)
            
            chain_start = input_vector
    
    
            #this computers a gibbs sampling of my input vector
            for step in range(self.num_gibbs_samplings):
                if step == 0:
                    nv_means, nv_samples,\
                    nh_means, nh_samples = self.gibbs_vhv(chain_start)
                else:
                    nv_means, nv_samples,\
                    nh_means, nh_samples = self.gibbs_vhv(nh_samples)
    
    
            
            v_tilda = nv_samples
            h_tilda, _ = self.sample_visible(v_tilda)
            
            v_t = v_sample
            h_t, _ = self.sample_visible(v_t)
            
            #we now convert to numpy array, computer outer vector
            #and then return to pytorch tensor
            v_tilda_numpy = v_tilda.numpy()
            h_tilda_numpy = h_tilda.numpy()
            v_t_numpy = v_t.numpy()
            h_t_numpy = h_t.numpy()
            
            help_matrix_numpy = np.outer(h_t_numpy, v_t_numpy) - np.outer(h_tilda_numpy, v_tilda_numpy)
            help_matrix = torch.from_numpy(help_matrix_numpy)
            
            weight_matrix += help_matrix
            visible_bias_vector += (v_t - v_tilda)
            hidden_bias_vector += (h_t - h_tilda)
        
  
        #finding the mean value
        weight_matrix = weight_matrix/len(training_data)
        hidden_bias_vector = hidden_bias_vector/len(training_data)
        visible_bias_vector = visible_bias_vector/len(training_data)
        
        #adjusting the weights and biases
        self.weights = self.weights + self.learning_rate * help_matrix
        
        self.hidden_bias = self.hidden_bias + self.learning_rate * hidden_bias_vector
        
        self.visible_bias = self.visible_bias + self.learning_rate * visible_bias_vector
        
        
    def train(self, training_set, num_epochs):
        for i in range(num_epochs):
                self.contrastive_divergence(training_set)
        
        
        
    def _sigmoid(self, x):
        '''
        Standard definition of a sigmoid function
        '''
        return 1 / (1 + torch.exp(-x))

    def _random_probabilities(self, num):
        '''
        Returns a torch array of random probabilities
        '''
        random_probabilities = torch.rand(num)
        return random_probabilities























