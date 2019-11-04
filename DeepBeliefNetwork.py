'''
Deep Belief Network
'''


import torch
import numpy
import RestrictedBoltzmannMachine.py


class DBN(torch.nn.Module):
    super(DBN,self).__init__()
    self.desc = "DBN"
    
    
    def __init__(self):
        