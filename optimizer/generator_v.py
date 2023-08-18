from scipy.stats import special_ortho_group
import torch
import math
import numpy as np
import random


class GeneratorMask():
    """
    Generateor for sparse mask.
    """

    def __init__(self, mask_size, compression_rate, device):
        self.mask_size = mask_size
        self.n_elem = np.prod(mask_size)
        self.device = device
        self.comp_rate = compression_rate

        # this implementation is critical to generate the mask fast.
        self.origin_mask = torch.zeros(self.mask_size, device=self.device)
        
    def initialize(self, seed):
        print("No initialization")
        pass

    def change_comp_rate(self, comp_rate):
        self.comp_rate = comp_rate
        
    def get_v(self):
        return self.origin_mask.uniform_().ge(1-self.comp_rate).float()
    
"""
class GeneratorMask():

    def __init__(self, mask_size, compression_rate):
        self.mask_size = mask_size
        self.comp_rate = compression_rate
        self.generator = None

    def initialize(self, seed):
        if self.generator is not None:
            print("Warning: Generator is already initialized, check code!")
        self.generator = np.random.RandomState(seed)

        
    def change_comp_rate(self, comp_rate):
        self.comp_rate = comp_rate
        
    def get_v(self):
        #mask = torch.tensor(self.generator.choice([True, False], size=self.mask_size, p=[comp_rate, 1-comp_rate])).float()
        return torch.tensor(self.generator.choice([True, False], size=self.mask_size, p=[self.comp_rate, 1-self.comp_rate])).float()
"""

