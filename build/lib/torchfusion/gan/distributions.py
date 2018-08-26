from torch.utils.data import Dataset
import torch
import torch.distributions as distibutions

""" A Dataset containing Normal/Gaussian vectors of the specified dimension
    length: The total number of vectors
    size: the size of each vector
    mean: mean of the normal distribution
    std: standard deviation of the normal distribution
"""
class NormalDistribution(Dataset):
    def __init__(self,length,size,mean=0,std=1):
        super(NormalDistribution,self)

        self.size = size
        self.length = length
        self.mean = mean
        self.std = std

    def __getitem__(self, index):
        return torch.randn(self.size).normal_(self.mean,self.std)

    def __len__(self):
        return self.length




