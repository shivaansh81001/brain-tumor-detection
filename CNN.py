import torch
import torch.nn as nn
from torch.utils.data import Dataset
from preprocessing import Preprocessing as pre

class Classifier():
    def __init__(self):
        pass

    def forward(self):
        pass


class Params():
    def __init__(self):
        pass

    class Optimization():
        def __init__(self):
            self.type ='adam'
            self.lr = 5e-4
            self.momentum=0.9
            self.eps =1e-8
            self.weight_decay = 1e-4 

    