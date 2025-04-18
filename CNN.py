import torch
import torch.nn as nn
from torch.utils.data import Dataset
from preprocessing import Preprocessing as pre

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        pass

    def forward(self):
        pass

    def init_weights(self):
        #https://github.com/Xiaoming-Yu/RAGAN/blob/master/models/model.py#L50
        #https://stackoverflow.com/questions/61848635/how-to-decide-which-mode-to-use-for-kaiming-normal-initialization
        for layer in self.modules():
            if isinstance(layer,nn.Conv2d):    #weights for conv layer
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(layer,nn.Linear):  #Linear / fc layers weights init
                nn.init.xavier_normal_(layer.weight)
            elif isinstance(layer,(nn.BatchNorm2d, nn.BatchNorm1d)): #this is for normalization layers- (fc and conv) - normal distribution
                nn.init.constant_(layer.weight,1)
                nn.init.constant_(layer.bias,0)


class Optimization():
    def __init__(self,type,lr,momentum,eps,weight_decay):
        self.type =type
        self.lr = lr
        self.momentum=momentum
        self.eps =eps
        self.weight_decay = weight_decay 

    def backpass(self):
        pass


    