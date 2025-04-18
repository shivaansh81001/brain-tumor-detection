from CNN import Classifier
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

def main():
    model = Classifier