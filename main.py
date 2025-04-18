from CNN import Classifier
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,random_split


class Original_dataset:
    class_names=['notumor','pituitary','meningioma','glioma']

    def __init__(self,batch_size,train ='train',transform = None):
        
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        
        self.transform = transform 
        self.dataset = []
        self.images = []
        self.labels = []
        if train=='train' or train=='val':
            dataset = datasets.ImageFolder(root='Training', transform=transform)
            train_len = int(0.8*len(dataset))
            val_len = len(dataset)-train_len

            training_dataset, validation_dataset = random_split(dataset,[train_len,val_len])

            if train=='train':
                self.dataset = training_dataset
            elif train=='val':
                self.dataset = validation_dataset
        
        elif train=='test':
            self.dataset = datasets.ImageFolder(root='Testing', transform=transform)
        
        shuffle = (train == 'train')
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        self.images = [img for img, _ in self.dataset]
        self.labels = [label for _, label in self.dataset]
        self.transform = transform
        self.n_images = len(self.labels)

    def get_size(self):
        return self.n_images
    
    def get_loader(self):
        return self.loader
    

class Modified_dataset:
    pass

def main():

    original_training = Original_dataset(batch_size=32,train='train')
    original_train_loader = original_training.get_loader()
    print("Training size - ",original_training.get_size())

    original_validation = Original_dataset(batch_size=32,train='val') 
    original_val_loader= original_validation.get_loader()
    print("Validation size - ",original_validation.get_size())

    original_testing=Original_dataset(batch_size=32,train='test')
    original_test_loader= original_testing.get_loader()
    print("Testing size - ",original_testing.get_size())


    model = Classifier()

if __name__=='__main__':
    main()