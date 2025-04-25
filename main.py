from CNN import Classifier
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,random_split
from preprocessing import Preprocessing
from skimage import io,exposure,filters
from PIL import Image
from skimage.util import img_as_ubyte

np.random.seed(42)

def show_image_samples(original,modified):
    fig, axs = plt.subplots(2, 5, figsize=(15,6))
    for i in range(5):
        axs[0, i].imshow(original[i], cmap='gray')
        axs[0, i].set_title(f"Original {i+1}")
        axs[0, i].axis('off')
        axs[1, i].imshow(modified[i], cmap='gray')
        axs[1, i].set_title(f"Modified {i+1}")
        axs[1, i].axis('off')
    plt.tight_layout()
    plt.show()

class Original_dataset:
    class_names=['notumor','pituitary','meningioma','glioma']

    def __init__(self,batch_size,train ='train',modified=False,transform = None):
        
        REFERENCE_FILENAME = r"Training\notumor\Tr-no_0538.jpg"
        refr_img = img_as_ubyte(io.imread(REFERENCE_FILENAME,as_gray=True))
        preprocessing = Preprocessing(refr_img) 

        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        
        self.transform = transform 
        self.dataset = []
        self.images = []
        self.labels = []
        
        dataset = datasets.ImageFolder(root='Training' if train in ['train', 'val'] else 'Testing', transform=transform)
        if train=='train' or train=='val':
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
        if modified:
            self.images =[self.transform(Image.fromarray(preprocessing.hist_matching(io.imread(path, as_gray=True)))) for path, _ in dataset.samples]
        else:
            self.images = [img for img, _ in self.dataset]
        self.labels = [label for _, label in self.dataset]
        self.transform = transform
        self.n_images = len(self.labels)

    def get_size(self):
        return self.n_images
    
    def get_loader(self):
        return self.loader

    def get_five_random(self):
        samples = random.sample(self.images, 5)
        return [img.permute(1, 2, 0).numpy() for img in samples]



def main():
    '''============== original dataset =================='''
    original_training = Original_dataset(batch_size=32,train='train')
    original_train_loader = original_training.get_loader()
    print(type(original_train_loader))
    print("Training size - ",original_training.get_size())
    sample_original= original_training.get_five_random()

    original_validation = Original_dataset(batch_size=32,train='val') 
    original_val_loader= original_validation.get_loader()
    print("Validation size - ",original_validation.get_size())

    original_testing=Original_dataset(batch_size=32,train='test')
    original_test_loader= original_testing.get_loader()
    print("Testing size - ",original_testing.get_size())


    '''============== modified dataset =================='''
    modified_training = Original_dataset(batch_size=32,train='train',modified=True)
    modified_train_loader = modified_training.get_loader()
    print(" modified Training size - ",modified_training.get_size())
    sample_modified = modified_training.get_five_random()

    modified_validation = Original_dataset(batch_size=32,train='val',modified=True) 
    modified_val_loader= modified_validation.get_loader()
    print(" modified Validation size - ",modified_validation.get_size())

    modified_testing=Original_dataset(batch_size=32,train='test',modified=True)
    modified_test_loader= modified_testing.get_loader()
    print(" modified Testing size - ",modified_testing.get_size())

    '''============== print sample images =================='''
    show_image_samples(sample_original,sample_modified)


    model = Classifier()

if __name__=='__main__':
    main()