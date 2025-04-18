from CNN import Classifier
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,random_split
from preprocessing import Preprocessing
from skimage import io,exposure,filters
from PIL import Image
from skimage.util import img_as_ubyte


class Original_dataset:
    class_names=['notumor','pituitary','meningioma','glioma']

    def __init__(self,batch_size,train ='train',modified=False,transform = None):
        
        REFERENCE_FILENAME = r"Training\notumor\Tr-no_0538.jpg"
        refr_img = img_as_ubyte(io.imread(REFERENCE_FILENAME,as_gray=True))
        preprocessing = Preprocessing(refr_img) 

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
            self.images = [Image.fromarray(preprocessing.hist_matching(io.imread(path, as_gray=True))) for path, _ in dataset.samples]
        else:
            self.images = [img for img, _ in self.dataset]
        self.labels = [label for _, label in self.dataset]
        self.transform = transform
        self.n_images = len(self.labels)

    def get_size(self):
        return self.n_images
    
    def get_loader(self):
        return self.loader
    


def main():
    '''============== original dataset =================='''
    original_training = Original_dataset(batch_size=32,train='train')
    original_train_loader = original_training.get_loader()
    print("Training size - ",original_training.get_size())

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

    modified_validation = Original_dataset(batch_size=32,train='val',modified=True) 
    modified_val_loader= modified_validation.get_loader()
    print(" modified Validation size - ",modified_validation.get_size())

    modified_testing=Original_dataset(batch_size=32,train='test',modified=True)
    modified_test_loader= modified_testing.get_loader()
    print(" modified Testing size - ",modified_testing.get_size())


    model = Classifier()

if __name__=='__main__':
    main()