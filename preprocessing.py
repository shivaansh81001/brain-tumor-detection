import numpy as np 
from skimage import io,exposure,filters
from skimage.util import img_as_ubyte
import os, glob,pathlib
import matplotlib.pyplot as plt

def show_result( before, after):
    
    plt.figure(figsize=(10, 12)) 
    images = [
        (before[0], 'Original Image'),
        (after[0], '1'),
        (before[1], 'Original Image'),
        (after[1], '2'),
        (before[2],'original'),
        (after[2], '3'),
        (before[3], 'Original Image'),
        (after[3], '4'),
        (before[4], 'Original Image'),
        (after[4], '5')
    ]

    for i, (img, title) in enumerate(images, start=1):
        plt.subplot(5, 2, i) 
        plt.imshow(img, cmap='gray')
        plt.title(title)

    plt.tight_layout()  
    plt.show()


class Preprocessing():
    def __init__(self,refr):
        self.image = None
        self.refr = refr

    def apply_threshold(self):
        self.image[self.image< 40] = 0

    def apply_median(self):
        return filters.median(self.image)

    def hist_matching(self,image):
        self.image = img_as_ubyte(image)
        self.apply_threshold()
        matched= exposure.match_histograms(self.image, self.refr)
        matched =np.clip(matched, 0, 255)
        matched=matched.astype(np.uint8)
        matched[matched <20] =0
        matched =exposure.equalize_hist(matched, mask=matched < 255)
        matched=img_as_ubyte(matched)

        return matched


def test():
    REFERENCE_FILENAME = r"Training\notumor\Tr-no_0538.jpg"
    refr_img = img_as_ubyte(io.imread(REFERENCE_FILENAME,as_gray=True))

    for i in range(5):
        before_images=[]
        after_images=[]
        random_list = np.random.randint(10,1338,size=(5))
        print(random_list)

        for number in random_list:
            image = img_as_ubyte(io.imread(rf"Training/meningioma/Tr-me_{number:04}.jpg",as_gray=True))
            
            before_images.append(image)
            
            preprocessing = Preprocessing(image, refr_img)
            matched = preprocessing.hist_matching()
            after_images.append(matched)

        
        show_result(before_images,after_images)
    

if __name__=="__main__":
    test()