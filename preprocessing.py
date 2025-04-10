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

def apply_median(image):
    return filters.median(image)

def hist_matching(image,refr):
    matched = np.clip(exposure.match_histograms(image,refr),0,255).astype(np.uint8)
    #matched = apply_median(matched)
    matched[matched < 20] = 0
    matched = exposure.equalize_hist(matched,mask=matched<255)
    #print(exposure.histogram(matched))

    return matched

def main():
    pass

def test():
    REFERENCE_FILENAME = r"Training\notumor\Tr-no_0538.jpg"
    refr_img = img_as_ubyte(io.imread(REFERENCE_FILENAME,as_gray=True))

    for i in range(5):
        before_images=[]
        after_images=[]
        random_list = np.random.randint(10,1338,size=(5))
        print(random_list)

        for number in random_list:
            image = img_as_ubyte(io.imread(rf"Training/notumor/Tr-no_{number:04}.jpg",as_gray=True))
            image[image< 40] = 0
            before_images.append(image)
            matched = hist_matching(image, refr_img)
            after_images.append(matched)

        
        show_result(before_images,after_images)
    

if __name__=="__main__":
    test()