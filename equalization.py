import numpy as np 
from skimage import io,exposure,filters
from skimage.util import img_as_ubyte
import os, glob,pathlib
import matplotlib.pyplot as plt

def show_result( refr, images):
    
    plt.figure(figsize=(10, 12)) 
    images = [
        (refr, 'Original Image'),
        (laplacian, '1'),
        (refr, 'Original Image'),
        (gaussian, '2'),
        (refr,'original'),
        (b_part, '3'),
        (refr, 'Original Image'),
        (Q3, '4'),
        (refr, 'Original Image'),
        (Q4, '5')
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
    matched = exposure.equalize_hist(matched,mask=matched<255)
    print(exposure.histogram(matched))

    return matched

def main():
    pass

def test():
    REFERENCE_FILENAME = r"Training\notumor\Tr-no_0538.jpg"
    refr_img = img_as_ubyte(io.imread(REFERENCE_FILENAME,as_gray=True))

    images=[]
    for number in np.random.randint(1594,size=(5)):
        image = img_as_ubyte(io.imread(f"Training\notumor\Tr-no_{number}.jpg",as_gray=True))
        matched = hist_matching(image, refr_img)
        images.append(matched)

    
    show_result(refr_img,images)
    

if __name__=="__main__":
    test()