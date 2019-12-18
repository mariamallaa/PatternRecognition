from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import threshold_otsu
from commonfunctions import *

fname='testing.png'
blur_radius = 1.0
threshold = 50

thresh = threshold_otsu(image)
binary = image > thresh

img = io.imread("waww.jpg")
show_images([img])

img = np.asarray(img)
print(img.shape)
# (160, 240)

# smooth the image (to remove small objects)
#imgf = ndimage.gaussian_filter(img, blur_radius)
threshold = 50

# find connected components
labeled, nr_objects = ndimage.label(img)
print("Number of objects is {}".format(nr_objects))
# Number of objects is 4 


