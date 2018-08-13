import cv2
import numpy as np
from skimage import filters
from skimage.filters import threshold_local
from helper_functions import print_image_bw
class ImagePreprocessing(object):
    def __init__(self):
        pass

    def binarize_image(self, image, gaussian_constant):
        image = filters.gaussian(image, gaussian_constant)

        # Locally adaptive threshold
        adaptive_threshold = threshold_local(image, block_size=21, offset=0.02)

        # Return a binary array
        # 0 (WHITE): image >= adaptive_threshold
        # 1 (BLACK): image < adaptive_threshold
        image = np.array(image < adaptive_threshold) * 1
        return image

    def load_image(self,fpath,gaussian_constant):
        image = cv2.imread(fpath)
        print_image_bw(np.array(image),5,5)
        #image = cv2.resize(image, (1000, 1000))
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(str(e))
        image = np.array(image)
        return self.binarize_image(image,gaussian_constant)
