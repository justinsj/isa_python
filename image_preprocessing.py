
class ImagePreprocessing(object):
    def binarize_image(image,a):
        image = filters.gaussian(image,a)
        
        # Locally adaptive threshold
        adaptive_threshold = threshold_local(image, block_size=21, offset=0.02)
        
        # Return a binary array
        # 0 (WHITE): image >= adaptive_threshold
        # 1 (BLACK): image < adaptive_threshold
        image = np.array(image < adaptive_threshold) * 1
        return image
