
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
