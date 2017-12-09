from sklearn.metrics import confusion_matrix
import itertools
from random import sample
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

from constants import target_names_all, target_names

class ComponentClassifierPredict(object):
    """
    Predict component obtained by segmentation process
    """
    

    def __init__(self, min_percent_match, min_confidence):
        self.min_percent_match = min_percent_match # set to 0.7
        self.min_confidence = min_confidence # set to 0.3
    
    @staticmethod
    def expand_dimension(image, num_channel):
        '''
        Input: (100,100) image /matrix
        Output: (None,100,100,1)
        '''

        expanded_image = image
        if num_channel == 1:
            # Classifier only needs 1 channel
            if K.image_dim_ordering() =='th':
                # Classifier only needs 1 channel
                expanded_image = np.expand_dims(expanded_image, axis=0)
                expanded_image = np.expand_dims(expanded_image, axis=0)
            else:
                expanded_image = np.expand_dims(expanded_image, axis=3) 
                expanded_image = np.expand_dims(expanded_image, axis=0)
        else:
            if K.image_dim_ordering()=='th':
                # Modify data if using theano instead of tensorflow
                expanded_image = np.rollaxis(expanded_image, 2,0)
                expanded_image = np.expand_dims(expanded_image, axis=0)
            else:
                # Expand dimensions as needed in classifier
                expanded_image = np.expand_dims(expanded_image, axis=3)
                expanded_image = np.expand_dims(expanded_image, axis=0)
                
        return expanded_image

    def predict_class(self, image, model):
        if image.shape == (100,100):
            image = self.expand_dimension(image,3)
        
        prediction = model.predict(image)
        first_max = max(prediction[0])
        
        second_max = list(prediction[0])
        second_max.remove(max(second_max))
        second_max = max(second_max)

        # Get first, second, and third maximum percentage matches
        # To be used for entropy calculations
        
        index = (prediction[0]).tolist().index(first_max)
        
        
        
        return index, first_max, second_max 

    def use_entropy(self, index, first_max, second_max):
        """ Prediction for a single image """

        # If prediction is not confident or if confidence, as calculated by the
        # difference top two predictions is too hight, or if another third prediction
        # is close to the second prediction
        # Discard = raction as an 'unknown' class
        if first_max < self.min_percent_match or first_max - second_max < self.min_confidence:
            index = 23 # index 23 is the unknown class
        
        # Otherwise, if prediciton is confident, return the original index
        return index


    def predict_classes(self, ext_images, model):

        if len(ext_images) == 0: return

        # Initialization
        ext_class_index = []
        ext_class_name = []
        ext_match_first_max_percent = []
        ext_match_second_max_percent = []

        for i in range(len(ext_images)):
            image = ext_images[i]
            expanded_image = self.expand_dimension(image, 3)
    
            # Predict object class with entropy theory and record data
            index, first_max, second_max = self.predict_class(expanded_image, model)
            index = self.use_entropy(index, first_max, second_max)

            # Save extractions
            ext_class_index.append(index)
            ext_class_name.append(target_names_all[index])

            # Attach percentages to lists (in range of 0 to 1.0, ex: 91% is recorded as 0.91)
            ext_match_first_max_percent.append(first_max)
            ext_match_second_max_percent.append(second_max)

        return ext_class_index, ext_class_name, ext_match_first_max_percent, ext_match_second_max_percent

    def get_confusion_matrix(self, y_pred_one_hot, y_test_one_hot):
        """ Input one hot """
        y_pred_reverse = np.argmax(y_pred_one_hot, axis=1)
        y_test_reverse = np.argmax(y_test_one_hot, axis=1)

        return confusion_matrix(y_test_reverse, y_pred_reverse)

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        # Configuration
        np.set_printoptions(precision=2)
        plt.figure(figsize=(30, 30))

        # Plot
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def map_index(self, index):
        """ Adjust all class index """
        mapped_index = None

        if index >= 14 and index <= 17:
            mapped_index = 14    
        elif index >= 18 and index <= 21:
            #adjust predictions to merge clockwise moments
            mapped_index = 15
        elif index == 22:
            #adjust index of noise
            mapped_index = 16
        elif index == 23:
            #adjust index of random letters
            mapped_index = 17
        elif index >= 24 and index <= 31:
            #adjust predicitons to merge fixed supports
            mapped_index = 18
        elif index >= 32 and index <= 39:
            #adjust predicitons to merge pinned supports
            mapped_index = 19
        elif (index >= 40 and index <= 41) or (index >= 44 and index<= 45):
            #adjust predictions to merge vertical roller supports
            mapped_index = 20
        elif (index >= 42 and index <= 43) or (index >= 46 and index<= 47):
            #adjust predicitons to merge horizontal roller supports
            mapped_index = 21
        elif index >= 48 and index <= 63:
            #adjust index of last 12 classes
            mapped_index = index - 26

        return mapped_index

    def adjust_predictions(self, ext_class_index):
        adjusted_ext_class_index = []
        adjusted_ext_class_name = []
        for i in range(len(ext_class_index)):
            index = ext_class_index[i]
            mapped_index = self.map_index(index)
            adjusted_ext_class_index.append(mapped_index)
            adjusted_ext_class_name.append(target_names[mapped_index])

        return adjusted_ext_class_index, adjusted_ext_class_name

    def calculate_recall(self):
        pass

    def calculate_precision(self):
        pass

    def calculate_accuracy(self):
        pass

    def calculate_F1_score(self):
        pass
