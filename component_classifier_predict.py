from sklearn.metrics import confusion_matrix
import itertools
from random import sample
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from extraction_preprocessing import ExtractionPreprocessing
from component_classifier_training import ComponentClassifierTraining

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
        if image.shape == (100,100):    
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

    def get_first_and_second_max_and_indices(self, prediction_list):
        
        first_max_percent = max(prediction_list)
        first_max_index = prediction_list.index(first_max_percent)
        
        second_max_list = prediction_list[:]
        second_max_list.remove(max(second_max_list))
        
        second_max_percent = max(second_max_list)
        second_max_index = prediction_list.index(second_max_percent)
        
        return first_max_index, second_max_index, first_max_percent, second_max_percent
    def predict_class(self, image, model_1):#, model_2=None,model_3 = None):

        image = self.expand_dimension(image,3)
        
#        print(model_1.predict(image)[0])
#        print(model_1.predict(image)[0].tolist())
        prediction_1 = model_1.predict(image)[0].tolist()
        first_max_index_1, second_max_index_1, first_max_percent_1, second_max_percent_1 = \
        first_max_index, second_max_index, first_max_percent, second_max_percent = self.get_first_and_second_max_and_indices(prediction_1)
        
#        if model_2 != None and model_3 != None:
#            prediction_2 = model_2.predict(image)
#            first_max_index_2, second_max_index_2, first_max_percent_2, second_max_percent_2 = \
#            get_first_and_second_max_and_indices(prediction_2[0])
#            
#            prediction_3 = model_3.predict(image)
#            first_max_index_3, second_max_index_3, first_max_percent_3, second_max_percent_3 = \
#            get_first_and_second_max_and_indices(prediction_3[0])
#        
#        # Get first, second, and third maximum percentage matches
#        # To be used for entropy calculations
#            return first_max_index_1, second_max_index_1, first_max_percent_1, second_max_percent_1,\
#                first_max_index_2, second_max_index_2, first_max_percent_2, second_max_percent_2,\
#                first_max_index_3, second_max_index_3, first_max_percent_3, second_max_percent_3
        
        return first_max_index_1, second_max_index_1, first_max_percent_1, second_max_percent_1
    
    def predict_class_with_rotations(self,image,model_1, model_2=None, model_3=None):
        min_angle = -10
        max_angle = 10
        step = 1
        list_of_angles = list(np.arange(min_angle,max_angle,step))
        predictions_list = []
        prediction_percentages_list = []
        
        for angle in list_of_angles:
            rotated_image = Image.fromarray(image)
            rotated_image = rotated_image.rotate(angle)
            rotated_image_arr = np.array(rotated_image)
            extraction_obj = ExtractionPreprocessing(rotated_image_arr,'', '')
            extraction_obj.preprocess_extraction(rotated_image_arr, 100,100,100,100, 0.3, 0,0,0,0)
            if model_2 != None and model_3 != None:
                prediction = self.predict_class_3_models(rotated_image_arr,model_1,model_2,model_3)
            else:
                prediction,first_max, second_max = self.predict_class(rotated_image_arr, model_1)
                
            predictions_list.append(prediction)
            prediction_percentages_list.append(first_max)
            
        index = predictions_list[prediction_percentages_list.index(max(prediction_percentages_list))]
        
        return index
            
        # create list of predictions
        
    def select_best_prediction(self,prediction_list,percentage_matches):
        '''
        Get the index of highest percentage match
        Prediction list has list of indices predicted
        Percentage matches has list of percent matches with indices in prediction list
        '''
        index = prediction_list[prediction_list.index(max(percentage_matches))]
        return index
    
    def mode(self,arr):
        dict = {}
        for x in range(0,len(arr)):
            count = 1
            if arr[x] in dict:
                count = count + dict[arr[x]]
            dict[arr[x]] = count
        return max(dict, key=dict.get), arr.count(max(dict, key=dict.get))
    
    def select_most_common_prediction(self, prediction_list):
        total_number_of_models = len(prediction_list)
        needed_number_to_agree = int(np.ceil(total_number_of_models/2))
        most_common_index, count = self.mode(prediction_list)
        index = 23 #random class
        if count >= needed_number_to_agree:
            index = most_common_index
        return index
    
	
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

    def predict_classes_3_models(self, ext_images,PATH, model_1_weights_name, model_2_weights_name, model_3_weights_name,return_all=False):
        if type(ext_images) is list:
            if len(ext_images) == 0: return
        if type(ext_images) is np.ndarray:
            if ext_images.shape[1] == 0: return
        
        # Initialization
        ext_class_name = []
        ext_class_index = []
        ext_class_first_max_index_1 = []
        ext_class_second_max_index_1 = []
        ext_match_first_max_percent_1 = []
        ext_match_second_max_percent_1 = []
        
        ext_class_first_max_index_2 = []
        ext_class_second_max_index_2 = []
        ext_match_first_max_percent_2 = []
        ext_match_second_max_percent_2 = []
        
        ext_class_first_max_index_3 = []
        ext_class_second_max_index_3 = []
        ext_match_first_max_percent_3 = []
        ext_match_second_max_percent_3 = []

        for i in range(len(ext_images)):
            image = ext_images[i]
            if return_all:
                index, index_1,index_2,index_3, first_max_index_1, second_max_index_1, first_max_percent_1, second_max_percent_1,\
                first_max_index_2, second_max_index_2, first_max_percent_2, second_max_percent_2,\
                first_max_index_3, second_max_index_3, first_max_percent_3, second_max_percent_3 = self.predict_class_3_models(self, image, PATH, model_1_weights_name, model_2_weights_name, model_3_weights_name,return_all=return_all)
                
                ext_class_first_max_index_2.append(first_max_index_2)
                ext_class_second_max_index_2.append(second_max_index_2)
                ext_match_first_max_percent_2.append(first_max_percent_2)
                ext_match_second_max_percent_2.append(second_max_percent_2)
        
                ext_class_first_max_index_3.append(first_max_index_3)
                ext_class_second_max_index_3.append(second_max_index_3)
                ext_match_first_max_percent_3.append(first_max_percent_3)
                ext_match_second_max_percent_3.append(second_max_percent_3)
            else:
                index, index_1,index_2,index_3, first_max_index_1, second_max_index_1, first_max_percent_1, second_max_percent_1 = self.predict_class_3_models(self, image, PATH, model_1_weights_name, model_2_weights_name, model_3_weights_name,return_all=return_all)
            # Save extractions
            ext_class_index.append(index)
            ext_class_name.append(target_names_all[index])
            ext_class_first_max_index_1.append(first_max_index_1)
            ext_class_second_max_index_1.append(second_max_index_1)
            ext_match_first_max_percent_1.append(first_max_percent_1)
            ext_match_second_max_percent_1.append(second_max_percent_1)

        if return_all:
            return ext_class_index, ext_class_name, ext_class_first_max_index_1, ext_class_second_max_index_1, ext_match_first_max_percent_1, ext_match_second_max_percent_1,\
                    ext_class_first_max_index_2, ext_class_second_max_index_2, ext_match_first_max_percent_2, ext_match_second_max_percent_2,\
                    ext_class_first_max_index_3, ext_class_second_max_index_3, ext_match_first_max_percent_3, ext_match_second_max_percent_3
        
        return ext_class_index, ext_class_name, ext_class_first_max_index_1, ext_class_second_max_index_1, ext_match_first_max_percent_1, ext_match_second_max_percent_1


        
    def predict_class_preloaded(self, image, PATH, dropout, num_classes, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL, model_1):
        expanded_image = self.expand_dimension(image, 3)
        
        #do prediction
        first_max_index_1, second_max_index_1, first_max_percent_1, second_max_percent_1 = self.predict_class(image, model_1)
        index = self.use_entropy(first_max_index_1, first_max_percent_1, second_max_percent_1)
        
        return index, first_max_index_1, second_max_index_1, first_max_percent_1, second_max_percent_1
        
    def predict_class_3_models(self, image, PATH, dropout, num_classes, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL, model_1_weights_name, model_2_weights_name=None, model_3_weights_name=None,return_all=False):
        expanded_image = self.expand_dimension(image, 3)
        
        #load first weights
        training_obj = ComponentClassifierTraining(64,TRAINING_RATIO_TRAIN,TRAINING_RATIO_VAL)
        training_obj.model = training_obj.load_sketch_a_net_model(dropout, num_classes,(100,100,1))
        training_obj.model.load_weights(PATH+model_1_weights_name+'.h5')
        
        #do prediction
        first_max_index_1, second_max_index_1, first_max_percent_1, second_max_percent_1 = self.predict_class(image, training_obj.model)
        index_1 = self.use_entropy(first_max_index_1, first_max_percent_1, second_max_percent_1)
        
        if model_2_weights_name != None:
            #load second weights
            training_obj = ComponentClassifierTraining(64,TRAINING_RATIO_TRAIN,TRAINING_RATIO_VAL)
            training_obj.model = training_obj.load_sketch_a_net_model(dropout, num_classes,(100,100,1))
            training_obj.model.load_weights(PATH+model_2_weights_name+'.h5')
            
            #do prediction
            first_max_index_2, second_max_index_2, first_max_percent_2, second_max_percent_2 = self.predict_class(image, training_obj.model)
            index_2 = self.use_entropy(first_max_index_2, first_max_percent_2, second_max_percent_2)
        if model_3_weights_name != None:
            #load third weights
            training_obj = ComponentClassifierTraining(64,TRAINING_RATIO_TRAIN,TRAINING_RATIO_VAL)
            training_obj.model = training_obj.load_sketch_a_net_model(dropout, num_classes,(100,100,1))
            training_obj.model.load_weights(PATH+model_3_weights_name+'.h5')
            
            #do prediction
            first_max_index_3, second_max_index_3, first_max_percent_3, second_max_percent_3 = self.predict_class(image, training_obj.model)
            index_3 = self.use_entropy(first_max_index_3, first_max_percent_3, second_max_percent_3)
    
        
        
        if model_2_weights_name != None and model_3_weights_name != None:
            index = self.select_most_common_prediction([index_1,index_2,index_3])
        else:
            index = index_1
            return index, first_max_index_1, second_max_index_1, first_max_percent_1, second_max_percent_1
        
        if return_all:
            return index, index_1,index_2,index_3, first_max_index_1, second_max_index_1, first_max_percent_1, second_max_percent_1,\
                first_max_index_2, second_max_index_2, first_max_percent_2, second_max_percent_2,\
                first_max_index_3, second_max_index_3, first_max_percent_3, second_max_percent_3
        return index, index_1,index_2,index_3, first_max_index_1, second_max_index_1, first_max_percent_1, second_max_percent_1
        
    def predict_classes(self, ext_images, model_1, model_2 = None, model_3 = None):
        
        if type(ext_images) is list:
            if len(ext_images) == 0: return
        if type(ext_images) is np.ndarray:
            if ext_images.shape[1] == 0: return
        
        # Initialization
        ext_class_index = []
        ext_class_name = []
        ext_match_first_max_percent = []
        ext_match_second_max_percent = []

        for i in range(len(ext_images)):
            image = ext_images[i]
            expanded_image = self.expand_dimension(image, 3)
    
            if model_2 == None or model_3 == None:
                # Predict object class with entropy theory and record data
                index, first_max, second_max = self.predict_class(expanded_image, model_1)
                index = self.use_entropy(index, first_max, second_max)
                
                # Attach percentages to lists (in range of 0 to 1.0, ex: 91% is recorded as 0.91)
                ext_match_first_max_percent.append(first_max)
                ext_match_second_max_percent.append(second_max)
            else:
                index_1, first_max_1, second_max_1 = self.predict_class(expanded_image, model_1)
                index_1 = self.use_entropy(index_1, first_max_1, second_max_1)
                
                index_2, first_max_2, second_max_2 = self.predict_class(expanded_image, model_2)
                index_2 = self.use_entropy(index_2, first_max_2, second_max_2)
                
                index_3, first_max_3, second_max_3 = self.predict_class(expanded_image, model_3)
                index_3 = self.use_entropy(index_3, first_max_3, second_max_3)
                
                index, first_max, second_max = self.select_most_common_prediction([index_1,index_2,index_3],[first_max_1,first_max_2,first_max_3],[second_max_1, second_max_2, second_max_3])
                
            # Save extractions
            ext_class_index.append(index)
            ext_class_name.append(target_names_all[index])

        if not(model_2 == None or model_3 == None):
            ext_match_first_max_percent = np.zeros(len(ext_images)).tolist()
            ext_match_second_max_percent = np.zeros(len(ext_images)).tolist()

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
        mapped_index = index

            #adjust predictions to merge clockwise moments
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
