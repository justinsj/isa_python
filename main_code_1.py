# In[1]:

%load_ext autoreload
%autoreload 2
from __future__ import print_function
import numpy as np

from keras.models import Sequential

import random

import cv2
import time

from component_segmentation import ComponentSegmentation
from extraction_preprocessing import ExtractionPreprocessing
from component_classifier_training import ComponentClassifierTraining
from component_classifier_predict import ComponentClassifierPredict
from extraction_labelling import ExtractionLabelling
from testing_class import TestingClass
from helper_functions import print_image_bw
from helper_functions import plot_model_results_and_save

import gc
gc.enable()

print('Done Importing...')



# In[2]:


#selective search parameters
scale_input=200 #200 previous: #10
sigma_input=10 #10 previous: #15
min_size_input=10 #10 previous: #5

#noise reduction parameters
min_shape=10 #min. number of black pixels  
min_height=3 #min. height of bounding box
min_width=3 #min. width of bounding box

buffer_zone=2 #expand bounding box by this amount in all directions  
min_area=100 #min. area of bounding box
min_black=10 #min. number of black pixels
min_black_ratio=0.01 #min ratio of black pixels to the bounding box area

#Overlap parameters
overlap_repeats = 4 #set to 8
overlap_threshold = 0.3 #set to 0.3 (overlap has to be larger than the threshold)

#Removing unconnected pieces parameters
max_piece_percent=0.3  # set to 0.3

#Extractions preprocessing paramaters
img_rows, img_cols = 100,100
wanted_w, wanted_h, export_w, export_h = img_cols, img_rows, img_cols, img_rows

#CNN training parameters
num_classes = 64
TRAINING_RATIO_TRAIN = 0.7
TRAINING_RATIO_VAL = 0.15
dropout = 0

#CNN prediction parameters
min_percent_match = 0.0 # set to 0.45
min_confidence = 0.0 # set to 0.95

#Paths and names


PATH = 'C:/Users/JustinSanJuan/Desktop/Workspace/python/Testing Folder/' #must have "/" at the end

name = 'Sketch-a-Net_64_classes_100x100_0.0_all_100epochs'

base_dataset_name = 'Training_Samples_64_classes_100x100_all'

dataset_PATH = 'C:/Users/JustinSanJuan/Desktop/HKUST/UROP Deep Learning Image-based Structural Analysis/Code/Python/Testing Folder/'
dataset_name = 'Training_Samples_64_classes_100x100_all_cleaned_32898'
new_dataset_name = 'Training_Samples_64_classes_100x100_all_cleaned_32898'

'''

PATH = '/home/chloong/Desktop/Justin San Juan/isa_python/'
name = 'Sketch-a-Net_64_classes_100x100_0.0_all_100epochs'
base_dataset_name = 'Training_Samples_64_classes_100x100_all'
dataset_PATH = '/home/chloong/Desktop/Justin San Juan/Testing Folder/'
dataset_name = 'Training_Samples_64_classes_100x100_all_cleaned_32898'
new_dataset_name = 'Training_Samples_64_classes_100x100_all_cleaned_32898'
'''
print('Done setting hyperparamters...')


# # Load Image
# Image (binary, grayscale, 2D, numpy array) for regions of interest proposals is loaded.

# In[7]:

"""
start = time.time() # Begin time measurement
"""
seed = 1000

model_1_weights_name = 'Sketch-A-Net_all_29739+13301_7_layers'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_1'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_2'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_3'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_4'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_5'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_6'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_7'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_8'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_7_layers'

testing_obj = TestingClass(dataset_PATH, wanted_w, wanted_h, export_w, export_h, max_piece_percent)
prediction_list, ground_truth_list, ext_class_name, ext_class_first_max_index_1, ext_class_second_max_index_1, ext_match_first_max_percent_1, ext_match_second_max_percent_1\
 = testing_obj.test_classifier_multiple_slow(dataset_PATH, [], num_classes,dropout, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL,\
                                             100,seed,350,706,min_percent_match, min_confidence,model_1_weights_name = model_1_weights_name, model_num = -1, exclude = [],save = True)

del testing_obj
del prediction_list
del ground_truth_list
del ext_class_name
del ext_class_first_max_index_1
del ext_class_second_max_index_1
del ext_match_first_max_percent_1
del ext_match_second_max_percent_1
gc.collect()
#####
seed = 1000

#model_1_weights_name = 'Sketch-A-Net_all_29739+13301_7_layers'
model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_1'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_2'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_3'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_4'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_5'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_6'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_7'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_8'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_7_layers'

testing_obj = TestingClass(dataset_PATH, wanted_w, wanted_h, export_w, export_h, max_piece_percent)
prediction_list, ground_truth_list, ext_class_name, ext_class_first_max_index_1, ext_class_second_max_index_1, ext_match_first_max_percent_1, ext_match_second_max_percent_1\
 = testing_obj.test_classifier_multiple_slow(dataset_PATH, [], num_classes,dropout, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL,\
                                             100,seed,350,706,min_percent_match, min_confidence,model_1_weights_name = model_1_weights_name, model_num = 0, exclude = [],save = True)

del testing_obj
del prediction_list
del ground_truth_list
del ext_class_name
del ext_class_first_max_index_1
del ext_class_second_max_index_1
del ext_match_first_max_percent_1
del ext_match_second_max_percent_1
gc.collect()

#####
seed = 1000

#model_1_weights_name = 'Sketch-A-Net_all_29739+13301_7_layers'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)'
model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_1'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_2'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_3'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_4'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_5'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_6'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_7'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_8'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_7_layers'

testing_obj = TestingClass(dataset_PATH, wanted_w, wanted_h, export_w, export_h, max_piece_percent)
prediction_list, ground_truth_list, ext_class_name, ext_class_first_max_index_1, ext_class_second_max_index_1, ext_match_first_max_percent_1, ext_match_second_max_percent_1\
 = testing_obj.test_classifier_multiple_slow(dataset_PATH, [], num_classes,dropout, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL,\
                                             100,seed,350,706,min_percent_match, min_confidence,model_1_weights_name = model_1_weights_name, model_num = 1, exclude = [],save = True)

del testing_obj
del prediction_list
del ground_truth_list
del ext_class_name
del ext_class_first_max_index_1
del ext_class_second_max_index_1
del ext_match_first_max_percent_1
del ext_match_second_max_percent_1
gc.collect()

#####
seed = 1000

#model_1_weights_name = 'Sketch-A-Net_all_29739+13301_7_layers'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_1'
model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_2'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_3'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_4'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_5'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_6'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_7'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_8'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_7_layers'

testing_obj = TestingClass(dataset_PATH, wanted_w, wanted_h, export_w, export_h, max_piece_percent)
prediction_list, ground_truth_list, ext_class_name, ext_class_first_max_index_1, ext_class_second_max_index_1, ext_match_first_max_percent_1, ext_match_second_max_percent_1\
 = testing_obj.test_classifier_multiple_slow(dataset_PATH, [], num_classes,dropout, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL,\
                                             100,seed,350,706,min_percent_match, min_confidence,model_1_weights_name = model_1_weights_name, model_num = 2, exclude = [],save = True)

del testing_obj
del prediction_list
del ground_truth_list
del ext_class_name
del ext_class_first_max_index_1
del ext_class_second_max_index_1
del ext_match_first_max_percent_1
del ext_match_second_max_percent_1
gc.collect()

#####
seed = 1000

#model_1_weights_name = 'Sketch-A-Net_all_29739+13301_7_layers'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_1'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_2'
model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_3'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_4'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_5'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_6'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_7'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_8'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_7_layers'

testing_obj = TestingClass(dataset_PATH, wanted_w, wanted_h, export_w, export_h, max_piece_percent)
prediction_list, ground_truth_list, ext_class_name, ext_class_first_max_index_1, ext_class_second_max_index_1, ext_match_first_max_percent_1, ext_match_second_max_percent_1\
 = testing_obj.test_classifier_multiple_slow(dataset_PATH, [], num_classes,dropout, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL,\
                                             100,seed,350,706,min_percent_match, min_confidence,model_1_weights_name = model_1_weights_name, model_num = 3, exclude = [],save = True)
del testing_obj
del prediction_list
del ground_truth_list
del ext_class_name
del ext_class_first_max_index_1
del ext_class_second_max_index_1
del ext_match_first_max_percent_1
del ext_match_second_max_percent_1
gc.collect()

#####
seed = 1000

#model_1_weights_name = 'Sketch-A-Net_all_29739+13301_7_layers'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_1'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_2'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_3'
model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_4'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_5'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_6'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_7'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_8'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_7_layers'

testing_obj = TestingClass(dataset_PATH, wanted_w, wanted_h, export_w, export_h, max_piece_percent)
prediction_list, ground_truth_list, ext_class_name, ext_class_first_max_index_1, ext_class_second_max_index_1, ext_match_first_max_percent_1, ext_match_second_max_percent_1\
 = testing_obj.test_classifier_multiple_slow(dataset_PATH, [], num_classes,dropout, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL,\
                                             100,seed,350,706,min_percent_match, min_confidence,model_1_weights_name = model_1_weights_name, model_num = 4, exclude = [],save = True)
del testing_obj
del prediction_list
del ground_truth_list
del ext_class_name
del ext_class_first_max_index_1
del ext_class_second_max_index_1
del ext_match_first_max_percent_1
del ext_match_second_max_percent_1
gc.collect()

#####
seed = 1000

#model_1_weights_name = 'Sketch-A-Net_all_29739+13301_7_layers'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_1'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_2'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_3'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_4'
model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_5'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_6'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_7'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_8'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_7_layers'

testing_obj = TestingClass(dataset_PATH, wanted_w, wanted_h, export_w, export_h, max_piece_percent)
prediction_list, ground_truth_list, ext_class_name, ext_class_first_max_index_1, ext_class_second_max_index_1, ext_match_first_max_percent_1, ext_match_second_max_percent_1\
 = testing_obj.test_classifier_multiple_slow(dataset_PATH, [], num_classes,dropout, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL,\
                                             100,seed,350,706,min_percent_match, min_confidence,model_1_weights_name = model_1_weights_name, model_num = 5, exclude = [],save = True)

del testing_obj
del prediction_list
del ground_truth_list
del ext_class_name
del ext_class_first_max_index_1
del ext_class_second_max_index_1
del ext_match_first_max_percent_1
del ext_match_second_max_percent_1
gc.collect()

#####
seed = 1000

#model_1_weights_name = 'Sketch-A-Net_all_29739+13301_7_layers'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_1'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_2'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_3'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_4'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_5'
model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_6'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_7'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_8'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_7_layers'

testing_obj = TestingClass(dataset_PATH, wanted_w, wanted_h, export_w, export_h, max_piece_percent)
prediction_list, ground_truth_list, ext_class_name, ext_class_first_max_index_1, ext_class_second_max_index_1, ext_match_first_max_percent_1, ext_match_second_max_percent_1\
 = testing_obj.test_classifier_multiple_slow(dataset_PATH, [], num_classes,dropout, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL,\
                                             100,seed,350,706,min_percent_match, min_confidence,model_1_weights_name = model_1_weights_name, model_num = 6, exclude = [],save = True)

del testing_obj
del prediction_list
del ground_truth_list
del ext_class_name
del ext_class_first_max_index_1
del ext_class_second_max_index_1
del ext_match_first_max_percent_1
del ext_match_second_max_percent_1
gc.collect()

#####
seed = 1000

#model_1_weights_name = 'Sketch-A-Net_all_29739+13301_7_layers'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_1'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_2'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_3'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_4'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_5'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_6'
model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_7'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_8'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_7_layers'

testing_obj = TestingClass(dataset_PATH, wanted_w, wanted_h, export_w, export_h, max_piece_percent)
prediction_list, ground_truth_list, ext_class_name, ext_class_first_max_index_1, ext_class_second_max_index_1, ext_match_first_max_percent_1, ext_match_second_max_percent_1\
 = testing_obj.test_classifier_multiple_slow(dataset_PATH, [], num_classes,dropout, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL,\
                                             100,seed,350,706,min_percent_match, min_confidence,model_1_weights_name = model_1_weights_name, model_num = 7, exclude = [],save = True)

del testing_obj
del prediction_list
del ground_truth_list
del ext_class_name
del ext_class_first_max_index_1
del ext_class_second_max_index_1
del ext_match_first_max_percent_1
del ext_match_second_max_percent_1
gc.collect()

#####
seed = 1000

#model_1_weights_name = 'Sketch-A-Net_all_29739+13301_7_layers'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_1'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_2'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_3'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_4'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_5'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_6'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_7'
model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_8'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_7_layers'

testing_obj = TestingClass(dataset_PATH, wanted_w, wanted_h, export_w, export_h, max_piece_percent)
prediction_list, ground_truth_list, ext_class_name, ext_class_first_max_index_1, ext_class_second_max_index_1, ext_match_first_max_percent_1, ext_match_second_max_percent_1\
 = testing_obj.test_classifier_multiple_slow(dataset_PATH, [], num_classes,dropout, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL,\
                                             100,seed,350,706,min_percent_match, min_confidence,model_1_weights_name = model_1_weights_name, model_num = 8, exclude = [],save = True)

del testing_obj
del prediction_list
del ground_truth_list
del ext_class_name
del ext_class_first_max_index_1
del ext_class_second_max_index_1
del ext_match_first_max_percent_1
del ext_match_second_max_percent_1
gc.collect()

#####
seed = 1000

#model_1_weights_name = 'Sketch-A-Net_all_29739+13301_7_layers'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_1'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_2'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_3'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_4'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_5'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_6'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_7'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_8'
model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_7_layers'

testing_obj = TestingClass(dataset_PATH, wanted_w, wanted_h, export_w, export_h, max_piece_percent)
prediction_list, ground_truth_list, ext_class_name, ext_class_first_max_index_1, ext_class_second_max_index_1, ext_match_first_max_percent_1, ext_match_second_max_percent_1\
 = testing_obj.test_classifier_multiple_slow(dataset_PATH, [], num_classes,dropout, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL,\
                                             100,seed,350,706,min_percent_match, min_confidence,model_1_weights_name = model_1_weights_name, model_num = -1, exclude = [],save = True)

del testing_obj
del prediction_list
del ground_truth_list
del ext_class_name
del ext_class_first_max_index_1
del ext_class_second_max_index_1
del ext_match_first_max_percent_1
del ext_match_second_max_percent_1
gc.collect()

#%%
#####
seed = 1000

#model_1_weights_name = 'Sketch-A-Net_all_29739+13301_7_layers'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_1'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_2'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_3'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_4'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_5'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_6'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_7'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_model_8'
#model_1_weights_name = 'Sketch-A-Net_29739+7500(0-350)_7_layers'
model_1_weights_name = 'Sketch-A-Net_model_5_exclude_23'
testing_obj = TestingClass(dataset_PATH, wanted_w, wanted_h, export_w, export_h, max_piece_percent)
prediction_list, ground_truth_list, ext_class_name, ext_class_first_max_index_1, ext_class_second_max_index_1, ext_match_first_max_percent_1, ext_match_second_max_percent_1\
 = testing_obj.test_classifier_multiple_slow(dataset_PATH, [], num_classes,dropout, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL,\
                                             100,seed,350,706,min_percent_match, min_confidence,model_1_weights_name = model_1_weights_name, model_num = 5, exclude = [],save = True)

del testing_obj
del prediction_list
del ground_truth_list
del ext_class_name
del ext_class_first_max_index_1
del ext_class_second_max_index_1
del ext_match_first_max_percent_1
del ext_match_second_max_percent_1
gc.collect()


#%%
from helper_functions import separate_correct

correct_ext_match_first_max_percent_1, wrong_ext_match_first_max_percent_1 = separate_correct(ext_match_first_max_percent_1,prediction_list,ground_truth_list)
correct_ext_match_second_max_percent_1, wrong_ext_match_second_max_percent_1 = separate_correct(ext_match_second_max_percent_1,prediction_list,ground_truth_list)

correct_mean_first_max_percent_1 = sum(correct_ext_match_first_max_percent_1)/len(correct_ext_match_first_max_percent_1)
correct_max_first_max_percent_1 = max(correct_ext_match_first_max_percent_1)
correct_min_first_max_percent_1 = min(correct_ext_match_first_max_percent_1)

correct_mean_second_max_percent_1 = sum(correct_ext_match_second_max_percent_1)/len(correct_ext_match_second_max_percent_1)
correct_max_second_max_percent_1 = max(correct_ext_match_second_max_percent_1)
correct_min_second_max_percent_1 = min(correct_ext_match_second_max_percent_1)

wrong_mean_first_max_percent_1 = sum(wrong_ext_match_first_max_percent_1)/len(wrong_ext_match_first_max_percent_1)
wrong_max_first_max_percent_1 = max(wrong_ext_match_first_max_percent_1)
wrong_min_first_max_percent_1 = min(wrong_ext_match_first_max_percent_1)

wrong_mean_second_max_percent_1 = sum(wrong_ext_match_second_max_percent_1)/len(wrong_ext_match_second_max_percent_1)
wrong_max_second_max_percent_1 = max(wrong_ext_match_second_max_percent_1)
wrong_min_second_max_percent_1 = min(wrong_ext_match_second_max_percent_1)


correct_ext_match_first_max_percent_1_array = np.asarray(correct_ext_match_first_max_percent_1)
wrong_ext_match_first_max_percent_1_array = np.asarray(wrong_ext_match_first_max_percent_1)
correct_ext_match_second_max_percent_1_array = np.asarray(correct_ext_match_second_max_percent_1)
wrong_ext_match_second_max_percent_1_array = np.asarray(wrong_ext_match_second_max_percent_1)

#difference_ext_match_first_max_percent_1 = (correct_ext_match_first_max_percent_1_array - wrong_ext_match_first_max_percent_1_array).tolist()
#difference_ext_match_second_max_percent_1 = (correct_ext_match_second_max_percent_1_array - wrong_ext_match_second_max_percent_1_array).tolist()

difference_correct = (correct_ext_match_first_max_percent_1_array - correct_ext_match_second_max_percent_1_array).tolist()
difference_wrong = (wrong_ext_match_first_max_percent_1_array - wrong_ext_match_second_max_percent_1_array).tolist()

#difference_mean_first_max_percent_1 = sum(difference_ext_match_first_max_percent_1)/len(difference_ext_match_first_max_percent_1)
#difference_max_first_max_percent_1 = max(difference_ext_match_first_max_percent_1)
#difference_min_first_max_percent_1 = min(difference_ext_match_first_max_percent_1)
#
#difference_mean_second_max_percent_1 = sum(difference_ext_match_second_max_percent_1)/len(difference_ext_match_second_max_percent_1)
#difference_max_second_max_percent_1 = max(difference_ext_match_second_max_percent_1)
#difference_min_second_max_percent_1 = min(difference_ext_match_second_max_percent_1)

difference_correct_mean = sum(difference_correct)/len(difference_correct)
difference_wrong_mean = sum(difference_wrong)/len(difference_wrong)
difference_correct_max = max(difference_correct)
difference_wrong_max = max(difference_wrong)
difference_correct_min = min(difference_correct)
difference_wrong_min = min(difference_wrong)


f = open(dataset_PATH+'testing_results_entropy.txt','a')
f.writelines(str(model_1_weights_name)+'\n')
f.writelines(\
'correct_mean_first_max_percent_1 = ' + str(correct_mean_first_max_percent_1) + '\n'\
\
'correct_max_first_max_percent_1 = ' + str(correct_max_first_max_percent_1) + '\n'\
\
'correct_min_first_max_percent_1 = ' + str(correct_min_first_max_percent_1) + '\n'\
\
'correct_mean_second_max_percent_1 = ' + str(correct_mean_second_max_percent_1) + '\n'\
\
'correct_max_second_max_percent_1 = ' + str(correct_max_second_max_percent_1) + '\n'\
\
'correct_min_second_max_percent_1 = ' + str(correct_min_second_max_percent_1) + '\n'\
+'\n'\
'wrong_mean_first_max_percent_1 = ' + str(wrong_mean_first_max_percent_1) + '\n'\
\
'wrong_max_first_max_percent_1 = ' + str(wrong_max_first_max_percent_1) + '\n'\
\
'wrong_min_first_max_percent_1 = ' + str(wrong_min_first_max_percent_1) + '\n'\
\
'wrong_mean_second_max_percent_1 = ' + str(wrong_mean_second_max_percent_1) + '\n'\
\
'wrong_max_second_max_percent_1 = ' + str(wrong_max_second_max_percent_1) + '\n'\
\
'wrong_min_second_max_percent_1 = ' + str(wrong_min_second_max_percent_1) + '\n'\
+'\n'\
'difference_correct_mean = ' + str(difference_correct_mean) + '\n'\
\
'difference_wrong_mean = ' + str(difference_wrong_mean) + '\n'\
\
'difference_correct_max = ' + str(difference_correct_max) + '\n'\
\
'difference_wrong_max = ' + str(difference_wrong_max) + '\n'\
\
'difference_correct_min = ' + str(difference_correct_min) + '\n'\
\
'difference_wrong_min = ' + str(difference_wrong_min) + '\n'\
+'\n'\
)

f.close()
#%%


from helper_functions import separate_correct

correct_ext_match_first_max_percent_1, wrong_ext_match_first_max_percent_1 = separate_correct(ext_match_first_max_percent_1,prediction_list,ground_truth_list)
correct_ext_match_first_max_percent_2, wrong_ext_match_first_max_percent_2 = separate_correct(ext_match_first_max_percent_2,prediction_list,ground_truth_list)
correct_ext_match_first_max_percent_3, wrong_ext_match_first_max_percent_3 = separate_correct(ext_match_first_max_percent_3,prediction_list,ground_truth_list)

correct_ext_match_second_max_pecent_1, wrong_ext_match_second_max_pecent_1 = separate_correct(ext_match_second_max_percent_1,prediction_list,ground_truth_list)
correct_ext_match_second_max_pecent_2, wrong_ext_match_second_max_pecent_2 = separate_correct(ext_match_second_max_percent_2,prediction_list,ground_truth_list)
correct_ext_match_second_max_pecent_3, wrong_ext_match_second_max_pecent_3 = separate_correct(ext_match_second_max_percent_3,prediction_list,ground_truth_list)


correct_mean_first_max_pecent_1 = sum(correct_ext_match_first_max_percent_1)/len(correct_ext_match_first_max_percent_1)
correct_mean_first_max_pecent_2 = sum(correct_ext_match_first_max_percent_2)/len(correct_ext_match_first_max_percent_2)
correct_mean_first_max_pecent_3 = sum(correct_ext_match_first_max_percent_3)/len(correct_ext_match_first_max_percent_3)

correct_max_first_max_percent_1 = max(correct_ext_match_first_max_percent_1)
correct_max_first_max_percent_2 = max(correct_ext_match_first_max_percent_2)
correct_max_first_max_percent_3 = max(correct_ext_match_first_max_percent_3)

correct_min_first_max_percent_1 = min(correct_ext_match_first_max_percent_1)
correct_min_first_max_percent_2 = min(correct_ext_match_first_max_percent_2)
correct_min_first_max_percent_3 = min(correct_ext_match_first_max_percent_3)

correct_mean_second_max_pecent_1 = sum(correct_ext_match_second_max_percent_1)/len(correct_ext_match_second_max_percent_1)
correct_mean_second_max_pecent_2 = sum(correct_ext_match_second_max_percent_2)/len(correct_ext_match_second_max_percent_2)
correct_mean_second_max_pecent_3 = sum(correct_ext_match_second_max_percent_3)/len(correct_ext_match_second_max_percent_3)

correct_max_second_max_percent_1 = max(correct_ext_match_second_max_percent_1)
correct_max_second_max_percent_2 = max(correct_ext_match_second_max_percent_2)
correct_max_second_max_percent_3 = max(correct_ext_match_second_max_percent_3)

correct_min_second_max_percent_1 = min(correct_ext_match_second_max_percent_1)
correct_min_second_max_percent_2 = min(correct_ext_match_second_max_percent_2)
correct_min_second_max_percent_3 = min(correct_ext_match_second_max_percent_3)


wrong_mean_first_max_pecent_1 = sum(wrong_ext_match_first_max_percent_1)/len(wrong_ext_match_first_max_percent_1)
wrong_mean_first_max_pecent_2 = sum(wrong_ext_match_first_max_percent_2)/len(wrong_ext_match_first_max_percent_2)
wrong_mean_first_max_pecent_3 = sum(wrong_ext_match_first_max_percent_3)/len(wrong_ext_match_first_max_percent_3)

wrong_max_first_max_percent_1 = max(wrong_ext_match_first_max_percent_1)
wrong_max_first_max_percent_2 = max(wrong_ext_match_first_max_percent_2)
wrong_max_first_max_percent_3 = max(wrong_ext_match_first_max_percent_3)

wrong_min_first_max_percent_1 = min(wrong_ext_match_first_max_percent_1)
wrong_min_first_max_percent_2 = min(wrong_ext_match_first_max_percent_2)
wrong_min_first_max_percent_3 = min(wrong_ext_match_first_max_percent_3)

wrong_mean_second_max_pecent_1 = sum(wrong_ext_match_second_max_percent_1)/len(wrong_ext_match_second_max_percent_1)
wrong_mean_second_max_pecent_2 = sum(wrong_ext_match_second_max_percent_2)/len(wrong_ext_match_second_max_percent_2)
wrong_mean_second_max_pecent_3 = sum(wrong_ext_match_second_max_percent_3)/len(wrong_ext_match_second_max_percent_3)

wrong_max_second_max_percent_1 = max(wrong_ext_match_second_max_percent_1)
wrong_max_second_max_percent_2 = max(wrong_ext_match_second_max_percent_2)
wrong_max_second_max_percent_3 = max(wrong_ext_match_second_max_percent_3)

wrong_min_second_max_percent_1 = min(wrong_ext_match_second_max_percent_1)
wrong_min_second_max_percent_2 = min(wrong_ext_match_second_max_percent_2)
wrong_min_second_max_percent_3 = min(wrong_ext_match_second_max_percent_3)

f = open(dataset_PATH+'testing_results_entropy.txt','a')
f.writelines(\
'correct_mean_first_max_pecent_1 = ' + str(correct_mean_first_max_percent_1) + '\n'\
'correct_mean_first_max_pecent_2 = ' + str(correct_mean_first_max_percent_2) + '\n'\
'correct_mean_first_max_pecent_3 = ' + str(correct_mean_first_max_percent_3) + '\n'\
\
'correct_max_first_max_percent_1 = ' + str(correct_max_first_max_percent_1) + '\n'\
'correct_max_first_max_percent_2 = ' + str(correct_max_first_max_percent_2) + '\n'\
'correct_max_first_max_percent_3 = ' + str(correct_max_first_max_percent_3) + '\n'\
\
'correct_min_first_max_percent_1 = ' + str(correct_mix_first_max_percent_1) + '\n'\
'correct_min_first_max_percent_2 = ' + str(correct_min_first_max_percent_2) + '\n'\
'correct_min_first_max_percent_3 = ' + str(correct_min_first_max_percent_3) + '\n'\
\
'correct_mean_second_max_pecent_1 = ' + str(correct_mean_second_max_percent_1) + '\n'\
'correct_mean_second_max_pecent_2 = ' + str(correct_mean_second_max_percent_2) + '\n'\
'correct_mean_second_max_pecent_3 = ' + str(correct_mean_second_max_percent_3) + '\n'\
\
'correct_max_second_max_percent_1 = ' + str(correct_max_second_max_percent_1) + '\n'\
'correct_max_second_max_percent_2 = ' + str(correct_max_second_max_percent_2) + '\n'\
'correct_max_second_max_percent_3 = ' + str(correct_max_second_max_percent_3) + '\n'\
\
'correct_min_second_max_percent_1 = ' + str(correct_min_second_max_percent_1) + '\n'\
'correct_min_second_max_percent_2 = ' + str(correct_min_second_max_percent_2) + '\n'\
'correct_min_second_max_percent_3 = ' + str(correct_min_second_max_percent_3) + '\n'\
\
'wrong_mean_first_max_pecent_1 = ' + str(wrong_mean_first_max_percent_1) + '\n'\
'wrong_mean_first_max_pecent_2 = ' + str(wrong_mean_first_max_percent_2) + '\n'\
'wrong_mean_first_max_pecent_3 = ' + str(wrong_mean_first_max_percent_3) + '\n'\
\
'wrong_max_first_max_percent_1 = ' + str(wrong_max_first_max_percent_1) + '\n'\
'wrong_max_first_max_percent_2 = ' + str(wrong_max_first_max_percent_2) + '\n'\
'wrong_max_first_max_percent_3 = ' + str(wrong_max_first_max_percent_3) + '\n'\
\
'wrong_min_first_max_percent_1 = ' + str(wrong_mix_first_max_percent_1) + '\n'\
'wrong_min_first_max_percent_2 = ' + str(wrong_min_first_max_percent_2) + '\n'\
'wrong_min_first_max_percent_3 = ' + str(wrong_min_first_max_percent_3) + '\n'\
\
'wrong_mean_second_max_pecent_1 = ' + str(wrong_mean_second_max_percent_1) + '\n'\
'wrong_mean_second_max_pecent_2 = ' + str(wrong_mean_second_max_percent_2) + '\n'\
'wrong_mean_second_max_pecent_3 = ' + str(wrong_mean_second_max_percent_3) + '\n'\
\
'wrong_max_second_max_percent_1 = ' + str(wrong_max_second_max_percent_1) + '\n'\
'wrong_max_second_max_percent_2 = ' + str(wrong_max_second_max_percent_2) + '\n'\
'wrong_max_second_max_percent_3 = ' + str(wrong_max_second_max_percent_3) + '\n'\
\
'wrong_min_second_max_percent_1 = ' + str(wrong_min_second_max_percent_1) + '\n'\
'wrong_min_second_max_percent_2 = ' + str(wrong_min_second_max_percent_2) + '\n'\
'wrong_min_second_max_percent_3 = ' + str(wrong_min_second_max_percent_3) + '\n'\
)

f.close()
#%%
from helper_functions import get_optimal_entropy_parameters
from helper_functions import print_all_entropy_parameters
from data_analysis_data import prediction_list_exclude_23_32898 as prediction_list
from data_analysis_data import ground_truth_list_exclude_23_32898 as ground_truth_list
from data_analysis_data import ext_class_first_max_index_exclude_23_32898 as ext_class_first_max_index
from data_analysis_data import ext_class_second_max_index_exclude_23_32898 as ext_class_second_max_index
from data_analysis_data import ext_match_first_max_percent_exclude_23_32898 as ext_match_first_max_percent
from data_analysis_data import ext_match_second_max_percent_exclude_23_32898 as ext_match_second_max_percent
#
#from data_analysis_data import prediction_list_Sketch_A_Net_model_1_exclude_23 as prediction_list
#from data_analysis_data import ground_truth_list_Sketch_A_Net_model_1_exclude_23 as ground_truth_list
#from data_analysis_data import ext_class_first_max_index_1_Sketch_A_Net_model_1_exclude_23 as ext_class_first_max_index
#from data_analysis_data import ext_class_second_max_index_1_Sketch_A_Net_model_1_exclude_23 as ext_class_second_max_index
#from data_analysis_data import ext_match_first_max_percent_1_Sketch_A_Net_model_1_exclude_23 as ext_match_first_max_percent
#from data_analysis_data import ext_match_second_max_percent_1_Sketch_A_Net_model_1_exclude_23 as ext_match_second_max_percent


seed = 1000
iters = 100
resolution = 0.001
min_percent_match_start=0.8
min_percent_match_end=0.999
min_confidence_start=0.7
min_confidence_end = 0.85
X, Y ,Z = print_all_entropy_parameters(prediction_list,
                                    ground_truth_list,
                                    ext_class_first_max_index,
                                    ext_class_second_max_index,
                                    ext_match_first_max_percent,
                                    ext_match_second_max_percent, 
                                    resolution)
Zneg = 100-Z
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X1, Y1, Z1)
ax.set_xlabel('min_percent_match')
ax.set_ylabel('min_confidence')
ax.set_zlabel('accuracy (%)')
plt.show()
dataset_PATH = "C:/Users/JustinSanJuan/Desktop/HKUST/UROP Deep Learning Image-based Structural Analysis/Code/Python/Testing Folder/"
fig.savefig(dataset_PATH + '3D_plot_1')

seed = 1234
min_percent_match, min_confidence = get_optimal_entropy_parameters(prediction_list,
                                    ground_truth_list,
                                    ext_class_first_max_index,
                                    ext_class_second_max_index,
                                    ext_match_first_max_percent,
                                    ext_match_second_max_percent,iters,seed, resolution)


prediction_obj = ComponentClassifierPredict(min_percent_match, min_confidence)

#Following are lists:
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
    
index, index_1,index_2,index_3, first_max_index_1, second_max_index_1, first_max_percent_1, second_max_percent_1,\
first_max_index_2, second_max_index_2, first_max_percent_2, second_max_percent_2,\
first_max_index_3, second_max_index_3, first_max_percent_3, second_max_percent_3 = prediction_obj.predict_class_3_models(image,dataset_PATH, model_1_weights_name, model_2_weights_name, model_3_weights_name,return_all=True)



ext_class_index.append(index)
ext_class_name.append(target_names_all[index])

ext_class_first_max_index_1.append(first_max_index_1)
ext_class_second_max_index_1.append(second_max_index_1)
ext_match_first_max_percent_1.append(first_max_percent_1)
ext_match_second_max_percent_1.append(second_max_percent_1)

ext_class_first_max_index_2.append(first_max_index_2)
ext_class_second_max_index_2.append(second_max_index_2)
ext_match_first_max_percent_2.append(first_max_percent_2)
ext_match_second_max_percent_2.append(second_max_percent_2)

ext_class_first_max_index_3.append(first_max_index_3)
ext_class_second_max_index_3.append(second_max_index_3)
ext_match_first_max_percent_3.append(first_max_percent_3)
ext_match_second_max_percent_3.append(second_max_percent_3)

# In[6]:
start = time.time() # Begin time measurement

seed = 1000
training_obj = ComponentClassifierTraining(num_classes, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL)
'''
#To get model shape = (100, 100,1)
training_obj.shuffle_data(training_obj.load_data(dataset_PATH,base_dataset_name),seed)
print(training_obj.X_train.shape[1:])
'''
#Model is Sketch_a_net
training_obj.model = training_obj.load_sketch_a_net_model_7_layers(dropout, num_classes,(100,100,1))

dataset_name_1 = "Training_Samples_64_classes_100x100_all_controlled_30858"  # base training images
#dataset_name_2 = "Training_Samples_64_classes_100x100_all_cleaned_updated_7500_0-350" # problem ground truth images
dataset_name_list = [dataset_name_1]
#new_dataset_name = training_obj.control_dataset(dataset_PATH, dataset_name_list,num_classes,600)
data_count_list = training_obj.count_dataset(dataset_PATH, dataset_name_list,num_classes)

training_obj.train_from_multiple_files(100,seed,dataset_PATH,dataset_name_list,verbose = 1)
weights_name = "Sketch-A-Net_controlled_600_30858_7_layers"
training_obj.save(dataset_PATH+weights_name)



training_obj = ComponentClassifierTraining(num_classes, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL)
'''
#To get model shape = (100, 100,1)
training_obj.shuffle_data(training_obj.load_data(dataset_PATH,base_dataset_name),seed)
print(training_obj.X_train.shape[1:])
'''
#Model is Sketch_a_net
training_obj.model = training_obj.load_sketch_a_net_model_7_layers(dropout, num_classes,(100,100,1))

dataset_name_1 = "Training_Samples_64_classes_100x100_all_controlled_30858_1" # base training images
#dataset_name_2 = "Training_Samples_64_classes_100x100_all_cleaned_13291" # problem ground truth images
dataset_name_list = [dataset_name_1]
new_dataset_name = training_obj.control_dataset(dataset_PATH, dataset_name_list,num_classes,600)
data_count_list = training_obj.count_dataset(dataset_PATH, [new_dataset_name],num_classes)

training_obj.train_from_multiple_files(100,seed,dataset_PATH,dataset_name_list,verbose = 1)
weights_name = "Sketch-A-Net_controlled_600_30858_1_7_layers"
training_obj.save(dataset_PATH+weights_name)




training_obj = ComponentClassifierTraining(num_classes, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL)
'''
#To get model shape = (100, 100,1)
training_obj.shuffle_data(training_obj.load_data(dataset_PATH,base_dataset_name),seed)
print(training_obj.X_train.shape[1:])
'''
#Model is Sketch_a_net
training_obj.model = training_obj.load_sketch_a_net_model_7_layers(dropout, num_classes,(100,100,1))

dataset_name_1 = "Training_Samples_64_classes_100x100_all_cleaned_32898" # base training images
#dataset_name_2 = "Training_Samples_64_classes_100x100_all_cleaned_13291" # problem ground truth images
dataset_name_list = [dataset_name_1]
new_dataset_name = training_obj.control_dataset(dataset_PATH, dataset_name_list,num_classes,600)
data_count_list = training_obj.count_dataset(dataset_PATH, [new_dataset_name],num_classes)

training_obj.train_from_multiple_files(100,seed,dataset_PATH,dataset_name_list,verbose = 1)
weights_name = "Sketch-A-Net_exclude_23_32898_7_layers"
training_obj.save(dataset_PATH+weights_name)
#training_obj.train(100,seed)
#training_obj.save(name+'_'+str(i))
#training_obj.model.load_weights(PATH+name+'.h5')

trained_model = training_obj.model

#training_obj.plot_training_data_all(dataset_PATH,new_dataset_name,0)

end = time.time()#record time
print('ComponentClassifierTraining done... Time Elapsed : '+ str(end-start) + ' seconds...')
t4 = end-start





training_obj = ComponentClassifierTraining(num_classes, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL)
'''
#To get model shape = (100, 100,1)
training_obj.shuffle_data(training_obj.load_data(dataset_PATH,base_dataset_name),seed)
print(training_obj.X_train.shape[1:])
'''
#Model is Sketch_a_net
training_obj.model = training_obj.load_sketch_a_net_model_7_layers(dropout, num_classes,(100,100,1))

dataset_name_1 = "Training_Samples_64_classes_100x100_all_cleaned_updated_29739"
dataset_name_2 = "Training_Samples_64_classes_100x100_all_cleaned_updated_13301_all_problem_images"
dataset_name_list = [dataset_name_1,dataset_name_2]
#new_dataset_name = training_obj.control_dataset(dataset_PATH, dataset_name_list,num_classes,600)
data_count_list = training_obj.count_dataset(dataset_PATH, dataset_name_list,num_classes)

training_obj.train_from_multiple_files(100,seed,dataset_PATH,dataset_name_list,verbose = 1)
weights_name = "Sketch-A-Net_all_29739+13301_7_layers"
training_obj.save(dataset_PATH+weights_name)
#training_obj.train(100,seed)
#training_obj.save(name+'_'+str(i))
#training_obj.model.load_weights(PATH+name+'.h5')

trained_model = training_obj.model





"""






"""
# In[6]:

'''
start = time.time() # Begin time measurement

seed = 1000
i = 1234
training_obj = ComponentClassifierTraining(num_classes, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL)
'''
#To get model shape = (100, 100,1)
training_obj.shuffle_data(training_obj.load_data(dataset_PATH,base_dataset_name),seed)
print(training_obj.X_train.shape[1:])
'''
#Model is Sketch_a_net
training_obj.model = training_obj.load_sketch_a_net_model(dropout, num_classes,(100,100,1))

dataset_name_1 = "Training_Samples_64_classes_100x100_all_cleaned_updated_29739"
dataset_name_2 = "Training_Samples_64_classes_100x100_all_cleaned_updated_13301_all_problem_images"
dataset_name_list = [dataset_name_2, dataset_name_1]
count_list = training_obj.control_dataset(dataset_PATH, dataset_name_list,num_classes,600)
#data_count = training_obj.count_dataset(dataset_PATH, dataset_name_list,num_classes)
'''
#%%
training_obj.train_from_multiple_files(200,seed,dataset_PATH,dataset_name_list,verbose = 1)
weights_name = "Training_Samples_64_classes_100x100_all_cleaned_updated_29739+13301_121epochs"
#training_obj.is_trained = True
training_obj.save(dataset_PATH+weights_name)
#training_obj.train(100,seed)
#training_obj.save(name+'_'+str(i))
#training_obj.model.load_weights(PATH+name+'.h5')

trained_model = training_obj.model

#training_obj.plot_training_data_all(dataset_PATH,new_dataset_name,0)

end = time.time()#record time
print('ComponentClassifierTraining done... Time Elapsed : '+ str(end-start) + ' seconds...')
t4 = end-start


# # ComponentClassifierPredict
# ### The ComponentClassifierPredict object is first initialized with the entropy-based hyperparameters:
#     - min_percent_match
#     - min_confidence
# These parameters were explained in the Hyperparameters section above.
# ### The predict_classes function produces the following:
#     - ext_class_index_list: ordered list of highest % match class predictions for each
# The entropy-based modifications are applied to the above (such that if any of the two criteria are not satisfied, the prediction is classified as random)
#     - ext_class_name_list: ordered list of corresponding names to ext_class_index_list
#     - ext_match_first_max_percent_list: ordered list of corresponding first-highest match percentage
#     - ext_match_second_max_percent_list: ordered list of corresponding second-highest match percentage
# In[8]:
seed = 1000

#weights_name = "Training_Samples_64_classes_100x100_all_cleaned_updated_29739+7500(0-350)"
#weights_name = dataset_name

dataset_name_1 = "Training_Samples_64_classes_100x100_all_cleaned_29724" # base training images
dataset_name_2 = "Training_Samples_64_classes_100x100_all_cleaned_13291" # problem ground truth images
dataset_name_list = [dataset_name_1, dataset_name_2]

testing_obj = TestingClass(dataset_PATH, wanted_w, wanted_h, export_w, export_h, max_piece_percent)
ground_truth_list, prediction_list = testing_obj.test_classifier_multiple_slow(dataset_PATH, dataset_name_list,
                                     num_classes,dropout, 
                                     TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL,
                                     200,seed,350,706)
#, weights_name = weights_name
#testing_obj.test_classifier_all(dataset_PATH, dataset_name, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL,200,seed,400) 
# In[9]:
# Plot confusion matrix
#Add base data for confusion matrix
for i in range(64):
    ground_truth_list.append(i)
    prediction_list.append(i)
    
# Compute confusion matrix
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(ground_truth_list,prediction_list)

# Plot non-normalized confusion matrix
from helper_functions import plot_confusion_matrix
from constants import target_names_all
import matplotlib.pyplot as plt
plot_confusion_matrix(cnf_matrix, classes=target_names_all,
                      normalize=False,
                      title='Confusion matrix', 
                      cmap=plt.cm.Blues,PATH=dataset_PATH, name="confusion_matrix_"+str(confusion_matrix_index), verbose = False)

from helper_functions import confusion_matrix_analysis
dataset_PATH = "C:/Users/JustinSanJuan/Desktop/HKUST/UROP Deep Learning Image-based Structural Analysis/Code/Python/Testing Folder/"
name = "confusion_matrix_"+str(confusion_matrix_index)+"_analysis"
min_count = 5
confusion_matrix_analysis(cnf_matrix, dataset_PATH, name, min_count, verbose = False) #Turn verbose on to show data analysis

# In[3]:


start = time.time() # Begin time measurement

image_index = 50
image_set = np.load(PATH+'easy_training_images.npy')
image = np.copy(image_set[:,:,image_index])
image_set = None #clear image_set
gc.collect() #clear unreferenced data

end = time.time()#record time
print('Loading image done... Time Elapsed : '+ str(end-start) + ' seconds...')
t1 = end-start

l,w = 15,15 #dimension scales of print
print_image_bw(image,l,w)


# # Component Segmentation
# ##### Using the ComponentSegmentation class:
#     1. Selective Search is applied to the image (to generate bounding boxes)
#     2. A merging algorithm is applied to the selective search bounding boxes (to merge highly overlapping bounding boxes)
# ##### The ComponentSegmentation uses the following data for initialization:
#     - image: binary (grayscale) 2-D array for region proposal
#     - name: for unique prints saving
#     - min_shape, min_height, min_width, buffer_zone, min_area, min_black, min_black_ratio: for noise reduction
#     - overlap_repeats, overlap_threshold: for merging algorithm
# ##### Then, the RoI proposal is done using the custom search method, which uses the selective search hyper-parameters:
#     - scale_input
#     - sigma_input
#     - min_size_input
# ##### Then, the merging algorithm is applied within the search function, and a merged_set is retrieved.

# In[4]:


start = time.time() # Begin time measurement

#Create object ComponentSegmentation, which will use the search function to perform segmentation and merging.
segmentation_obj = ComponentSegmentation(image, name, min_shape, min_height, min_width, buffer_zone, min_area, min_black, min_black_ratio, overlap_repeats, overlap_threshold)
segmentation_obj.search(scale_input, sigma_input, min_size_input) # run search (segmentation code)
merged_set = segmentation_obj.merged_set

end = time.time()#record time
print('ComponentSegmentation done... Time Elapsed : '+ str(end-start) + ' seconds...')
t2 = end-start


# # ExtractionPreprocessing
# #### Merged set from ComponentSegmentation is passed to ExtractionPreprocessing and the following is applied:
#     1. Trim: extra space around the farthest black pixels are removed
#     2. Remove Unconnected Parts: extra pixels (from other components) captured by the bounding box are removed
#     3. Trim: trimming again as empty spaces may be released
#     4. Resize: extraction is resized to the prescribed 100x100 dimension using max pooling for downsampling to preserve data
# ext_images_list = extraction images: list of 100x100 binary (grayscale) 2-D arrays<br>
# ext_data_list = extraction data: list of x, y, w, h data of extractions bounding boxes<br> 
# where:<br>
#     - x, y: top-left corner coordinates of bounding box
#     - w, h: width and height of bounding box respectively
# #### The preprocess_extractions function is called and the extraction images and extraction data are acquired.
# #### The plot_bounding_boxes_with_names function is then used to display the bounding boxes on the original image.

# In[5]:


start = time.time() # Begin time measurement

#Transport data into ExtractionPreprocessing class, which will trim, remove unconnected parts, then trim, and resize
extraction_obj = ExtractionPreprocessing(image, name, merged_set)

# Get 4 lists from preprocess_extractions function
ext_images_list, ext_data_list = extraction_obj.preprocess_extractions(wanted_w, wanted_h, export_w, export_h, max_piece_percent)
extraction_obj.plot_bounding_boxes_with_names()

end = time.time()#record time
print('ExtractionPreprocessing done... Time Elapsed : '+ str(end-start) + ' seconds...')
t3 = end-start


# # ComponentClassifierTraining (Not pre-trained)
# ### If model has been trained before:
# ### Then the train and save functions should be replaced with:
# training_obj.model.load_weights(PATH+name+'.h5')
# ### Such that the below code is the following:
# training_obj = ComponentClassifierTraining(PATH, base_dataset_name, num_classes, dropout, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL)<br>
# training_obj.shuffle_data(training_obj.load_data(PATH,base_dataset_name),1000)<br>
# <br>
# #Model is Sketch_a_net<br>
# training_obj.model = training_obj.load_sketch_a_net_model(dropout, num_classes, training_obj.X_train.shape[1:])<br>
# training_obj.model.load_weights(PATH+name+'.h5')<br>
# ##### The ComponentClassifierTraining object is first initialized with:
#     - PATH: working directory
#     - base_dataset_name: for loading training data set
#     - num_classes, dropout: CNN model parameters
#     - TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL: training parameters
# ##### The shuffle_data is then called to shuffle the training data using a seed<br>The Sketch_A_Net model is then loaded<br>Then the model is trained with 100 epochs<br>Then the model weights are saved<br>Finally the trained model is stored in trained_model to be passed onto a ComponentClassifierPredict object
# #### If model weights have been trained before, the training and saving is not required, and the load_weights function has to be called instead.



# In[6]:
start = time.time() # Begin time measurement

seed = 1000
i = 1234
training_obj = ComponentClassifierTraining(num_classes, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL)
'''
#To get model shape = (100, 100,1)
training_obj.shuffle_data(training_obj.load_data(dataset_PATH,base_dataset_name),seed)
print(training_obj.X_train.shape[1:])
'''
#Model is Sketch_a_net
training_obj.model = training_obj.load_sketch_a_net_model(dropout, num_classes,(100,100,1))

dataset_name_1 = "Training_Samples_64_classes_100x100_all_cleaned_29724" # base training images
dataset_name_2 = "Training_Samples_64_classes_100x100_all_cleaned_13291" # problem ground truth images
dataset_name_list = [dataset_name_1, dataset_name_2]
count_list = training_obj.control_dataset(dataset_PATH, dataset_name_list,num_classes,600)
data_count_list = training_obj.count_dataset(dataset_PATH, dataset_name_list,num_classes)

training_obj.train_from_multiple_files(100,seed,dataset_PATH,dataset_name_list,verbose = 1)
weights_name = "Training_Samples_64_classes_100x100_all_cleaned_updated_29724+13291"
training_obj.save(dataset_PATH+weights_name)
#training_obj.train(100,seed)
#training_obj.save(name+'_'+str(i))
#training_obj.model.load_weights(PATH+name+'.h5')

trained_model = training_obj.model

#training_obj.plot_training_data_all(dataset_PATH,new_dataset_name,0)

end = time.time()#record time
print('ComponentClassifierTraining done... Time Elapsed : '+ str(end-start) + ' seconds...')
t4 = end-start


# # ComponentClassifierPredict
# ### The ComponentClassifierPredict object is first initialized with the entropy-based hyperparameters:
#     - min_percent_match
#     - min_confidence
# These parameters were explained in the Hyperparameters section above.
# ### The predict_classes function produces the following:
#     - ext_class_index_list: ordered list of highest % match class predictions for each
# The entropy-based modifications are applied to the above (such that if any of the two criteria are not satisfied, the prediction is classified as random)
#     - ext_class_name_list: ordered list of corresponding names to ext_class_index_list
#     - ext_match_first_max_percent_list: ordered list of corresponding first-highest match percentage
#     - ext_match_second_max_percent_list: ordered list of corresponding second-highest match percentage

# In[7]:

"""
start = time.time() # Begin time measurement
"""
prediction_obj = ComponentClassifierPredict(min_percent_match, min_confidence)


#Following are lists:
ext_class_index, ext_class_name, ext_class_first_max_index_1, ext_class_second_max_index_1, ext_match_first_max_percent_1, ext_match_second_max_percent_1,\
ext_class_first_max_index_2, ext_class_second_max_index_2, ext_match_first_max_percent_2, ext_match_second_max_percent_2,\
ext_class_first_max_index_3, ext_class_second_max_index_3, ext_match_first_max_percent_3, ext_match_second_max_percent_3 = prediction_obj.predict_classes_3_models(ext_images_list,PATH, model_1_weights_name, model_2_weights_name, model_3_weights_name,return_all=True)


#separate correct and wrong
from data_analysis_data import ground_truth_list as ground_truth_list

def separate_correct(list_to_be_separated, prediction_list, ground_truth_list):
    ground_truth_array = np.asarray(ground_truth_list)
    prediction_array = np.asarray(prediction_list)
    correct_list = (ground_truth_array == prediction_array)
    correct_output_list = []
    wrong_output_list = []
    for i in range(len(correct_list)):
        if correct_list[i] == True:
            correct_output_list.append(list_to_be_separated[i])
        else:
            wrong_output_list.append(list_to_be_separated[i])
    return correct_output_list, wrong_output_list

correct_ext_match_first_max_pecent_1, wrong_ext_match_first_max_pecent_1 = separate_correct(ext_match_first_max_percent_1,prediction_list,ground_truth_list)
correct_ext_match_first_max_pecent_2, wrong_ext_match_first_max_pecent_2 = separate_correct(ext_match_first_max_percent_2,prediction_list,ground_truth_list)
correct_ext_match_first_max_pecent_3, wrong_ext_match_first_max_pecent_3 = separate_correct(ext_match_first_max_percent_3,prediction_list,ground_truth_list)

correct_ext_match_second_max_pecent_1, wrong_ext_match_second_max_pecent_1 = separate_correct(ext_match_second_max_percent_1,prediction_list,ground_truth_list)
correct_ext_match_second_max_pecent_2, wrong_ext_match_second_max_pecent_2 = separate_correct(ext_match_second_max_percent_2,prediction_list,ground_truth_list)
correct_ext_match_second_max_pecent_3, wrong_ext_match_second_max_pecent_3 = separate_correct(ext_match_second_max_percent_3,prediction_list,ground_truth_list)


correct_mean_first_max_pecent_1 = sum(correct_ext_match_first_max_percent_1)/len(correct_ext_match_first_max_percent_1)
correct_mean_first_max_pecent_2 = sum(correct_ext_match_first_max_percent_2)/len(correct_ext_match_first_max_percent_2)
correct_mean_first_max_pecent_3 = sum(correct_ext_match_first_max_percent_3)/len(correct_ext_match_first_max_percent_3)

correct_max_first_max_percent_1 = max(correct_ext_match_first_max_percent_1)
correct_max_first_max_percent_2 = max(correct_ext_match_first_max_percent_2)
correct_max_first_max_percent_3 = max(correct_ext_match_first_max_percent_3)

correct_min_first_max_percent_1 = min(correct_ext_match_first_max_percent_1)
correct_min_first_max_percent_2 = min(correct_ext_match_first_max_percent_2)
correct_min_first_max_percent_3 = min(correct_ext_match_first_max_percent_3)

correct_mean_second_max_pecent_1 = sum(correct_ext_match_second_max_percent_1)/len(correct_ext_match_second_max_percent_1)
correct_mean_second_max_pecent_2 = sum(correct_ext_match_second_max_percent_2)/len(correct_ext_match_second_max_percent_2)
correct_mean_second_max_pecent_3 = sum(correct_ext_match_second_max_percent_3)/len(correct_ext_match_second_max_percent_3)

correct_max_second_max_percent_1 = max(correct_ext_match_second_max_percent_1)
correct_max_second_max_percent_2 = max(correct_ext_match_second_max_percent_2)
correct_max_second_max_percent_3 = max(correct_ext_match_second_max_percent_3)

correct_min_second_max_percent_1 = min(correct_ext_match_second_max_percent_1)
correct_min_second_max_percent_2 = min(correct_ext_match_second_max_percent_2)
correct_min_second_max_percent_3 = min(correct_ext_match_second_max_percent_3)


wrong_mean_first_max_pecent_1 = sum(wrong_ext_match_first_max_percent_1)/len(wrong_ext_match_first_max_percent_1)
wrong_mean_first_max_pecent_2 = sum(wrong_ext_match_first_max_percent_2)/len(wrong_ext_match_first_max_percent_2)
wrong_mean_first_max_pecent_3 = sum(wrong_ext_match_first_max_percent_3)/len(wrong_ext_match_first_max_percent_3)

wrong_max_first_max_percent_1 = max(wrong_ext_match_first_max_percent_1)
wrong_max_first_max_percent_2 = max(wrong_ext_match_first_max_percent_2)
wrong_max_first_max_percent_3 = max(wrong_ext_match_first_max_percent_3)

wrong_min_first_max_percent_1 = min(wrong_ext_match_first_max_percent_1)
wrong_min_first_max_percent_2 = min(wrong_ext_match_first_max_percent_2)
wrong_min_first_max_percent_3 = min(wrong_ext_match_first_max_percent_3)

wrong_mean_second_max_pecent_1 = sum(wrong_ext_match_second_max_percent_1)/len(wrong_ext_match_second_max_percent_1)
wrong_mean_second_max_pecent_2 = sum(wrong_ext_match_second_max_percent_2)/len(wrong_ext_match_second_max_percent_2)
wrong_mean_second_max_pecent_3 = sum(wrong_ext_match_second_max_percent_3)/len(wrong_ext_match_second_max_percent_3)

wrong_max_second_max_percent_1 = max(wrong_ext_match_second_max_percent_1)
wrong_max_second_max_percent_2 = max(wrong_ext_match_second_max_percent_2)
wrong_max_second_max_percent_3 = max(wrong_ext_match_second_max_percent_3)

wrong_min_second_max_percent_1 = min(wrong_ext_match_second_max_percent_1)
wrong_min_second_max_percent_2 = min(wrong_ext_match_second_max_percent_2)
wrong_min_second_max_percent_3 = min(wrong_ext_match_second_max_percent_3)



    

#data in lists
#ext_class_index_list, ext_class_name_list, ext_match_first_max_percent_list, ext_match_second_max_percent_list = prediction_obj.predict_classes(ext_images_list,trained_model)
""


#y_pred_one_hot = prediction_obj.get_one_hot(prediction_list)
#y_test_one_hot = prediciton_obj.get_one_hot(ground_truth_list)

#from helper_functions import get_confusion_matrix
from helper_functions import plot_confusion_matrix

from sklearn.metrics import classification_report,confusion_matrix
import itertools

Y_pred = trained_model.predict(training_obj.X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
### if y_pred and y_test is a list:
for i in range(num_classes):
    y_pred = np.hstack((y_pred,i))
    values = [i]
    n_values = 64
    one_hot_correct = np.eye(n_values)[values]
    training_obj.y_test = np.vstack((training_obj.y_test,one_hot_correct))

#y_pred = model.predict_classes(X_test)
#print(y_pred)

# Compute confusion matrix
#confusion_matrix(ground_truth_list,prediction_list)
from constants import target_names_all
one_hot_correct_array = training_obj.y_test
print(np.argmax(one_hot_correct_array,axis=1)) #horizontal correct ans
predictions_vector = y_pred
#
cnf_matrix = (confusion_matrix(np.argmax(one_hot_correct_array,axis=1), predictions_vector))
#print(cnf_matrix)
#print(cnf_matrix.shape)




# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names_all,
                      title='Confusion matrix',dataset_PATH, "confusion_matrix_4")
#plt.figure()
# Plot normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
                      title='Normalized confusion matrix')
#plt.figure()



end = time.time()#record time
print('ComponentClassifierPredict done... Time Elapsed : '+ str(end-start) + ' seconds...')
t5 = end-start




# # Printing Results
# #### Results are plotted on the original image using:
#     - image: for background
#     - name: for saving
#     - ext_data_list: list of x, y, w, h coordinates for each bounding box
#     - ext_class_index_list: list of predicted class indices
#     - ext_class_name_list: list of corresponding class names per predicted index
#     - ground_truth_index_list: list of ground truth class indices (currently set as the predicted classes.)*
# ##### Each bounding box is labelled with two items separated by a colon:
#     - First: index of that object in the full list.
#     - Second: predicted class name of that object.
# ##### If the predicted class matches with the ground truth class, <br> then the predicted label is coloured green, otherwise it is red. <br> * For this example, the ground truth labels were set as the predicted classes, so all labels are green.

""

'''




























'''

#%%

start = time.time() # Begin time measurement

ground_truth_index_list = ext_class_index_list # set all answers as correct answers for now
plot_model_results_and_save(image,name, ext_data_list, ext_class_index_list, ext_class_name_list, ground_truth_index_list)

end = time.time()#record time
print('Printing Results done... Time Elapsed : '+ str(end-start) + ' seconds...')
t6 = end-start


# # ExtractionLabelling
# #### The ExtractionLabelling class is used to label the problem image with ground truths in the form of x, y, w, h, c <br>where x, y, w, h are the coordinates of the bounding box, and c is the class of the object in the box.
# #### The class is an interactive program where the user will be asked to verify the correctness <br> of the bounding box regions and classes predicted by the current best segmentation + classification algorithm.
# ##### The class is initialized using:
#     - PATH: working directory
#     - ext_images_list: list of extraction images
#     - ext_data_list: list of extraction image coordinates (x, y, w, h)
#     - ext_class_index_list: list of predicted class indices (c)
#     - ext_class_name_list: list of predicted class names
#     - num_classes, img_rows, img_cols: selected CNN model parameters
# ##### The define_model function is then called to load the model (from the trained_model variable). <br> Then the select_good_bounding_boxes function is called to allow the user to verify the bounding boxes and predicted classes. <br> Finally, after all the objects have been segmented and classified and saved in a text file, the data is plotted on the original image.
# ##### The process of the interactive program is described in the diagram below:
# <img src="https://justinsj.weebly.com/uploads/6/4/9/2/64923303/extraction-labelling-flowchart_orig.jpg" alt="Drawing" style="width: 800px;"/>
start = time.time() # Begin time measurement

labelling_obj = ExtractionLabelling(dataset_PATH,
                          ext_images_list, ext_data_list,ext_class_index_list, ext_class_name_list, 
                          num_classes, img_rows, img_cols)
labelling_obj = ExtractionLabelling(dataset_PATH, [],[],[],[],64,100,100)
#new_dataset_name = labelling_obj.update_answers_list(dataset_PATH, dataset_name,350,706,exclude=[23])

final_dataset_name = labelling_obj.clean_dataset(dataset_PATH,new_dataset_name)
#print(final_dataset_name)

labelling_obj.define_model(trained_model)
#labelling_obj.select_good_bounding_boxes(image, PATH+"GT/"+"easy_" + str(image_index))
labelling_obj.plot_ground_truths(labelling_obj.image, "all_2")

end = time.time()#record time
print('Acquiring and Printing Ground Truth Data done... Time Elapsed : '+ str(end-start) + ' seconds...')
t7 = end-start
# # Active Learning
# #### Ground Truth Data (from ExtractionLabelling) is added into the training dataset for improvement in accuracy
start = time.time() # Begin time measurement



end = time.time()#record time
print('Acquiring and Printing Ground Truth Data done... Time Elapsed : '+ str(end-start) + ' seconds...')
t8 = end-start
# # TestingClass



#%% 
'''
Test different sample sizes using Testing Class by loading data
'''
PATH = '/home/chloong/Desktop/Justin San Juan/Testing Folder/'

# image_set = np.load(PATH+'all_training_images.npy')
name = 'Sketch-a-Net_64_classes_100x100_0.0_all_100epochs'
  
data_set_name = 'Training_Samples_64_classes_100x100_all'
start = 20000 # smallest sample size to be tested
end =20000 # largest sample size to be tested
step = 1000 # steps in sample sizes
list_of_n = list(np.arange(start,end+1,step)) # list of sample sizes
k = 10 # number of times the samples size is tested
#list of_images = list(np.arange(0,200+1,5)) # list of images to be tested on
iters = 200 # number of iterations in testing, also used for file naming
testing_obj = TestingClass(PATH, wanted_w, wanted_h, export_w, export_h, max_piece_percent)


for n in list_of_n:
    for i in range(k):
        img_start = (i+6)*25
        img_end = img_start+150
        img_step = 5
        list_images = list(np.arange(img_start,img_end,img_step))
        
        testing_obj = TestingClass(PATH, wanted_w, wanted_h, export_w, export_h, max_piece_percent)
        gc.collect()
        seed = 1038+n+i*100
        testing_obj.test_classifier_remapped_load_3_models(PATH+data_set_name, 
                                    TRAINING_RATIO_TRAIN, i, 
                                    n, iters, list_images, seed)
        testing_obj=None
        gc.collect()

# # Time Cost Analysis
print('Loading image : ' + str(t1) + '\n'
      'ComponentSegmentation : ' + str(t2) + '\n'
      'ExtractionPreprocessing : ' + str(t3) + '\n'
      'ComponentClassifierTraining : ' + str(t4) + '\n'
      'ComponentClassifierPredict : ' + str(t5) + '\n'
      'Printing Results : ' + str(t6) + '\n'
      'Plot extractions with names : ' + str(t7) + '\n')###### OLD CODE ######



#%%
#'''
#Train 10 different models using different seeds
#'''
#
#PATH = '/home/chloong/Desktop/Justin San Juan/Testing Folder/'
#
## image_set = np.load(PATH+'all_training_images.npy')
#name = 'Sketch-a-Net_64_classes_100x100_0.0_all_100epochs'
#seed = 4581
#accuracies =[]
#for i in range(1):
#    print('BUILDING... '+str(i))
#    seed = int(random.random()*10000)
#    random.seed(seed)
#    data_set_name = 'Training_Samples_64_classes_100x100_all'
#    dropout = 0
#    training_obj = ComponentClassifierTraining(PATH, data_set_name, num_classes, dropout, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL)
#    training_obj.shuffle_data(training_obj.load_data(PATH,data_set_name),seed)
#    training_obj.model = training_obj.load_sketch_a_net_model(dropout, num_classes, training_obj.X_train.shape[1:])
#    
#    seed = int(random.random()*10000)
#    random.seed(seed)
#    
#    training_obj.model.load_weights(name+'_'+str(i))
#    training_obj.is_trained = True
#    training_obj.get_stats()
#    

#%%
'''
Load different models
'''

PATH = '/home/chloong/Desktop/Justin San Juan/Testing Folder/'

# image_set = np.load(PATH+'all_training_images.npy')
name = 'Sketch-a-Net_64_classes_100x100_0.0_all_100epochs'
seed = 4581
for i in range(30):
    print('BUILDING... '+str(i))
    seed = int(random.random()*10000)
    random.seed(seed)
    data_set_name = 'Training_Samples_64_classes_100x100_all'
    dropout = 0
    training_obj = ComponentClassifierTraining(PATH, data_set_name, num_classes, dropout, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL)
    training_obj.shuffle_data(training_obj.load_data(PATH,data_set_name),seed)
    training_obj.model = training_obj.load_sketch_a_net_model(dropout, num_classes, training_obj.X_train.shape[1:])
    
    seed = int(random.random()*10000)
    random.seed(seed)
    
    training_obj.train(200,seed)
    training_obj.model.load_weights(name+'_'+str(i)+'.h5')
#    training_obj.
#%%
'''
Test rotations
'''
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the image file")
args = vars(ap.parse_args())

# load the image from disk
image = cv2.imread(args["image"])
 
# loop over the rotation angles
for angle in np.arange(0, 360, 15):
	rotated = imutils.rotate(image, angle)
	cv2.imshow("Rotated (Problematic)", rotated)
	cv2.waitKey(0)
 
# loop over the rotation angles again, this time ensuring
# no part of the image is cut off
for angle in np.arange(0, 360, 15):
	rotated = imutils.rotate_bound(image, angle)
	cv2.imshow("Rotated (Correct)", rotated)
	cv2.waitKey(0)

image = np.asarray(  
        [[0,0,0,1,0,0,0,0,0,0],
         [0,0,1,1,0,0,0,1,1,0],
         [0,0,0,1,0,0,1,0,0,1],
         [0,0,0,1,0,0,0,1,1,1],
         [0,0,0,1,0,0,0,0,0,1],
         [0,0,0,1,0,0,0,0,0,1],
         [0,0,0,0,0,1,0,1,0,1],
         [0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0]])

    
#%%
PATH = '/home/chloong/Desktop/Justin San Juan/Testing Folder/'
src_im = Image.open(PATH + "sample01.JPG")
angle = 5
size = 1000, 1000


im = src_im
rot = im.rotate(angle, expand=1 )
rot.save(PATH + "test.png")

#%%

image = np.asarray(  
        [[0,0,0,1,0,0,0,0,0,0],
         [0,0,1,1,0,0,0,1,1,0],
         [0,0,0,1,0,0,1,0,0,1],
         [0,0,0,1,0,0,0,1,1,1],
         [0,0,0,1,0,0,0,0,0,1],
         [0,0,0,1,0,0,0,0,0,1],
         [0,0,0,0,0,1,0,1,0,1],
         [0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0]])
data = [(1,0,4,7),(5,0,5,8)]

extraction_obj = ExtractionPreprocessing(image, '', data)
gt_extraction_list, gt_extraction_data = extraction_obj.preprocess_extractions(wanted_w, wanted_h, export_w, export_h,
                                                max_piece_percent)
b = np.asarray(
        [[0,1],
         [1,1],
         [0,1],
         [0,1],
         [0,1],
         [0,1]])
c = np.asarray(
        [[0,0,0,0,0],
         [0,0,1,1,0],
         [0,1,0,0,1],
         [0,0,1,1,1],
         [0,0,0,0,1],
         [0,0,0,0,1],
         [0,0,1,0,1],
         [0,0,0,0,0]])