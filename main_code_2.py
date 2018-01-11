
# coding: utf-8

# # Image-Based Structural Analysis
# ### Code created and maintained by Justin David Q. SAN JUAN, <br>email: jdqsj1997@yahoo.com, <br> personal website: justinsj.weebly.com
# 
# #### This code focuses in the segmentation and classification processes (except reconstruction) of the complete project pipeline as described below:
# <img src="https://justinsj.weebly.com/uploads/6/4/9/2/64923303/process-flowchart_orig.jpg" alt="Drawing" style="width: 800px;"/>

# # Import Dependencies
# #### Dependencies:
# numpy: for handling data types (mostly handled as numpy arrays)<br>
# Sequential (from keras.models): for CNN setup<br>
# random: for pseudo-random shuffling of data<br>
# cv2: for raw RBG image import and transformation to grayscale<br>
# time: for measuring time elapsed per function<br>
# ##### Custom Classes:
# ComponentSegmentation: for proposing regions of interest (RoI's)<br>
# ExtractionPreprocessing: for trimming, noise removal, and resizing of image<br>
# ComponentClassifierTraining: for loading the CNN model, training data, and training the model<br>
# ComponentClassifierPredict: for using the CNN model to predict the class of preprocessed extractions<br>
# ExtractionLabelling: for labelling ground truth bounding boxes and classes in problem images<br>
# TestingClass: for testing the accuracy of a CNN model on the problem images<br>
# <br>
# print_image_bw is used to print a simple 2-D array<br>
# gc: for clearing up space after acquiring data from larger datasets

# In[1]:


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
from helper_functions import print_time_string, store_time

import gc
gc.enable()

print('Done Importing...')


# # Hyper-parameters
# #### Selective Search Parameters:
# scale_input<br>
# sigma_input<br>
# min_size_input<br>
# #### Noise Reduction Parameters:
# min_shape: for minimum number of black pixels in bounding box<br>
# min_height: for minimum height of bounding box<br>
# min_width: for minimum width of bounding box<br>
# <br>
# buffer_zone: for expanding bounding box all directions<br>
# min_area: for minimum area of bounding box<br>
# min_black: for minimum number of black pixels in bounding box<br>
# min_black_ratio: for minimum ratio of black pixels to the bounding box area<br>
# #### Overlap Parameters:
# overlap_repeats: for number of iterations for merging algorithm to be applied<br>
# overlap_threshold: threshold of area overlap over union area for merging to be applied<br>
# #### Removing Unconnected Pieces Parameters:
# max_piece_percent: maximum percentage of piece to be removed<br>
# (if percentage is larger, piece will not be removed as it is more likely an important piece)<br>
# #### Extractions Preprocessing Parameters:
# img_rows, img_cols: for classifier input shape<br>
# wanted_w, wanted_h: for black pixels edges resizing boundary shape<br>
# export_w, export_h: for overall image resizing shape ([export_w-wanted_w]/2 = horizontal buffer on each side)<br>
# #### CNN Training Parameters:
# num_classes: number of classes for classifier to predict<br>
# TRAINING_RATIO_TRAIN: ratio of training samples to total number of samples<br>
# TRAINING_RATIO_VAL: ratio of validation samples to total number of samples<br>
# TRAINING_RATIO_TEST: ratio of test samples to total number of samples <br>
# Note: TRAINING_RATIO_TEST is implicitly calculated as [1-{TRAINING_RATIO_TRAIN + TRAINING_RATIO_VAL}]<br>
# dropout: dropout value to be used in all layers except last layer of Sketch-A-Net CNN model<br>
# #### CNN Prediction Parameters:
# min_percent_match: minimum probability of class prediction for that class to be set as the prediction<br>
# min_confidence: minimum difference between first-highest % match and second-highest % match<br>
# (higher difference means less ambiguity between first and second highest match, which means less likelihood of random object)<br>
# ##### The directory is also defined in the PATH variable.<br>The name of the CNN model data is defined in the name variable.<br>The training data set name for the CNN is defined in the data_set_name variable.

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
min_percent_match = 0 # set to 0.7
min_confidence = 0 # set to 0.3

#Time Cost parameters
time_cost_string_list = ['Loading image','Component Segmentation','Extraction Preprocessing',
                         'Component Classifier Training','Component Classifier Predict',
                        'Printing Results','Acquiring and Printing Ground Truth Data',
                        'Data Concatenation & Cleaning','Data Control, Counting, & Training from Multiple Files',
                        'Testing','Printing Confusion Matrix']
time_cost_time_list = np.zeros(len(time_cost_string_list)).astype(np.int).tolist()

#Paths and names
PATH = 'C:/Users/JustinSanJuan/Desktop/Workspace/python/Testing Folder/' #must have "/" at the end

name = 'Sketch-a-Net_64_classes_100x100_0.0_all_100epochs'

base_dataset_name = 'Training_Samples_64_classes_100x100_all'

dataset_PATH = 'C:/Users/JustinSanJuan/Desktop/HKUST/UROP Deep Learning Image-based Structural Analysis/Code/Python/Testing Folder/'
dataset_name = 'Training_Samples_64_classes_100x100_all_cleaned_32898'
new_dataset_name = 'Training_Samples_64_classes_100x100_all_cleaned_32898'

print('Done setting hyperparamters...')


# # Print Confusion Matrix

# # dataset_2 first controlled to 600 max

# In[ ]:


start = time.time() # Begin time measurement

seed = 1000

weights_name = "Sketch-A-Net_controlled_600_30858"
#weights_name = "Training_Samples_64_classes_100x100_all_cleaned_updated_29739+7500(0-350)"
#weights_name = dataset_name
#dataset_name_1 = "Training_Samples_64_classes_100x100_all_cleaned_updated_29739"
#dataset_name_2 = "Training_Samples_64_classes_100x100_all_cleaned_updated_7500_0-350"
dataset_name_list = ["Training_Samples_64_classes_100x100_all_controlled_30858"]

### Long procedure
testing_obj = TestingClass(dataset_PATH, wanted_w, wanted_h, export_w, export_h, max_piece_percent)
# Slower does testing with as little memory at all times as possible
ground_truth_list, prediction_list = testing_obj.test_classifier_multiple_slow(dataset_PATH, dataset_name_list,
                                     num_classes,dropout, 
                                     TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL,
                                     200,seed,350,706, weights_name = weights_name)

end = time.time()#record time
time_cost_time_list = store_time(9,time_cost_time_list,end-start)
print_time_string(9,time_cost_string_list,time_cost_time_list)

start = time.time() # Begin time measurement

confusion_matrix_index = 1

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

end = time.time()#record time
time_cost_time_list = store_time(10,time_cost_time_list,end-start)
print_time_string(10,time_cost_string_list,time_cost_time_list)


# # dataset_2 first controlled to 600 max

# In[ ]:


start = time.time() # Begin time measurement

seed = 1000

weights_name = "Sketch-A-Net_controlled_600_30858_1"
#weights_name = "Training_Samples_64_classes_100x100_all_cleaned_updated_29739+7500(0-350)"
#weights_name = dataset_name
#dataset_name_1 = "Training_Samples_64_classes_100x100_all_cleaned_updated_29739"
#dataset_name_2 = "Training_Samples_64_classes_100x100_all_cleaned_updated_7500_0-350"
dataset_name_list = ["Training_Samples_64_classes_100x100_all_controlled_30858_1"]

### Long procedure
testing_obj = TestingClass(dataset_PATH, wanted_w, wanted_h, export_w, export_h, max_piece_percent)
# Slower does testing with as little memory at all times as possible
ground_truth_list, prediction_list = testing_obj.test_classifier_multiple_slow(dataset_PATH, dataset_name_list,
                                     num_classes,dropout, 
                                     TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL,
                                     200,seed,350,706, weights_name = weights_name)

end = time.time()#record time
time_cost_time_list = store_time(9,time_cost_time_list,end-start)
print_time_string(9,time_cost_string_list,time_cost_time_list)

start = time.time() # Begin time measurement

confusion_matrix_index = 1

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

end = time.time()#record time
time_cost_time_list = store_time(10,time_cost_time_list,end-start)
print_time_string(10,time_cost_string_list,time_cost_time_list)


# # Exclude 23's

# In[ ]:


start = time.time() # Begin time measurement

seed = 1000

weights_name = "Sketch-A-Net_exclude_23_32898"
#weights_name = "Training_Samples_64_classes_100x100_all_cleaned_updated_29739+7500(0-350)"
#weights_name = dataset_name
#dataset_name_1 = "Training_Samples_64_classes_100x100_all_cleaned_updated_29739"
#dataset_name_2 = "Training_Samples_64_classes_100x100_all_cleaned_updated_7500_0-350"
dataset_name_list = ["Training_Samples_64_classes_100x100_all_cleaned_32898"]

### Long procedure
testing_obj = TestingClass(dataset_PATH, wanted_w, wanted_h, export_w, export_h, max_piece_percent)
# Slower does testing with as little memory at all times as possible
ground_truth_list, prediction_list = testing_obj.test_classifier_multiple_slow(dataset_PATH, dataset_name_list,
                                     num_classes,dropout, 
                                     TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL,
                                     200,seed,350,706, weights_name = weights_name)

end = time.time()#record time
time_cost_time_list = store_time(9,time_cost_time_list,end-start)
print_time_string(9,time_cost_string_list,time_cost_time_list)

start = time.time() # Begin time measurement

confusion_matrix_index = 1

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

end = time.time()#record time
time_cost_time_list = store_time(10,time_cost_time_list,end-start)
print_time_string(10,time_cost_string_list,time_cost_time_list)


# # Time Cost Analysis

# time_cost_string = ''
# for i in range(len(time_cost_string_list)):
#     time_cost_string += time_cost_string_list[i] +' : ' + time_cost_time_list[i]+'\n'
# print(time_cost_string)

# In[ ]:





# In[ ]:




###### OLD CODE ######



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
    training_obj.
    
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