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

# In[ ]:


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


# In[ ]:


start = time.time() # Begin time measurement

seed = 1000

weights_name = "Sketch-A-Net_controlled_600_30858_7_layers"
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
                                     200,seed,350,706, min_percent_match, min_confidence, 
                                     model_1_weights_name = weights_name,model_7_layers = True, exclude = [])

f = open(dataset_PATH+'testing_results.txt','a')
f.writelines('ground_truth_list_8 = '+str(ground_truth_list) +'\n'
             'prediction_list_8 = '+str(prediction_list)+'\n')
f.close()

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

f = open(dataset_PATH+'testing_results.txt','a')
f.writelines('ground_truth_list_6 = '+str(ground_truth_list) +'\n'
             'prediction_list_6 = '+str(prediction_list)+'\n')
f.close()

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

f = open(dataset_PATH+'testing_results.txt','a')
f.writelines('ground_truth_list_7 = '+str(ground_truth_list) +'\n'
             'prediction_list_7 = '+str(prediction_list)+'\n')
f.close()

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




