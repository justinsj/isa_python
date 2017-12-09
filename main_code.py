#%%
%load_ext autoreload
%autoreload 2
from __future__ import print_function
import numpy as np

from random import sample
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json
from keras.models import load_model
from keras import backend as K

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import filters
from skimage.filters import threshold_local


import selectivesearch

from skimage import measure
import cv2
import time

from component_segmentation import ComponentSegmentation
from extraction_preprocessing import ExtractionPreprocessing
from extraction_labelling import ExtractionLabelling
from component_classifier_training import ComponentClassifierTraining
from component_classifier_predict import ComponentClassifierPredict
from testing_class import TestingClass
from constants import target_names_all, target_names
#import helper_functions

print('Done Importing...')

###########################
##### HYPER-PARAMTERS #####
###########################

#change ALL CAPS for constants (hyperparamaters)
#selective search parameters
scale_input=10 #10
sigma_input=0 #15
min_size_input=5 #5

#noise reduction parameters
min_shape=40 #min. number of black pixels  
min_height=5 #min. height of bounding box
min_width=5 #min. width of bounding box

buffer_zone=2 #expand bounding box by this amount in all directions  
min_area=150 #min. area of bounding box
min_black=50 #min. number of black pixels
min_black_ratio=0.03 #min ratio of black pixels to the bounding box area

img_rows, img_cols = 100,100

#overlap parameters
overlap_repeats = 8 #set to 8
overlap_threshold = 0.3 #set to 0.3

#removing unconnected pieces parameters
max_piece_percent=0.3  # set to 0.3
min_percent_match = 0.7 # set to 0.7
min_confidence = 0.3 # set to 0.3
wanted_w, wanted_h, export_w, export_h = img_cols, img_rows, img_cols, img_rows

num_classes = 64
TRAINING_RATIO_TRAIN = 0.7
TRAINING_RATIO_VAL = 0.15

print('Done setting hyperparamters...')
#%%

start = time.time() # Begin time measurement

PATH = 'C:/Users/JustinSanJuan/Desktop/HKUST/UROP Deep Learning Image-based Structural Analysis/Code/Python/Testing Folder/'

# image_set = np.load(PATH+'all_training_images.npy')
name = 'Sketch-a-Net_64_classes_100x100_0.0_all_100epochs'
image_index = 371
image=np.load(PATH+'all_training_images.npy')[:,:,image_index]
#fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
#ax.imshow(image)
#plt.show()

############## Record time
end = time.time()
print('Loading image done... Time Elapsed : '+ str(end-start) + ' seconds...')
t1 = end-start
start = time.time()
##############

############## ComponentSegmentation ################
#Create object ComponentSegmentation, which will use the search function to perform segmentation and merging.
segmentation_obj = ComponentSegmentation(image, name, 
                            min_shape, min_height, min_width, 
                            buffer_zone, min_area, min_black, min_black_ratio,
                            overlap_repeats, overlap_threshold)

segmentation_obj.search(scale_input, sigma_input, min_size_input) # run search (segmentation code)
merged_set = segmentation_obj.merged_set
############### ExtractionPreprocessing ##################

#Transport data into ExtractionPreprocessing class, which will trim, remove unconnected parts, then trim, and resize
extraction_obj = ExtractionPreprocessing(image, name, merged_set)

# Get 4 lists from preprocess_extractions function
ext_images, ext_data = extraction_obj.preprocess_extractions(wanted_w, wanted_h, export_w, export_h, max_piece_percent)
#obj.plot_bounding_boxes_with_names()

########### ComponentClassifierTraining ################
data_set_name = 'Training_Samples_64_classes_100x100_all'
dropout = 0
training_obj = ComponentClassifierTraining(PATH, data_set_name, num_classes, dropout, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL)
#training_obj.train(1)
training_obj.model.load_weights(PATH+name+'.h5')

trained_model = training_obj.model
#training_obj.save(PATH+name)
############### ComponentClassifierPredict ################
prediction_obj = ComponentClassifierPredict(min_percent_match, min_confidence)

ext_class_index, ext_class_name, \
ext_match_first_max_percent, \
ext_match_second_max_percent = prediction_obj.predict_classes(ext_images,trained_model)

labelling_obj = ExtractionLabelling(PATH,
                          ext_images, ext_data,ext_class_index, ext_class_name, 
                          num_classes, img_rows, img_cols)

labelling_obj.define_model(trained_model)
labelling_obj.select_good_bounding_boxes(image, "all_" + str(image_index))
labelling_obj.plot_ground_truths(image, "all_" + str(image_index))





#%%   



 #%%
    # Create/reset list of images, coordinates (x,y,w,h) data, class indices, class names, and top three percentage matches
ext_images=[]
ext_data=[]
ext_class_index=[]
ext_class_name=[]
ext_match_percent=[]
ext_match_percent2=[]
ext_next_round=[]
ext_next_round_index=[]

    # define wanted_w, and wanted_h, which is the are where the extraction is limited to
    # define export_w and export_h as required by the classifier
wanted_w=img_cols
wanted_h=img_rows
export_w=img_cols
export_h=img_rows

    # prepare extractions to be sent to classifier
    # ext_class_index and _name are empty

name = 'Sketch-a-Net_Single_Model_'+str(t)

##############
end = time.time()
print('Segmentation done... Time Elapsed : '+str(end-start)+' seconds...')
t2=end-start
start = time.time()
##############
name = 'TestGaussian'+'_'+str(num_classes)+'_classes_'+str(img_rows)+'x'+str(img_cols)+'_'+str(a)#+'_gaussian'

    # plot bounding boxes on original image
plot_bounding_boxes_with_names(image,candidates, name) 

##############
end = time.time()
print('Plotting bounding boxes done... Time Elapsed : '+str(end-start)+' seconds...')
t3=end-start
start = time.time()
##############
    #load data only if not yet loaded, or update data if number of samples in data_all does not match current y_train number of data samples
try: y_train
except: 
    try:
        data_all=np.load('Training_Samples_'+str(num_classes)+'_classes_'+str(img_rows)+'x'+str(img_cols)+'_all.npy')
        if y_train.shape[0]==data_all.shape[0]:
            y_train=y_train
        else:
            x_train, y_train, x_test, y_test,input_shape,data_all=load_data()
    except:
        x_train, y_train, x_test, y_test,input_shape,data_all=load_data()
#data_all=np.load('Training_Samples_'+str(num_classes)+'_classes_'+str(img_rows)+'x'+str(img_cols)+'_all.npy')
#x_train, y_train, x_test, y_test,input_shape,data_all=load_data()

##############
end = time.time()
print('Loading training data done... Time Elapsed : '+str(end-start)+' seconds...')
t4=end-start
start = time.time()
##############

epochs=100
group='all'
    #list of dropout values to be tested
#di=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
di=[0.0]
for d in di:

    name = 'Sketch-a-Net'+'_'+str(num_classes)+'_classes_'+str(img_rows)+'x'+str(img_cols)+'_'+str(d)#+'_gaussian'
    print('name = '+name)
    model = Sequential() #model needs to be defined as a global variable before using load_model_layers, train_model, or load_model_weights
    
    load_model_layers(d)
#    train_model(epochs)
#    save_model_weights(name,epochs)
    print('loading model weights...')
    load_model_weights(name,epochs)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1]) 
    name = 'TestGaussian'+'_'+str(num_classes)+'_classes_'+str(img_rows)+'x'+str(img_cols)+'_'+str(d)+'_'+str(a)#+'_gaussian'

    ##############
    end = time.time()
    print('Loading/training model done... Time Elapsed : '+str(end-start)+' seconds...')
    t5 = end-start
    start = time.time()
    ##############
    
    print('predicting classes...')
    predict_classes(ext_images,group,ext_class_index,ext_class_name,ext_next_round,ext_next_round_index)
    
    ##############
    end = time.time()
    print('predict classes done... Time Elapsed : '+str(end-start)+' seconds...')
    t6 = end-start
    start = time.time()
    ##############
    
   
        #create figure with all extractions and percent matches if no answers
    
    select_good_bounding_boxes(image,imagename,ext_images,ext_data,ext_class_index,ext_class_name,target_names)
    
    print('plotting extractions with names...')
#    plot_extractions_with_names(ext_images, ext_data, ext_class_name, ext_class_index, name) 
#    plot_extractions_with_names(ext_images, ext_data, ext_class_name, ext_class_index, name, ans = adjusted_ans) 

    ##############
    end = time.time()
    print('plot extractions with names done... Time Elapsed : '+str(end-start)+' seconds...')
    t7 = end-start
    start = time.time()
    ##############
    #update answers if necessary
#update_answers(ext_images,ans) 

print('Loading image : ' + str(t1) + '\n'
      'Segmentation : ' + str(t2) + '\n'
      'Plot bounding boxes : ' + str(t3) + '\n'
      'Load training data : ' + str(t4) + '\n'
      'Load/train model : ' + str(t5) + '\n'
      'Predict classes : ' + str(t6) + '\n'
      'Plot extractions with names : ' + str(t7) + '\n')