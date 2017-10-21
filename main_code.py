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

print('Done Importing...')
#%%
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
max_piece_percent=0.30

wanted_w, wanted_h, export_w, export_h = img_cols, img_rows, img_cols, img_rows

print('Done setting hyperparamters...')

#%%

start = time.time()

PATH = 'C:/Users/JustinSanJuan/Desktop/HKUST/UROP Deep Learning Image-based Structural Analysis/Code/Python/Testing Folder/'
image_set = np.load(PATH+'easy_training_images.npy')
name = 'sample_1_test'

image=image_set[:,:,6]


##############
end = time.time()
print('Loading image done... Time Elapsed : '+str(end-start)+' seconds...')
t1 = end-start
start = time.time()
##############

sample_test = ComponentSegmentation(image, name, 
                                      min_shape, min_height, min_width, 
                                      buffer_zone, min_area, min_black, min_black_ratio,
                                      overlap_repeats, overlap_threshold)

sample_test.search(scale_input, sigma_input, min_size_input)

merged_set = sample_test.merged_set # to be passed on to ExtractionPreprocessing class

sample_preprocessed = ExtractionPreprocessing(image, name, merged_set)

#get 4 lists from preprocess_extractions function
ext_images, ext_data, ext_class_index, ext_class_name = sample_preprocessed.preprocess_extractions(wanted_w, wanted_h, export_w, export_h,
                                                max_piece_percent)
sample_preprocessed.plot_bounding_boxes_with_names()



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