#%%
from __future__ import print_function
import numpy as np

from random import sample
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import matplotlib.pyplot as plt



from skimage import filters
from skimage.filters import threshold_local

import matplotlib.patches as mpatches
import selectivesearch

from skimage import measure
import cv2
import time
print('Done Importing...')
#%%
###########################
##### HYPER-PARAMTERS #####
###########################


#change ALL CAPS for constants (hyperparamaters)
#selective search parameters
scale_input=10 #10
sigma_input=15 #15
min_size_input=5 #5

#noise reduction parameters
min_shape=50 #min. number of black pixels  
min_height=5 #min. height of bounding box
min_width=5 #min. width of bounding box

buffer_zone=2 #expand bounding box by this amount in all directions  
min_area=200 #min. area of bounding box
min_black=100 #min. number of black pixels
min_black_ratio=0.05 #min ratio of black pixels to the bounding box area



#overlap parameters
overlap_repeats = 8 #set to 8
overlap_threshold = 0.3 #set to 0.3

#removing unconnected pieces parameters
max_piece_percent=0.30

print('Done setting hyperparamters...')

#%%
##### LOAD IMAGE SET ##### (Only need to load once)
set_num=1
image_set = np.load('/home/chloong/Desktop/Justin San Juan/Testing Folder/all_training_images_'+str(set_num)+'.npy')
#%%
        ######################### TEST FULL CODE ON SAMPLE ######################
        
####### Load hand-drawn image #########
#al=np.linspace(1.2,3,int((3-1.2)/0.1+1))
#al=[2.4] # to test different gaussian filters
a=2.4

e = int(input("Which image?"+'\n'))


imagename='all_'+str(e+(set_num-1)*400)

#    image = cv2.imread('/home/chloong/Desktop/Justin San Juan/Testing Folder/'+imagename+'.JPG')
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    image = np.array(image)
#    image = binarize_image(image,a)
start = time.time()

image=image_set[:,:,e]

#current samples answers only work with this loading method of hand-drawn image
#test_image = cv2.imread('/home/chloong/Desktop/Justin San Juan/Testing Folder/Previous Data/samples1.jpg')
#
#test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
#test_image = np.array(test_image)
#test_image = test_image.astype('float32')
#test_image /= 255
#test_image = 1 - test_image # invert image since in this code, 0 is set to white areas, 1 is black area, since majority of images should be white, so more 0s and less written data
#test_image = 1.48 * test_image # amplify image to prevent loss of data from blur
#test_image = np.round(test_image)
#image=test_image

#image = np.load('/home/chloong/Desktop/Justin San Juan/Testing Folder/easy_training_images.npy')[:,:,6]
##############
end = time.time()
print('Loading image done... Time Elapsed : '+str(end-start)+' seconds...')
t1 = end-start
start = time.time()
##############0.3
    # reset sets and lists
good_candidates = set()
good_candidates_list=[]
bad_candidates = set()

    # get good_candidates set from search function, and also put it into 'candidates'
search(image, good_candidates, bad_candidates, scale_input, sigma_input, min_size_input)
candidates=good_candidates # select candidates to be classified, can be only good, only bad, or both sets of candidates
    #full class 1-60 list (index 0 to 59) available at: https://drive.google.com/open?id=1AlqCo-44dX59BYiu5-u0Dk6uDpNi89PI21rgJE1kfts
    # define merged class names

    
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
ext_images, ext_data, ext_class_index, ext_class_name = preprocess_extractions(image,candidates, wanted_w, wanted_h, export_w, export_h) 

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