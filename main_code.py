#%%
#%load_ext autoreload
#%autoreload 2
from __future__ import print_function
import numpy as np

from keras.models import Sequential

import random

import cv2
import time

from component_segmentation import ComponentSegmentation
from extraction_preprocessing import ExtractionPreprocessing
from extraction_labelling import ExtractionLabelling
from component_classifier_training import ComponentClassifierTraining
from component_classifier_predict import ComponentClassifierPredict
from testing_class import TestingClass

import gc
gc.enable()

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
min_percent_match = 0 # set to 0.7
min_confidence = 0 # set to 0.3
wanted_w, wanted_h, export_w, export_h = img_cols, img_rows, img_cols, img_rows

num_classes = 64
TRAINING_RATIO_TRAIN = 0.7
TRAINING_RATIO_VAL = 0.15

print('Done setting hyperparamters...')
#%%
start = time.time() # Begin time measurement
#print('Beginning loading of image...')

PATH = '/home/chloong/Desktop/Justin San Juan/Testing Folder/'

# image_set = np.load(PATH+'all_training_images.npy')
name = 'Sketch-a-Net_64_classes_100x100_0.0_all_100epochs'

image_index = 0
image_set = np.load(PATH+'all_training_images.npy')
image = np.copy(image_set[:,:,image_index])
image_set = None #clear image_set
#print('clearing unreferenced data...')
#gc.collect() #clear unused/unreferenced image_set data
##fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
##ax.imshow(image)
##plt.show()
#
############### Record time
#end = time.time()
#print('Loading image done... Time Elapsed : '+ str(end-start) + ' seconds...')
#t1 = end-start


#'''
start = time.time()
##############

############## ComponentSegmentation ################
#Create object ComponentSegmentation, which will use the search function to perform segmentation and merging.
segmentation_obj = ComponentSegmentation(image, name, min_shape, min_height, min_width, buffer_zone, min_area, min_black, min_black_ratio, overlap_repeats, overlap_threshold)
segmentation_obj.search(scale_input, sigma_input, min_size_input) # run search (segmentation code)
merged_set = segmentation_obj.merged_set
############### ExtractionPreprocessing ##################

#Transport data into ExtractionPreprocessing class, which will trim, remove unconnected parts, then trim, and resize
extraction_obj = ExtractionPreprocessing(image, name, merged_set)

# Get 4 lists from preprocess_extractions function
ext_images, ext_data = extraction_obj.preprocess_extractions(wanted_w, wanted_h, export_w, export_h, max_piece_percent)
#extraction_obj.plot_bounding_boxes_with_names()
#'''


########### ComponentClassifierTraining ################
data_set_name = 'Training_Samples_64_classes_100x100_all'
dropout = 0
training_obj = ComponentClassifierTraining(PATH, data_set_name, num_classes, dropout, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL)
training_obj.shuffle_data(training_obj.load_data(PATH,data_set_name),1000)
training_obj.model = training_obj.load_sketch_a_net_model(dropout, num_classes, training_obj.X_train.shape[1:])

training_obj.train(100)
training_obj.save(name+'_'+str(i))



#training_obj.model.load_weights(PATH+name+'.h5')
#
#
#trained_model = training_obj.model
#
################ ComponentClassifierPredict ################
#prediction_obj = ComponentClassifierPredict(min_percent_match, min_confidence)
#
#ext_class_index, ext_class_name, \
#ext_match_first_max_percent, \
#ext_match_second_max_percent = prediction_obj.predict_classes(ext_images,trained_model)
##'''
#labelling_obj = ExtractionLabelling(PATH,
#                          ext_images, ext_data,ext_class_index, ext_class_name, 
#                          num_classes, img_rows, img_cols)
#
##labelling_obj.define_model(trained_model)
##labelling_obj.select_good_bounding_boxes(image, "all_" + str(image_index))
#labelling_obj.plot_ground_truths(image, "all_" + str(image_index))

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
from PIL import Image
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