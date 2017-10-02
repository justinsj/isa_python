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
from scipy import interpolate
from adjustText import adjust_text
import matplotlib.patheffects as PathEffects

import matplotlib.ticker as ticker
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

t = 12 #Version number

#training parameters
batch_size = 200
num_classes = 64
epochs = 300
training_ratio = 0.7
img_rows, img_cols = 100, 100

#overlap parameters
overlap_repeats = 8 #set to 8
overlap_threshold = 0.3 #set to 0.3

#removing unconnected pieces parameters
max_piece_percent=0.30

#final answer parameters
min_percent_match = 0 # set to 0.7 possiby set to 0.5
min_confidence = 0 # set to 0.6 possibly set to 0.3

print('Done setting hyperparamters...')
#%%
    #function: split dataset randomly
def random_split_dataset(data_set, training_ratio):
       l = data_set.shape[0]
       f = int(l * training_ratio)
       train_indices = sample(range(l),f)
       test_indices = np.delete(np.array(range(0, l)), train_indices)
       train_data = data_set[train_indices]
       test_data = data_set[test_indices]
       x_train = train_data[:,:-1]
       y_train = train_data[:,(-1)]
       y_train=y_train.reshape(y_train.shape[0],1)
       print(x_train.shape)
       x_test = test_data[:,:-1]
       y_test = test_data[:,(-1)]
       y_test = y_test.reshape(y_test.shape[0],1)
       return x_train, y_train, x_test, y_test
   
   #function: load pre-shuffled data
def load_data():
    data_all = np.load('/home/chloong/Desktop/Justin San Juan/Testing Folder/'+'Training_Samples_'+str(num_classes)+'_classes_'+str(img_rows)+'x'+str(img_cols)+'_all'+'.npy')
    train_data=np.load('/home/chloong/Desktop/Justin San Juan/Testing Folder/'+'Training_Samples_'+str(num_classes)+'_classes_'+str(img_rows)+'x'+str(img_cols)+'_'+'train_data.npy')
    test_data=np.load('/home/chloong/Desktop/Justin San Juan/Testing Folder/'+'Training_Samples_'+str(num_classes)+'_classes_'+str(img_rows)+'x'+str(img_cols)+'_'+'test_data.npy')
    x_train = train_data[:,:-1]
    y_train = train_data[:,(-1)]
    y_train=y_train.reshape(y_train.shape[0],1)
    x_test = test_data[:,:-1]
    y_test = test_data[:,(-1)]
    y_test = y_test.reshape(y_test.shape[0],1)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
        
        # Reshape back to 3D matrix to be passed into CNN
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)
    
        # Necessary transformation
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
        
        #change data type    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
        # Preparation and training of neural network\n",
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
        
        # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print('Loaded dataset')
    
    return x_train, y_train, x_test, y_test, input_shape, data_all
   

    #function: use random_split_dataset and shuffle data in batches (small due to memory error)
def load_data_in_batches(batch_size):
    data_all = np.load('/home/chloong/Desktop/Justin San Juan/Testing Folder/'+'Training_Samples_'+str(num_classes)+'_classes_'+str(img_rows)+'x'+str(img_cols)+'_all.npy')
    print(data_all.shape)
        # for X batches
    for j in range(0,int(np.ceil(data_all.shape[0]/batch_size))):
        print(int(np.ceil(data_all.shape[0]/batch_size)))
            #if first batch, don't stack
        if j ==0:
            try:
                data_set=data_all[j*batch_size:(j+1)*batch_size,:]
            except:
                data_set=data_all[j*batch_size:int(data_all.shape[0]),:]
            x_train, y_train, x_test, y_test = random_split_dataset(data_set, training_ratio)
            #if second or further batch, stack with previous data
        else:
            try:
                data_set=data_all[j*batch_size:(j+1)*batch_size,:]

            except:
                data_set=data_all[j*batch_size:int(data_all.shape[0]),]
            x_train1, y_train1, x_test1, y_test1 = random_split_dataset(data_set, training_ratio)
            x_train = np.vstack((x_train,x_train1))
            x_test = np.vstack((x_test,x_test1))
            print(x_train1.shape)
            print(x_train.shape)

            y_train = np.vstack((y_train,y_train1))
            y_test = np.vstack((y_test,y_test1))
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
        
        # Reshape back to 3D matrix to be passed into CNN
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)
    
        # Necessary transformation
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
        
        #change data type    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
        # Preparation and training of neural network\n",
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
        
        # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print('Loaded dataset')
    
    return x_train, y_train,x_test,y_test, input_shape, data_all
    
    #function: load Sketch-A-Net keras model layers ***model must have been declared as a global variable
def load_model_layers(d):
    #L1
    model.add(Conv2D(64, (15,15),strides=3, activation='relu',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3),strides=1))
    model.add(Dropout(d))
    #L2
    model.add(Conv2D(128, (5,5),strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
    model.add(Dropout(d))
    #L3
    model.add(Conv2D(256, (3,3),strides=1,padding='same', activation='relu'))
    model.add(Dropout(d))
    #L4
    model.add(Conv2D(256, (3,3),strides=1,padding='same', activation='relu'))
    model.add(Dropout(d))
    #L5
    model.add(Conv2D(256, (3,3),strides=1,padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
    model.add(Dropout(d))
    #L6
    model.add(Conv2D(512, (5,5),strides=1, activation='relu'))
    model.add(Dropout(0.5))
    #L7
    model.add(Conv2D(512, (1,1),strides=1, activation='relu'))
    model.add(Dropout(0.5))
    #L8
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    
    model.summary()
    model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

    #function: train model with input number of epochs ***model must have been declared as a global variable
def train_model(epochs):
    model.fit(x_train, y_train, batch_size=batch_size,
                                epochs=epochs,
                                verbose=1,
                                validation_data=(x_test, y_test))
    
    #function: save model weights ***model must have been declared as a global variable
def save_model_weights(name,epochs):
    model.save_weights(name + '_all_'+str(epochs)+"epochs"+".h5")
    print('saved model weights as '+name + "_" + str(epochs)+"epochs"+".h5")
    
    #function: load model weights from file ***model must have been declared as a global variable
def load_model_weights(name,epochs):
    model.load_weights(name + "_all_"+ str(epochs)+"epochs"+".h5")
    print('loaded model weights from '+name + "_" + str(epochs)+"epochs"+".h5")

    #function: use connectivity and selective search to create candidate bounding boxes for classification
def search(image, good_candidates, bad_candidates, _scale, _sigma, _min_size):

        # Label the matrix with different connected components
    labeled_matrix, num_cropped = measure.label(image, background=0, connectivity=1, return_num=True)
    used_set=set()

        # decleare temp_set as an empty list
    temp_set=[]
    
        # Loop through all connected components
        # range list is different in python. numbers from 1 to 5 is range(1, 5+1), (last number is not included)
    for i in range(1, num_cropped + 1):
        
            # Get the coordinates of current labels
        x = np.array(np.where(labeled_matrix == i))

            # Eliminate case of noise, tuneable
            # 'continue' skips everything under the if statement in the for loop
        if x.shape[1] < min_shape: continue
       
            # We have down > up and right > left # To find corners of image
        up = x[0][0]
        down = x[0][-1]
        left = np.amin(x[1])
        right = np.amax(x[1])


            # Essential if there is noise, because it will be counted as one conencted component
        if down - up < min_height or right - left < min_width: continue

            # Buffering zone: 2 (To exapnd image), tuneable
            # Crop the image of current connected component with buffer
        cropped = image[up-buffer_zone:down+buffer_zone, left-buffer_zone:right+buffer_zone]

            # Convert to RGB --> selective search requires RGB
        temp = np.zeros([cropped.shape[0], cropped.shape[1], 3])
        temp[:, :, 0] = cropped
        temp[:, :, 1] = cropped
        temp[:, :, 2] = cropped
        cropped = temp
        
        cropped_ls = [left, up, right - left + 1, down - up + 1]
        temp_set.append(tuple(cropped_ls))
  
            # perform selective search
        img_lbl, regions = selectivesearch.selective_search(cropped, scale=_sigma, sigma=_sigma, min_size=_min_size)
        
            # each r in regions is a dictionary (rect: x, y, w, h; size: n ...)
        for r in regions:

            # exclude regions smaller than min_area pixels, tuneable
            if r['size'] < min_area:
                continue
            
                # get number of pixels in each connected component and store in black_spot
            x, y, w, h = r['rect']
            x1 = x + left
            x2 = x + w + left - 1
            y1 = y + up
            y2 = y + h + up - 1

            cropped_region = image[y1:y2, x1:x2] 
            black_spot = np.array(np.where(cropped_region == 1))
            
                # filter those with very few black dots (noise)
            if black_spot.shape[1] < min_black: continue
            if float(black_spot.shape[1]) / (w * h) < min_black_ratio: continue

                
            ls = list(r['rect'])
            ls[0] = ls[0] + left
            ls[1] = ls[1] + up
                
                #add ls as list into temp_set, which will be merged based on overlap
            temp_set.append(ls)

########### #loop merging algorithm
    for o in range(overlap_repeats+1):
        
            # convert list of lists into matrix to do column swaps
        temp_set=np.asmatrix(temp_set)
        temp_set1=np.copy(temp_set)

        if temp_set.shape[1] <1:
            continue
        temp=np.copy(temp_set[:,1])
            
            # modify matrix from [x,y,w,h] into [x1,x2,y1,y2]
            # change x2 (1) of new data set into x+w
        temp_set[:,1]=temp_set[:,0]+temp_set[:,2]
            # change y2 (3) of new data set into y+h
        temp_set[:,3]=temp+temp_set[:,3]
        
        temp_set[:,2]=temp
        
            # create empty used set, to be filled with previously checked bounding boxes 
            # so they will not be re-added or re-removed into good_candidates
        used_set=set()
        skip=False
        
            # cycle through rows 'u' of [x1,x2,y1,y2] matrix
        for u in range(temp_set.shape[0]):
            
                # skip variable is set to true once row u has been used to merge, so it will not be reused
            if skip==True:
                skip=False
                continue
            
            # set test as [x1,x2,y1,y2] data in row u
            test=temp_set[u,:]
            
                # check if row u overlaps with other rows n in matrix
                # cycle through rows 'n' of [x1,x2,y1,y2] matrix
            for n in range(temp_set.shape[0]):
                
                    # skip if checking same row. if checking different row, check for overlap
                if u != n:
                    
                        # check if region n is outside of region u
                        # xi' is the xi of the fixed test [x1,x2,y1,y2]
                        # if x1 is greater than x2', or x2 is less than x1' ,then same with y direction
                        # if this statement is passed, it means u does not overlap with n
                    if temp_set[n,0] > test[0,1] or temp_set[n,1] < test[0,0] or temp_set[n,2] > test[0,3] or temp_set[n,3] < test[0,2]:
                        
                            # change test back into (x,y,w,h) as tuple called test1 to be put in good_candidates
                        test1=tuple([test.tolist()[0][0],test.tolist()[0][2],test.tolist()[0][1]-test.tolist()[0][0],test.tolist()[0][3]-test.tolist()[0][2]])
                            # if test1 is not yet in good_candidates, add test1. (to prevent duplication & error message)
                        if not ((test1 in good_candidates) or (test1 in used_set)):
                            good_candidates.add(test1)
                                # good_candidates_list is a list version of good_cnadidates, which has tuples; the list version is necessary to loop the process
                            good_candidates_list.append(list(test1))
                                # also add test1 into used_set
                            used_set.add(test1)
                        
                        # else, if test is inside temp_set row n data, calculate overlap percentage
                    else: 
                            # overlap percentage of regions defined as (A1 intersect with A2)/(A1 union A2) * 100%
                            # create 2x4 matrix called overlap set, which has test row 'u' and temp_set row 'n' data
                        overlap_set=[]
                        overlap_set.append(test.tolist()[0])
                        overlap_set.append(temp_set[n,:].tolist()[0])
                        overlap_set=np.asmatrix(overlap_set)
                        
                            # get overlap area
                        overlap_x1=max(overlap_set[:,0])
                        overlap_x2=min(overlap_set[:,1])
                        overlap_y1=max(overlap_set[:,2])
                        overlap_y2=min(overlap_set[:,3])
                        
                        overlap_w=overlap_x2-overlap_x1
                        overlap_h=overlap_y2-overlap_y1
                        
                        area_overlap=int(overlap_w*overlap_h)
                 
                            # get union area
                        outer_x1=min(overlap_set[:,0])
                        outer_x2=max(overlap_set[:,1])
                        outer_y1=min(overlap_set[:,2])
                        outer_y2=max(overlap_set[:,3])
                        
                        union_w=outer_x2-outer_x1
                        union_h=outer_y2-outer_y1
                        
                        gaps=0
                            # get max possible area - gaps, since there are at most 4 possible squares of gaps from the largest square area created by the two overlapping bounding boxes
                            
                            # if x1 smaller than x1', and y1 greater than y1' (top left corner)
                        if (overlap_set[1,0] < overlap_set[0,0] and overlap_set[1,2] > overlap_set[0,2]) or (overlap_set[1,0] > overlap_set[0,0] and overlap_set[1,2] < overlap_set[0,2]):
                            gaps += int(abs((overlap_set[0,0]-overlap_set[1,0])*(overlap_set[1,2]-overlap_set[0,2])))
                            # if x2 greater than x2' and y1 greater than y1' (top right corner)
                        if (overlap_set[1,1] > overlap_set[0,1] and overlap_set[1,2] > overlap_set[0,2]) or (overlap_set[1,1] < overlap_set[0,1] and overlap_set[1,2] < overlap_set[0,2]):
                            gaps += int(abs((overlap_set[1,1]-overlap_set[0,1])*(overlap_set[1,2]-overlap_set[0,2])))
                            # if x1 greater than x1' and y2 greater than y2' (bottom left corner)
                        if (overlap_set[1,0] > overlap_set[0,0] and overlap_set[1,3] > overlap_set[0,3]) or (overlap_set[1,0] < overlap_set[0,0] and overlap_set[1,3] < overlap_set[0,3]):
                            gaps += int(abs((overlap_set[1,0]-overlap_set[0,0])*(overlap_set[1,3]-overlap_set[0,3])))
                            # if x2 greater than x2' and y2 less than y2' (bottom right corner)
                        if (overlap_set[1,1] > overlap_set[0,1] and overlap_set[1,3] < overlap_set[0,3]) or (overlap_set[1,1] < overlap_set[0,1] and overlap_set[1,3] > overlap_set[0,3]):
                            gaps += int(abs((overlap_set[1,1]-overlap_set[0,1])*(overlap_set[1,3]-overlap_set[0,3])))

                        area_union = int(union_w*union_h) - gaps
                        percent_overlap = area_overlap/area_union

                            # Merge bounding boxes if overlap percent is large enough
                        if percent_overlap > overlap_threshold:
                            
                                # convert back to x,y,w,h and save as good_segment
                            good_segment = [int(outer_x1),int(outer_y1),int(outer_x2-outer_x1), int(outer_y2-outer_y1)]
                            good_segment = tuple(good_segment)
                            
                                # remove check temp_set row 'n' if inside good_candidates since it is now merged
                            if tuple(temp_set1[n,:].tolist()) in good_candidates:
                                good_candidates.remove(tuple(temp_set1[n,:].tolist()))
                                
                                # also remove from list if needed
                            if temp_set1[n,:].tolist() in good_candidates_list:
                                good_candidates_list.remove(temp_set1[n,:].tolist())
                                
                                # change test back into (x,y,w,h) as tuple called test1 to be put in good_candidates
                            test1=tuple([test.tolist()[0][0],test.tolist()[0][2],test.tolist()[0][1]-test.tolist()[0][0],test.tolist()[0][3]-test.tolist()[0][2]])
                            
                            used_set.add(test1)
                                
                                # remove test1 from good_candidates if needed since it is now merged
                            if tuple(test1) in good_candidates:
                                good_candidates.remove(test1)
                                
                                # also remove from list version
                            if list(test1) in good_candidates_list:
                                good_candidates_list.remove(list(test1))
                                
                                # if merged bounding box not yet in good_candidates, add it
                            if not(good_segment in good_candidates):
                                good_candidates.add(good_segment)
                                good_candidates_list.append(list(good_segment))
                                
                                # also record used data
                            used_set.add(good_segment)
                            used_set.add(tuple(temp_set1[n,:].tolist()))
                            used_set.add(test1)
                            skip=True
            # rest temp_set to be what is inside good_candidates_list for looping process
        temp_set=good_candidates_list
        
    # create fig1, ax1, create single subplot, then draw bounding boxes x, y, w, and save figure with name of model
def plot_bounding_boxes_with_names(image, candidates, name):
    # draw rectangles on the original image
    fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(25, 25))
    ax1.imshow(image)
    for x, y, w, h in ext_data: #or in candidates
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax1.add_patch(rect)
    plt.show()
    try:    
        fig1.savefig('/home/chloong/Desktop/Justin San Juan/Testing Folder/Output/'+str(name)+'.jpg')

    except:
        fig1.savefig('C:/Users/JustinSanJuan/Desktop/HKUST/UROP Deep Learning Image-based Structural Analysis/Training Data Set/Output/'+str(name)+'.jpg')

    
    # define padding function, to be used in preprocessing
def padwithzeros(vector, pad_width, iaxis, kwargs):
                # pad vector (image) in all directions with 0's for a length of pad_width
            vector[:pad_width[0]] = 0
            vector[-pad_width[1]:] = 0
            return vector
        
    # Edge-trim, remove unconnected edge pieces, resize, then predict classes
def preprocess_extractions(image,candidates, wanted_w, wanted_h, export_w, export_h):
    for x,y,w,h in candidates:
        # Given x,y,w,h, store each extraction and coordinates in lists    
        extraction = np.copy(image[y:y+h,x:x+w])
        
        labelled_array, max_label = measure.label(extraction, background=0, connectivity=1, return_num=True)
        
            # skip if array is empty
        if max_label == 0: 
            continue 

############## Adjust extraction window to remove excess empty area ##############
        black = np.array(np.where(extraction == 1))
        x1=min(black[1])
        x2=max(black[1])
        y1=min(black[0])
        y2=max(black[0])
        extraction = extraction[y1:y2,x1:x2]
            
        dist_x = x1
        dist_y = y1

########################## Remove unconnected edge pieces #######################
        #####Get array of different connected components, and record highest label number in max_label
        labelled_array, max_label = measure.label(extraction, background=0, connectivity=1, return_num=True)
            
        largest_array_index = 0
        largest_array_count = 0
        array_height = labelled_array.shape[0]
        array_width = labelled_array.shape[1]
        total_pixels = len(np.where(labelled_array!=0))
        ##### Parse through all labelled pieces to get largest piece
        for q in range(1,max_label+1):
            num_black = len(np.where(labelled_array == q)[0])
            if num_black > largest_array_count:
                largest_array_count = num_black
                largest_array_index = q
        ##### Parse through all labelled pieces except largest piece again and delete piece if on edge and not part of largest object
        for q in range(1,max_label+1):
            array_coords = np.where(labelled_array==q)
            piece_percent = len(array_coords)/total_pixels
            if q != largest_array_index and (min(array_coords[0]) == 0 or min(array_coords[1]) == 0 or max(array_coords[0])==array_height-1 or max(array_coords[1])==array_width-1) and piece_percent <= max_piece_percent:
                    # for all of the given coordinates of the labelled piece, change it to 0 (remove)
                extraction[array_coords[0],array_coords[1]]=0
############# Re-adjust extraction window to remove excess empty area ################
        black = np.array(np.where(extraction == 1))
        x1=min(black[1])
        x2=max(black[1])
        y1=min(black[0])
        y2=max(black[0])
        extraction = extraction[y1:y2,x1:x2]
        extraction = np.asmatrix(extraction)
################## redefine w and h #################
            #total distance added from x is x1+dist_x, similarly for y
        x = x + x1 + dist_x
        y = y + y1 + dist_y
        w = x2-x1
        h = y2-y1
################# RESIZE EXTRACTION ACCORDING TO WANTED_W and WANTED_H #############
            #
        subsample_length=max((w/wanted_w),(h/wanted_h))
        
        extraction1 = np.zeros((int(np.ceil(h/subsample_length)),int(np.ceil(w/subsample_length))))
        extraction1 = np.asmatrix(extraction1)
        for u in range(int(h)):
            for i in range(int(w)):
                
                if subsample_length>1:
                    w_left=int(i*subsample_length)
                    w_right=int(w_left+subsample_length+1)
                    h_up=int(u*subsample_length)
                    h_down=int((u+1)*subsample_length)
                    if np.sum(extraction[h_up:h_down,w_left:w_right]) > 0:
                        try:
                            extraction1[u,i]=1
                        except:
                            extraction[u,i-1]=1
                else:
                    duplication_length = 1/subsample_length #(wanted_w/w)
                    w_left=int(i*duplication_length)
                    w_right=int(w_left+duplication_length+1)
                    h_up=int(u*duplication_length)
                    h_down=int((u+1)*duplication_length)

                    extraction1[h_up:h_down,w_left:w_right]=extraction[u,i]
                            
        
        extraction2=extraction1[:,:]
        ########### NO NEED TO ADD PADDING USING 128 by 92 ###########
            # calculate necessary pad and cropping length to get (export_w,export_h) size matrix based on wanted_w and wanted
        pad_length=int(np.ceil(max((export_w-extraction2.shape[1])/2,(export_h-extraction2.shape[0])/2)))
        
        extraction2=np.pad(extraction2,pad_length,padwithzeros)
        extraction3=np.zeros((export_h,export_w))
        extraction3=np.asmatrix(extraction3)
        
        h_up=int((extraction2.shape[0]-export_h)/2)
        h_down=int((extraction2.shape[0]+export_h)/2)
        w_left=int((extraction2.shape[1]-export_w)/2)
        w_right=int((extraction2.shape[1]+export_w)/2)
        extraction3[:,:]=extraction2[h_up:h_down,w_left:w_right]
         
        ###############################################################
        
        ext_images.append(extraction3) #record the image array in list with first index = 0
        ext_data.append([x,y,w,h]) #record the (x,y,w,h) coordinates in list with first index = 0, this list data will be used for the reconstruction of the system (since coordinates are saved here)
    ext_class_index = np.zeros((len(ext_images))).tolist()
    ext_class_name = np.zeros((len(ext_images))).tolist()
    return ext_images, ext_data, ext_class_index, ext_class_name
def predict_classes(ext_images,group,ext_class_index,ext_class_name,ext_next_round,ext_next_round_index):
    if group =='numbers' or group =='all': 
        indices = range(len(ext_images))
    else: 
        indices = ext_next_round_index[:]
    if indices != []:
        for i in indices:
            image = ext_images[i]
    ########### ADD DIMENSIONS TO MATCH CLASSIFIER DIMENSIONS ################
            num_channel = 3 # since we need RGB
            
            if num_channel==1: # if classifier only needs 1 channel
                if K.image_dim_ordering()=='th': # modify data if using theano instead of tensorflow
                    image = np.expand_dims(image, axis=0)
                    image = np.expand_dims(image, axis=0)
                else:
                    image = np.expand_dims(image, axis=3) 
                    image = np.expand_dims(image, axis=0)
            		
            else:
                if K.image_dim_ordering()=='th': # modify data if using theano instead of tensorflow
                    image=np.rollaxis(image,2,0)
                    image = np.expand_dims(image, axis=0)
                else:
                    # expand dimensions as needed in classifier
                    image = np.expand_dims(image, axis=3)
                    image = np.expand_dims(image, axis=0)
    
    ########### PREDICT OBJECT CLASS W/ ENTROPY THEORY & RECORD DATA ############## 
    
                # get match percentages for each class from classifier
            prediction=model.predict(image)
            
            second_max=list(prediction[0])
            second_max.remove(max(second_max))
           
                # get first, second, and third maximum percentage matches, to be used for entropy calculations
            first_max=max(prediction[0])
            second_max=max(second_max)
            
                # attach percentages to lists (in range of 0 to 1.0, ex: 91% is recorded as 0.91)
            ext_match_percent.append(first_max)
            ext_match_percent2.append(second_max)
            
                # if prediction is not confident or if confidence, as calculated by the difference top two predictions is too hight, or if another third prediction is close to the second prediction
                # discard =raction as an 'unknown' class
            if first_max < min_percent_match or first_max-second_max < min_confidence:
                index=17 # index 17 is class 18, the unknown class
                
                # otherwise, if prediciton is confident, record the index and class name
            else:
                index=((prediction[0]).tolist().index(first_max))
            
                #save extractions
            ext_class_index[i] = index
            ext_class_name[i] = target_names_all[index]
                
    ######################## ADJUST PREDICTIONS ########################
                #adjust predictions to merge counter-clockwise moments
def adjust_predictions(ext_class_index,ext_class_name):
    for k in range(len(ext_class_index)):
        index= ext_class_index[k]
        
        if index >=14 and index <=17:
            index = 14
            
            #adjust predictions to merge clockwise moments
        elif index >=18 and index <=21:
            index = 15
            #adjust index of noise
        elif index == 22:
            index = 16
            
            #adjust index of random letters
        elif index ==23:
            index = 17
            
            #adjust predicitons to merge fixed supports
        elif index >= 24 and index <=31:
            index = 18
            
            #adjust predicitons to merge pinned supports
        elif index >=32 and index <=39:
            index = 19
            
            #adjust predictions to merge vertical roller supports
        elif (index >=40 and index <=41) or (index >=44 and index<=45):
            index = 20
            
            #adjust predicitons to merge horizontal roller supports
        elif (index >=42 and index <=43) or (index >=46 and index<=47):
            index = 21
            
            #adjust index of last 12 classes
        elif index >=48 and index <=63:
            index = index - 26
        
        ext_class_index[k] = index
        ext_class_name[k] = target_names[index]
                
###############################################################################
        
    # calculate accuracy, create a plot with the prediction, and percentage matches
def plot_extractions_with_names(ext_images,ext_data,ext_class_name,ext_class_index,name,**kwargs):
    
        #load kwarg ans, else leave ans variable as empty string
    ans = kwargs.get('ans','')

        #prepare figure size
    num_of_samples=len(ext_data)
    #include all input data in title
    subplot_num=int(np.ceil(np.sqrt(num_of_samples)))
    fig=plt.figure(figsize=(num_of_samples,num_of_samples))
    
    ##### Find Ground Truth Values if Available #####
    if ans != '':

        ans=np.array(ans)
        ext_class_index = np.array(ext_class_index)
        score_matrix = ans/ext_class_index
            # Replace all incorrect predictions with 0
        for i in range(0,len(ext_class_index)):
            if ans[i]==0 and ext_class_index[i]==0: #fix case of 0/0
                score_matrix[i]=1
            if score_matrix[i] != 1: #else change all non-1's (incorrect) to 0
                score_matrix[i] = 0
                
        li1= ['numbers','0','1','2','3','4','5','6','7','8','9']
        li2= ['forces','up','down','right','left']
        li3= ['moments','ctrcl_moments','cl_moments']
        li4= ['random','noise','alphab']
        li5= ['supports']
        li6= ['fixed_supports','fixed_right','fixed_left','fixed_down','fixed_up']
        li7= ['pinned_supports','pinned_down','pinned_up','pinned_left','pinned_right']
        li8= ['roller_supports','roller_down','roller_up','roller_left','roller_right']
        li9= ['distributed_loads', 'uniform_distributed','linear_distributed','quadratic_distributed','cubic_distributed']
        li10=['beams','horizontal','vertical','downward_diagonal','upward_diagonal']
        li11=['dimensions','length','height','ctrcl_angle','cl_angle']
            #create empty group lists 
        for j in (li1,li2,li3,li4,li5,li6,li7,li8,li9,li10,li11):
            for l in j:
                exec(str('acc_'+str(l))+'= []')

            #include scores into individual categories
        for i in range(1,len(ext_data)+1):
            if ans[i-1]>=0 and ans[i-1]<=9:
                exec('acc_'+'numbers'+'.append(score_matrix[i-1])')
                
                for j in range(1,len(li1)):
                    exec('if ans[i-1]=='+str(j-1)+': '+'acc_'+str(li1[j])+'.append(score_matrix[i-1])')
                        
            if ans[i-1]>=10 and ans[i-1]<=13:
                exec('acc_'+'forces'+'.append(score_matrix[i-1])')
                
                for j in range(1,len(li2)):
                    exec('if ans[i-1]=='+str(j+9)+': '+'acc_'+str(li2[j])+'.append(score_matrix[i-1])')
                    
            if ans[i-1]>=14 and ans[i-1]<=15:
                exec('acc_'+'moments'+'.append(score_matrix[i-1])')

                
                for j in range(1,len(li3)):
                    exec('if ans[i-1]=='+str(j+13)+': acc_'+str(li3[j])+'.append(score_matrix[i-1])')
            
            if ans[i-1]>=16 and ans[i-1]<=17:
                exec('acc_'+'random'+'.append(score_matrix[i-1])')
                
                for j in range(1,len(li4)):
                    exec('if ans[i-1]=='+str(j+15)+': acc_'+str(li4[j])+'.append(score_matrix[i-1])')
            
            if ans[i-1]>=18 and ans[i-1]<=21:
                exec('acc_'+'supports'+'.append(score_matrix[i-1])')
                
                if ans[i-1]==18:
                    exec('acc_'+'fixed_supports'+'.append(score_matrix[i-1])')
                if ans[i-1]==19:
                    exec('acc_'+'pinned_supports'+'.append(score_matrix[i-1])')
                if ans[i-1]==20 or ans[i-1]==21:
                    exec('acc_'+'roller_supports'+'.append(score_matrix[i-1])')
            if ans[i-1]>=22 and ans[i-1]<=25:
                exec('acc_'+'distributed_loads'+'.append(score_matrix[i-1])')
                
                for j in range(1,len(li9)):
                    exec('if ans[i-1]=='+str(j+21)+': acc_'+str(li9[j])+'.append(score_matrix[i-1])')
            
            if ans[i-1]>=26 and ans[i-1] <=29:
                exec('acc_'+'beams'+'.append(score_matrix[i-1])')
                
                for j in range(1,len(li10)):
                    exec('if ans[i-1]=='+str(j+25)+': acc_'+str(li10[j])+'.append(score_matrix[i-1])')
            
            if ans[i-1]>=30 and ans[i-1] <=33:
                exec('acc_'+'dimensions'+'.append(score_matrix[i-1])')
                
                for j in range(1,len(li11)):
                    exec('if ans[i-1]=='+str(j+29)+': acc_'+str(li11[j])+'.append(score_matrix[i-1])')
            
            #calculate accuracies by doing sum of 1's (correct answers) divided by number of entries in that category
        for j in (li1,li2,li3,li4,li5,li6,li7,li8,li9,li10,li11):
            for l in j:
                exec('try: acc1_'+str(l)+'= str(round((sum(eval(str(acc_'+str(l)+')))/len(eval(str(acc_'+str(l)+'))))*100,2))'+'\n'+'except:acc1_'+str(l)+"='N/A'")
            
            #prepare a list of "accuracy_x = some value" strings
        s=[]
        for j in (li1,li2,li3,li4,li5,li6,li7,li8):
            for l in j:
                s.append("acc_"+str(l)+" = "+str(eval("acc1_" +str(l)))+"% ")
            s.append("\n") #start new line after each category
            # join accuracy strings with spaces
        string=" ".join(s)

        ##### Calculate Overall Accuracy #####
        acc= sum(score_matrix)/len(score_matrix)
        print('Accuracy is '+str(acc*100)+" %")
        plt.title("Accuracy is "+str(acc*100)+" %" "\n" +string ,fontsize=20,color='blue')
    else:
        string='' #else, do anything useless
    
        #plot images with matching percentages, change to red if incorrect
    for i in range(1,len(ext_data)+1):
        color='black'
        if ans != '': #only do change to red if ans list is available
            if score_matrix[i-1] == 0:
                color='red'
            else:
                color='black'
        ax = fig.add_subplot(subplot_num, subplot_num, i)
        temp_image = ext_images[i-1]
            #plot extractions with matching percentages and corresponding color of text
        ax.set_title(str(i)+" prediction is : '"+ ext_class_name[i-1]+"' " "\n" + str(np.ceil(ext_match_percent[i - 1]*100*100)/100)+"% match" "\n" + str(((ext_match_percent[i-1]-ext_match_percent2[i-1])*100*100)/100) +"  1-2 difference %", color=color)
        ax.imshow(temp_image)
#        plt.tight_layout()
    
    plt.tight_layout()
    plt.show()
    
        #save the figure by name

    try:
        fig.savefig('/home/chloong/Desktop/Justin San Juan/Testing Folder/Output/ext'+str(name)+'.jpg')
        print('Saved extractions figure as /home/chloong/Desktop/Justin San Juan/Testing Folder/Output/ext'+str(name)+'.jpg')
    except:
        fig.savefig('C:/Users/JustinSanJuan/Desktop/HKUST/UROP Deep Learning Image-based Structural Analysis/Training Data Set/Output/ext'+str(name)+'.jpg')
        print('Saved extractions figure as C:/Users/JustinSanJuan/Desktop/HKUST/UROP Deep Learning Image-based Structural Analysis/Training Data Set/Output/ext'+str(name)+'.jpg')

        # draw rectangles on the original image
    fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(25, 25))
    
    
    together = []
    for i in range(len(ext_data)):
        x, y, w, h = ext_data[i]
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax1.add_patch(rect)
        color='b' #not compared to any answer
        try:
            if ext_class_index[i] != ans[i]:
                color='r' # incorrect
            else:
                color = 'g' # correct
                
        except:
            color = 'b' #not compared to any answer
#        ax1.annotate(str(i) + ' : ' + str(ext_class_name[i]),xy=(x, y-2),fontsize=12,color=color)
        string = str(i+1) + ' : ' + str(ext_class_name[i])
        
        together.append((string,x,y,color))
    
    together.sort()

    text = [a for (a,b,c,d) in together]
    eucs = [b for (a,b,c,d) in together]
    covers = [c for (a,b,c,d) in together]
    cols = [d for (a,b,c,d) in together]

    plt.plot(eucs,covers,color="black", alpha=0.0)
    texts = []
    for xt, yt, s, color in zip(eucs, covers, text,cols):
        txt = plt.text(xt, yt, s,size = 18,color=color)
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
        texts.append(txt)

    f = interpolate.interp1d(eucs, covers)
    x = np.arange(min(eucs), max(eucs), 20)
    y = f(x)
    #y = f(x)
    #print(y)
    adjust_text(texts,x=x,y=y, arrowprops=dict(arrowstyle="->", color='r', lw=1),
                expand_points=(1.2,1.2),
                force_text=0.9,
                force_points = 0.15)
############ convert to graymap
    image1=np.multiply(np.subtract(image,1),-1)
    cmap = plt.cm.gray
    norm = plt.Normalize(image1.min(),image1.max())
    rgba = cmap(norm(image1))
    ax1.imshow(rgba,interpolation='nearest')
##################
#    ax1.imshow(image)
#    plt.tight_layout()
    plt.show()
    try:    
        fig1.savefig('C:/Users/JustinSanJuan/Desktop/HKUST/UROP Deep Learning Image-based Structural Analysis/Training Data Set/Output/print_'+str(name)+'.jpg')
    except:
        fig1.savefig('/home/chloong/Desktop/Justin San Juan/Testing Folder/Output/print_'+str(name)+'.jpg')

    #Stack new images and answers then add to training samples
def update_answers(ext_images,ans):
    print('Updating answers...')
        #Load answers as a vertical column array
    y_ans = np.transpose(np.asarray(ans).reshape(1,len(ans)))
    print(y_ans.shape)
        #Load image data as x array
    x_ans = np.asarray(ext_images).reshape(len(ext_images),img_rows*img_cols)
    print(x_ans.shape)
        #Put together x and y as single array
    data_ans = np.hstack((x_ans,y_ans))
    print('Adding ' + data_ans.shape[0] +' training samples...')
    
        #Load current answers
    data_all=np.load('Training_Samples_'+str(num_classes)+'_classes_'+str(img_rows)+'x'+str(img_cols)+'_all.npy')
    print('Inital shape = '+ eval('data_all').shape)
        #Add new answers to old answers
    data_all=np.vstack((data_all,data_ans))
    print('Final shape = '+ eval('data_all').shape)
        #Save data
    np.save('Training_Samples_'+str(num_classes)+'_classes_'+str(img_rows)+'x'+str(img_cols)+'_all',data_all)
def binarize_image(image,a):
    image = filters.gaussian(image,a)
    
    # Locally adaptive threshold
    adaptive_threshold = threshold_local(image, block_size=21, offset=0.02)
    
    # Return a binary array
    # 0 (WHITE): image >= adaptive_threshold
    # 1 (BLACK): image < adaptive_threshold
    image = np.array(image < adaptive_threshold) * 1
    return image

def select_good_bounding_boxes(image,imagename,ext_images,ext_data,ext_class_index,ext_class_name,target_names):
    
    lines=[]
    address = "/home/chloong/Desktop/ISA_FYP/GT/"
    filename = address+'GT_'+str(imagename)
    mode = 'w+'#input('mode of file (w:write,a:append, include "+" to create if not there'+'\n'))
    f = open(str(filename)+'.txt',str(mode))
    lines.append(str(imagename)+'\n')
    ext_class_index_temp=[]
    ext_class_name_temp=[]
    ext_images_temp=[]
    ext_data_temp=[]
    for k in range(len(ext_data)):
        fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
        ax1.imshow(image)
        x, y, w, h = ext_data[k]
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax1.add_patch(rect)
        plt.show()
        
        fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(1.5, 1.5))
        x1=ext_data[k][0]
        y1=ext_data[k][1]
        x2=x1+ext_data[k][2]
        y2=y1+ext_data[k][3]
        ext_image_from_image = image[y1:y2,x1:x2]
        ax1.imshow(ext_image_from_image)
        plt.show()
        
        answer = input(str(k+1)+'/'+str(len(ext_data))+' '
                       'Prediction is '+ str(ext_class_name[k])+'\n'+
                       'If it is correct, press enter, otherwise enter correct class index' + '\n'+
                       'Else if it is a bad segmentation, enter "b"' +'\n'+
                       'If you want to go back to the previous segmentation enter "r"'+'\n')
        
        if answer =='' or answer ==' ':
            print('Data was correct')
            ext_class_index_temp.append(ext_class_index[k])
            ext_class_name_temp.append(ext_class_name[k])
            ext_images_temp.append(ext_images[k])
            ext_data_temp.append(ext_data[k])
            
            string = str(ext_data[k][0])+' '+ str(ext_data[k][1])+' '+ str(ext_data[k][2])+' '+ str(ext_data[k][3])+' '+str(ext_class_index[k])
        elif answer =='b' or answer =='"b"' or answer == "'b'":
            ext_class_index_temp.append(int(23))
            ext_class_name_temp.append(target_names_all[23])
            ext_images_temp.append(ext_images[k])
            ext_data_temp.append(ext_data[k])
            string = str(ext_data[k][0])+' '+ str(ext_data[k][1])+' '+ str(ext_data[k][2])+' '+ str(ext_data[k][3])+' '+str(23)

            
        elif answer =='r': #revert
            #remove previous answer from list
            ext_class_index_temp= ext_class_index_temp[0:-1]
            ext_class_name_temp= ext_class_name_temp[0:-1]
            ext_images_temp=ext_images_temp[0:-1]
            ext_data_temp=ext_data_temp[0:-1]
            lines = lines[:-1]
            
            
            #print bounding box on problem image
            fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
            ax1.imshow(image)
            x, y, w, h = ext_data[k-1]
            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='red', linewidth=2)
            ax1.add_patch(rect)
            plt.show()
            
            #print 
            fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(1.5, 1.5))
            x1=ext_data[k-1][0]
            y1=ext_data[k-1][1]
            x2=x1+ext_data[k-1][2]
            y2=y1+ext_data[k-1][3]
            ext_image_from_image = image[y1:y2,x1:x2]
            ax1.imshow(ext_image_from_image)
            plt.show()
            
            answer = input(str(k)+'/'+str(len(ext_data))+' '
                           'Prediction is '+ str(ext_class_name[k-1])+'\n'+
                           'If it is correct, press enter, otherwise enter correct class index' + '\n'+
                           'Else if it is a bad segmentation, enter "b"' +'\n'+
                           'If you want to go back to the previous segmentation enter "r"'+'\n')
            
            if answer =='' or answer ==' ':
                print('Data was correct')
                ext_class_index_temp.append(ext_class_index[k-1])
                ext_class_name_temp.append(ext_class_name[k-1])
                ext_images_temp.append(ext_images[k-1])
                ext_data_temp.append(ext_data[k-1])
                string = str(ext_data[k-1][0])+' '+ str(ext_data[k-1][1])+' '+ str(ext_data[k-1][2])+' '+ str(ext_data[k-1][3])+' '+str(ext_class_index[k-1])
            
            elif answer =='b' or answer =='"b"' or answer == "'b'":
                ext_class_index_temp.append(int(23))
                ext_class_name_temp.append(target_names_all[23])
                ext_images_temp.append(ext_images[k])
                ext_data_temp.append(ext_data[k])
                string = str(ext_data[k-1][0])+' '+ str(ext_data[k-1][1])+' '+ str(ext_data[k-1][2])+' '+ str(ext_data[k-1][3])+' '+str(23)
            
            elif answer =='c' or answer == '"c"' or answer =='"c"':
                f.writelines([item for item in lines])
                f.close()
                break
            else:
                
                ext_class_index_temp.append(int(answer))
                ext_class_name_temp.append(target_names_all[int(answer)])
                ext_images_temp.append(ext_images[k-1])
                ext_data_temp.append(ext_data[k-1])
                string = str(ext_data[k-1][0])+' '+ str(ext_data[k-1][1])+' '+ str(ext_data[k-1][2])+' '+ str(ext_data[k-1][3])+' '+str(answer)
            
            if string != '':
                lines.append(string+'\n')
            
            fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
            ax1.imshow(image)
            x, y, w, h = ext_data[k]
            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
            ax1.add_patch(rect)
            plt.show()
            
            fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(1.5, 1.5))
            x1=ext_data[k][0]
            y1=ext_data[k][1]
            x2=x1+ext_data[k][2]
            y2=y1+ext_data[k][3]
            ext_image_from_image = image[y1:y2,x1:x2]
            ax1.imshow(ext_image_from_image)
            plt.show()
            
            answer = input(str(k+1)+'/'+str(len(ext_data))+' '
                           'Prediction is '+ str(ext_class_name[k])+'\n'+
                           'If it is correct, press enter, otherwise enter correct class index' + '\n'+
                           'Else if it is a bad segmentation, enter "b"' +'\n'+
                           'If you want to go back to the previous segmentation enter "r"'+'\n')
            #check current k again
            if answer =='' or answer ==' ':
                print('Data was correct')
                ext_class_index_temp.append(ext_class_index[k])
                ext_class_name_temp.append(ext_class_name[k])
                ext_images_temp.append(ext_images[k])
                ext_data_temp.append(ext_data[k])
                string = str(ext_data[k][0])+' '+ str(ext_data[k][1])+' '+ str(ext_data[k][2])+' '+ str(ext_data[k][3])+' '+str(ext_class_index[k])
            
            elif answer =='b' or answer =='"b"' or answer == "'b'":
                ext_class_index_temp.append(int(23))
                ext_class_name_temp.append(target_names_all[23])
                ext_images_temp.append(ext_images[k])
                ext_data_temp.append(ext_data[k])
                string = str(ext_data[k][0])+' '+ str(ext_data[k][1])+' '+ str(ext_data[k][2])+' '+ str(ext_data[k][3])+' '+str(23)
            
            elif answer =='c':
                f.writelines([item for item in lines])
                f.close()
                break
            else:
                
                ext_class_index_temp.append(int(answer))
                ext_class_name_temp.append(target_names_all[int(answer)])
                ext_images_temp.append(ext_images[k])
                ext_data_temp.append(ext_data[k])
                string = str(ext_data[k][0])+' '+ str(ext_data[k][1])+' '+ str(ext_data[k][2])+' '+ str(ext_data[k][3])+' '+str(answer)
            
        elif answer =='c':
            f.writelines([item for item in lines])
            f.close()
            break
        else:
            
            ext_class_index_temp.append(int(answer))
            ext_class_name_temp.append(target_names_all[int(answer)])
            ext_images_temp.append(ext_images[k])
            ext_data_temp.append(ext_data[k])
            
            string = str(ext_data[k][0])+' '+ str(ext_data[k][1])+' '+ str(ext_data[k][2])+' '+ str(ext_data[k][3])+' '+str(answer)

    
    
        if string != '':
            lines.append(string+'\n')
        print(ext_data_temp)
    
    
    while True:
            #show final set of bounding boxes and classes
        fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(25, 25))
        ax1.imshow(image)
        print(ext_data_temp)
        for k in range(len(ext_data_temp)):
            if ext_class_index_temp[k]!=23:
                x, y, w, h = ext_data_temp[k]
                rect = mpatches.Rectangle(
                    (x, y), w, h, fill=False, edgecolor='red', linewidth=2)
                ax1.add_patch(rect)
            #need to add classes
        tick_spacing = 20
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.xticks(rotation=70)
        plt.grid()
        plt.show()
        
        value = input('If above figure captures all objects, press enter'+'\n'+'Otherwise, enter "a"'+'\n')
        if value == '' or value == ' ':
            f.writelines([item for item in lines])
            f.close()
            break
        elif value =='c':
            f.writelines([item for item in lines])
            f.close()
            break
        else:
            x1=int(input('Enter the top left corner x coordinate'+'\n'))
            y1=int(input('Enter the top left corner y coordinate'+'\n'))
            x2=int(input('Enter the bottom right corner x coordinate'+'\n'))
            y2=int(input('Enter the bottom right corner y coordinate'+'\n'))
            
            x=x1
            y=y1
            w = x2-x1
            h = y2-y1
            #do prediction
            extraction=image[y1:y2,x1:x2]

            wanted_w = 100
            wanted_h = 100
            
            subsample_length=max((w/wanted_w),(h/wanted_h))
        
            extraction1 = np.zeros((int(np.ceil(h/subsample_length)),int(np.ceil(w/subsample_length))))
            extraction1 = np.asmatrix(extraction1)
            for u in range(int(h)):
                for i in range(int(w)):
                    
                    if subsample_length>1:
                        w_left=int(i*subsample_length)
                        w_right=int(w_left+subsample_length+1)
                        h_up=int(u*subsample_length)
                        h_down=int((u+1)*subsample_length)
                        if np.sum(extraction[h_up:h_down,w_left:w_right]) > 0:
                            try:
                                extraction1[u,i]=1
                            except:
                                extraction[u,i-1]=1
                    else:
                        duplication_length = 1/subsample_length #(wanted_w/w)
                        w_left=int(i*duplication_length)
                        w_right=int(w_left+duplication_length+1)
                        h_up=int(u*duplication_length)
                        h_down=int((u+1)*duplication_length)
    
                        extraction1[h_up:h_down,w_left:w_right]=extraction[u,i]
                                
            
            extraction2=extraction1[:,:]
            ########### NO NEED TO ADD PADDING USING 128 by 92 ###########
                # calculate necessary pad and cropping length to get (export_w,export_h) size matrix based on wanted_w and wanted
            pad_length=int(np.ceil(max((export_w-extraction2.shape[1])/2,(export_h-extraction2.shape[0])/2)))
            
            extraction2=np.pad(extraction2,pad_length,padwithzeros)
            extraction3=np.zeros((export_h,export_w))
            extraction3=np.asmatrix(extraction3)
            
            h_up=int((extraction2.shape[0]-export_h)/2)
            h_down=int((extraction2.shape[0]+export_h)/2)
            w_left=int((extraction2.shape[1]-export_w)/2)
            w_right=int((extraction2.shape[1]+export_w)/2)
            extraction3[:,:]=extraction2[h_up:h_down,w_left:w_right]
         
            ext = extraction3
            ext_im=ext[:,:]
            num_channel = 3 # since we need RGB
            
            if num_channel==1: # if classifier only needs 1 channel
                if K.image_dim_ordering()=='th': # modify data if using theano instead of tensorflow
                    ext_im = np.expand_dims(ext_im, axis=0)
                    ext_im = np.expand_dims(ext_im, axis=0)
                else:
                    ext_im = np.expand_dims(ext_im, axis=3) 
                    ext_im = np.expand_dims(ext_im, axis=0)
            		
            else:
                if K.image_dim_ordering()=='th': # modify data if using theano instead of tensorflow
                    ext_im = np.rollaxis(ext_im,2,0)
                    ext_im = np.expand_dims(ext_im, axis=0)
                else:
                    # expand dimensions as needed in classifier
                    ext_im = np.expand_dims(ext_im, axis=3)
                    ext_im = np.expand_dims(ext_im, axis=0)
            prediction=model.predict(ext_im)[0]
            c = int(prediction.tolist().index(max(prediction)))
         
            fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
            ax1.imshow(image)
            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='red', linewidth=2)
            ax1.add_patch(rect)
            plt.show()
            
            fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(1.5, 1.5))
            ax1.imshow(extraction)
            plt.show()
            print('Prediction is ' +str(target_names_all[c]))
            check= input('Please enter the class index if prediciton is incorrect'+'\n'
                         + 'Otherwise, press enter to confirm prediction'+'\n')
            if check == '' or check == ' ':
                string = str(x1)+' '+str(y1)+' '+str(w)+' '+str(h)+' '+str(c)
                lines.append(string+'\n')
                ext_images_temp.append(ext)
                data=(x1,y1,w,h)
                ext_data_temp.append(data)
                ext_class_index_temp.append(c)
                ext_class_name.append(target_names_all[c])
            else:
                c = int(check)
                string = str(x1)+' '+str(y1)+' '+str(w)+' '+str(h)+' '+str(c)
                lines.append(string+'\n')
                ext_images_temp.append(ext)
                data=(x1,y1,w,h)
                ext_data_temp.append(data)
                ext_class_index_temp.append(c)
                ext_class_name.append(target_names_all[c])
        
            
        fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(25, 25))
        ax1.imshow(image)
        print(ext_data_temp)
        for k in range(len(ext_data_temp)):
            if ext_class_index_temp[k]!=23:
                x, y, w, h = ext_data_temp[k]
                rect = mpatches.Rectangle(
                    (x, y), w, h, fill=False, edgecolor='red', linewidth=2)
                ax1.add_patch(rect)
            #need to add classes
        tick_spacing = 20
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.xticks(rotation=70)
        plt.grid()
        plt.show()
        delete = input('If you want to delete the previous input, enter "y"'+'\n'+'To close: enter "c"'+'\n'+'To continue: press enter'+'\n')
        if delete =='y' or delete=='"y"' or delete=="'y'":
            lines = lines[:-1]
        elif delete=='c'or delete=='"c"' or delete=="'c'":
            f.writelines([item for item in lines])
            f.close()
            break
            
        
 
print('Done defining functions...')

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
target_names_all = ['0','1','2','3','4','5','6','7','8','9',
            #10
            'upwards force','downwards force','rightwards force','leftwards force',
            #14
            'counter-clockwise moment right', 'counter-clockwise moment up', 'counter-clockwise moment left', 'counter-clockwise moment down', 
            #18
            'clockwise moment right','clockwise moment up','clockwise moment left','clockwise moment down',
            #22
            'unknown','random alphabet',
            #24
            'fixed support right','fixed support left','fixed support down', 'fixed support up',
            #28
            'fixed support right w/ beam','fixed support left w/ beam','fixed support down w/ beam', 'fixed support up w/ beam',
            #32
            'pinned support down', 'pinned support up','pinned support left', 'pinned support right',
            #36
            'pinned support down w/ beam', 'pinned support up w/ beam','pinned support left w/ beam', 'pinned support right w/ beam',
            #40
            'roller support down', 'roller support up','roller support left','roller support right',
            #44
            'roller support down w/ beam', 'roller support up w/ beam','roller support left w/ beam','roller support right w/ beam',
            #48
            'uniformly distributed load', 'linearly distributed load','quadratically distributed load', 'cubically distributed load',
        	   #52
        	   'horizontal beam','vertical beam','downward diagonal beam', 'upward diagonal beam',
            #56
            'length','height','counter-clockwise angle','clockwise angle',
            #60
            'measure left','measure right','measure up','measure down'
            ]
target_names=['0','1','2','3','4','5','6','7','8','9',
                #10
                'upwards force','downwards force','rightwards force','leftwards force',
                #14
                'counter-clockwise moment', 'clockwise moment','unknown','random',
                #18
                'fixed support','pinned support','roller support vertical', 'roller support horizontal',
                #22
                'uniformly distributed load', 'linearly distributed load','quadratically distributed load', 'cubically distributed load',
                #26
                'horizontal beam','vertical beam','downward diagonal beam', 'upward diagonal beam',
                #30
                'length','height','counter-clockwise angle','clockwise angle',
                #34
                'measure left','measure right', 'measure up','measure down',
                ]

    
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
    
    ##### all lists of of samples answers available at: https://docs.google.com/document/d/1-baP1bGHS5Eyu9cKchAxfyYbnh9kxVCdC72uX6lOuj0/edit?usp=sharing
        #answers for samples1.jpg
#    ans=[28,52,3,0,33,23,54,53,23,45,55,52,0,52,13,23,
#    23,53,23,10,1,18,10,41,2,23,10,1,53,24,18,1,
#    23,23,33,55,9,26,5,7,18,27,55,23,53,23,23,41,
#    5,2,37,23,23,13,9,11,52,23,23,52,44,23,23,23,
#    27,52,53,1,23,12,23,12,23,7,23,23,23,52,55,54,
#    23,23,24,53,0,1,45,52,41,23,23,23,24,40,52,6,
#    7,7,23,52,45,4,53,23,2,23,53,1,55,23,0,23,
#    23,52,6,27,23,53,23,23,23,23,8,23,16,11,11,23,
#    8,23,9,23,8,11,7,29,23,36,32,28,52,45,1,23,
#    52,23,33,10,1,5,36,23,13,28,52,11,41,27,52,7,
#    11,23,55,55,11,52,52,23,52,44,4,1,7,23,27,41,
#    27,53,23,52,52,45,55,23,52,16,23,53,55,28,23,23,
#    41,52,23,11,55,52,25,23,11,4,23,23,23,23,52,23,
#    36,52,44,37,52,41,3,18,23,52,3,23,45,54,37,12,
#    28,23,52,23,16,23,23,53,0,23,6
#    ]
#        # answers for samples.jpg
#    ans = [13,8,23,16,14,4,5,12,28,53,52,
#    0,11,23,2,6,8,23,23,52,23,13,
#    18,23,1,2,53,7,11,53,11,33,41,
#    58,55,27,1,23,23,23,22,33,23,41,
#    9,24,53,12,18,7,52,6,53,23,53,
#    23,10,27,23,5,16,23,33,23,23,1,
#    12,4,52,13,9,23,10,41,17,41,27,
#    53,23,10,6,10,23,3,12,13,56,23,
#    52,52,10,5,9,8,23,1,0,27,0,
#    23,9,52,58,12,13,12,53,19,1,2,
#    23,52,23,53,3,3,1,7]
#    
    try:
        adjusted_ans = ans[:]
        for r in range(0,int(len(ans))):
            if ans[r] >= 14 and ans[r] <= 17:
                adjusted_ans[r] = 14
            elif ans[r] >=18 and ans[r] <=21:
                adjusted_ans[r] = 15
            elif ans[r] ==22:
                adjusted_ans[r] = 16
            elif ans[r] ==23:
                adjusted_ans[r] = 17
            
            elif ans[r] >= 24 and ans[r] <=31:
                adjusted_ans[r] = 18
            elif ans[r] >=32 and ans[r] <=39:
                adjusted_ans[r] = 19
            elif (ans[r] >=40 and ans[r] <=41) or (ans[r] >=44 and ans[r]<=45):
                adjusted_ans[r] = 20
            elif (ans[r] >=42 and ans[r] <=43) or (ans[r] >=46 and ans[r]<=47):
                adjusted_ans[r] = 21
            elif ans[r] >=48:
                adjusted_ans[r] = adjusted_ans[r] - 26
    except:
        adjusted_ans=''
        #create figure with all extractions and percent matches if no answers
    
    select_good_bounding_boxes(image,imagename,ext_images,ext_data,ext_class_index,ext_class_name,target_names)
    
    adjust_predictions(ext_class_index,ext_class_name)
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
#%%

    #Print test loss & accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1]) 

#%%
########## Plot confusion Matrix ################

from sklearn.metrics import confusion_matrix
import itertools

Y_pred = model.predict(x_test)
y_pred = np.argmax(Y_pred, axis=1)

 # set classes
target_names = ['0','1','2','3','4','5','6','7','8','9','upwards force','downwards force','rightwards force','leftwards force',
                'counter-clockwise moment', 'clockwise moment','unknown','random alphabet',
                'fixed support right','fixed support left','fixed support down', 'fixed support up', 
                'fixed support right w/ beam','fixed support left w/ beam','fixed support down w/ beam', 'fixed support up w/ beam',
                'pinned support down', 'pinned support up','pinned support left', 'pinned support right',
                'pinned support down w/ beam', 'pinned support up w/ beam','pinned support left w/ beam', 'pinned support right w/ beam',
                'roller support down', 'roller support up','roller support left','roller support right',
                'roller support down w/ beam', 'roller support up w/ beam','roller support left w/ beam','roller support right w/ beam'
                ]
					
# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), y_pred))

np.set_printoptions(precision=2)

plt.figure(figsize=(30,30))

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')

plt.show()
        #*******************saving does not work properly
try:
    plt.savefig('/home/chloong/Desktop/Justin San Juan/Testing Folder/Output/cnf_matrix_'+name+'_'+str(epochs)+'epochs.jpg')
except:
    plt.savefig('C:/Users/JustinSanJuan/Desktop/HKUST/UROP Deep Learning Image-based Structural Analysis/Code/Python/Testing Folder/cnf_matrix_'+name+'_'+str(epochs)+'epochs.jpg')

#%%
#################################################
########### PLOT GROUND TRUTH SAMPLES ###########
#################################################

target_names = ['0','1','2','3','4','5','6','7','8','9','upwards force','downwards force','rightwards force','leftwards force',
                'counter-clockwise moment', 'clockwise moment','unknown','random alphabet',
                'fixed support right','fixed support left','fixed support down', 'fixed support up', 
                'fixed support right w/ beam','fixed support left w/ beam','fixed support down w/ beam', 'fixed support up w/ beam',
                'pinned support down', 'pinned support up','pinned support left', 'pinned support right',
                'pinned support down w/ beam', 'pinned support up w/ beam','pinned support left w/ beam', 'pinned support right w/ beam',
                'roller support down', 'roller support up','roller support left','roller support right',
                'roller support down w/ beam', 'roller support up w/ beam','roller support left w/ beam','roller support right w/ beam'
                ]

    # Load set of correct answers
    # answers are in form answers[image index][coordinates = 0, classes = 1][bounding box index: 0,1,2,3,...]

#################################Load image with answers ################################################

try:
    image = np.load('C:/Users/JustinSanJuan/Desktop/HKUST/UROP Deep Learning Image-based Structural Analysis/Training Data Set/Input (numpy file)/easy_training_images.npy')
except:
    image = np.load('/home/chloong/Desktop/Justin San Juan/Testing Folder/easy_training_images.npy')

First_Entry=True
first_img=0
last_img=1
coords=[]

    # load answers file
f = open('GroundTruth.txt')

for i in range(first_img,last_img):
        #set temp_image as image number i
    temp_image = image[:,:,i]

    for l in f:
        
        if l.startswith("img_"):
            if First_Entry==True:
                First_Entry=False
                    #clean all data lists
                answers = []
                img_index=l.split("img_",1)[1]
                img_answers = {}
                coords = []
                classes = []
            else:
                    # create img_answers by attaching coordinates and classes
                img_answers = []
                img_answers.append(coords)
                img_answers.append(classes)
                
                    # save img_answers into answers
                answers.append(img_answers)
                
                    # reset other data lists
                img_index=l.split("img_",1)[1]
                img_answers = []
                coords = []
                classes = []
        elif len(l.split()) == 5:
            y1, y2, x1, x2, c = l.split()
            coords.append(tuple((int(x1),int(y1),int(x2)-int(x1),int(y2)-int(y1))))
            classes.append(int(c))
            
            #answers are stored as answers[image][coords or classes][answer_k]
            
            #plot image with bounding boxes and class names with tick spacing of 20
    fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(30, 30))
    ax1.imshow(image[:,:,i])
    
    tick_spacing = 20
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.grid()
    for k in range(len(answers[i][0])):
        x, y, w, h = answers[i][0][k]
        c = answers[i][1][k]
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax1.add_patch(rect)
        ax1.annotate(target_names[c],xy=(x, y-10),fontsize=20,color='w')
    plt.show()

    try:
        fig1.savefig('/home/chloong/Desktop/Justin San Juan/Testing Folder/easy'+str(i)+'ans.jpg')
    except:
        try:
            fig1.savefig('C:/Users/JustinSanJuan/Desktop/HKUST/UROP Deep Learning Image-based Structural Analysis/Code/Python/Testing Folder 1.0/easy'+str(i)+'ans.jpg')
        except:
            print('put new file destination here')
            
            
        # Prepare clean data
    ext_images=[]
    ext_data=[]
    ext_class_index=[]
    ext_class_name=[]
    ext_match_percent=[]
    ext_match_percent2=[]
    ext_match_percent3=[]
    
        # define wanted_w, and wanted_h, which is the are where the extraction is limited to
        # define export_w and export_h as required by the classifier
    wanted_w=img_cols
    wanted_h=img_rows    
    export_w=img_cols
    export_h=img_rows

        # prepare extractions to be sent to classifier
    ext_images, ext_data, ext_class_index, ext_class_name = preprocess_extractions(image,answers[i][0], wanted_w, wanted_h, export_w, export_h) 
    
    name = 'extractions_image_'+str(i)
    plot_extractions_with_names(ext_images, ext_data, ext_class_name, ext_class_index, name, ans = answers[i][1]) # create figure with all extractions and percentage matches
#%%
##########################################################################
########### TEST SEGMENTATION AND CLASSIFIER ON SAMPLE PROBLEM ###########
##########################################################################

try:
    image = np.load('C:/Users/JustinSanJuan/Desktop/HKUST/UROP Deep Learning Image-based Structural Analysis/Training Data Set/Input (numpy file)/easy_training_images.npy')
except:
    image = np.load('/home/chloong/Desktop/Justin San Juan/Testing Folder/easy_training_images.npy')
image=image[:,:,6]

    # reset sets and lists
good_candidates = set()
good_candidates_list=[]
bad_candidates = set()

    # get good_candidates set from search function, and also put it into 'candidates'
search(image, good_candidates, bad_candidates, scale_input, sigma_input, min_size_input)
candidates=good_candidates # select candidates to be classified, can be only good, only bad, or both sets of candidates
name = 'Test_easy'
plot_bounding_boxes_with_names(image,candidates, name) # plot bounding boxes on original image

    # define class names (use full class list)
target_names = ['0','1','2','3','4','5','6','7','8','9',
                #10
                'upwards force','downwards force','rightwards force','leftwards force',
                #14
                'counter-clockwise moment right', 'counter-clockwise moment up', 'counter-clockwise moment left', 'counter-clockwise moment down', 
                #18
                'clockwise moment right','clockwise moment up','clockwise moment left','clockwise moment down',
                #22
                'unknown','random alphabet',
                #24
                'fixed support right','fixed support left','fixed support down', 'fixed support up',
                #28
                'fixed support right w/ beam','fixed support left w/ beam','fixed support down w/ beam', 'fixed support up w/ beam',
                #32
                'pinned support down', 'pinned support up','pinned support left', 'pinned support right',
                #36
                'pinned support down w/ beam', 'pinned support up w/ beam','pinned support left w/ beam', 'pinned support right w/ beam',
                #40
                'roller support down', 'roller support up','roller support left','roller support right',
                #44
                'roller support down w/ beam', 'roller support up w/ beam','roller support left w/ beam','roller support right w/ beam'
                #48
                'uniformly distributed load', 'linearly distributed load','quadratically distributed load', 'cubically distributed load',
            	   #52
            	   'horizontal beam','vertical beam','downward diagonal beam', 'upward diagonal beam',
                #56
                'length','height','counter-clockwise angle','clockwise angle'
                ]


    # Create/reset list of images, coordinates (x,y,w,h) data, class indices, class names, and top three percentage matches
ext_images=[]
ext_data=[]
ext_class_index=[]
ext_class_name=[]
ext_match_percent=[]
ext_match_percent2=[]

    # define wanted_w, and wanted_h, which is the are where the extraction is limited to
    # define export_w and export_h as required by the classifier
wanted_w=img_cols
wanted_h=img_rows
export_w=img_cols
export_h=img_rows
    # prepare extractions to be sent to classifier
ext_images, ext_data, ext_class_index, ext_class_name = preprocess_extractions(image,candidates, wanted_w, wanted_h, export_w, export_h)

plot_extractions_with_names(ext_images, ext_data, ext_class_name, ext_class_index, name) # create figure with all extractions and percentage matches

#%%
############### PRINT IMAGES PER CLASS ################
preview_npy=np.load('Training_Samples_60_classes_100x100_all.npy')[:,::-1].reshape(data_all.shape[0],img_rows,img_cols)
for i in range(0,int((preview_npy.shape[0]+1)/625)):
    print(i)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
    ax.imshow(preview_npy[i*625,:,:])
    plt.show()
#%%
############# PLOT TRAINING IMAGES WITH GRID ################
try:
    image = np.load('C:/Users/JustinSanJuan/Desktop/HKUST/UROP Deep Learning Image-based Structural Analysis/Training Data Set/Input (numpy file)/easy_training_images.npy')
except:
    image = np.load('/home/chloong/Desktop/Justin San Juan/Testing Folder/easy_training_images.npy')

first_img=0
last_img=image.shape[2]
for i in range(first_img,last_img):
    print(str(i))
    fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(30, 30))
    ax1.imshow(image[:,:,i])
    tick_spacing = 20
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.grid()
    plt.show()
    
    fig1.savefig('C:/Users/JustinSanJuan/Desktop/HKUST/UROP Deep Learning Image-based Structural Analysis/Code/Python/Testing Folder 1.0/easy'+str(i)+'.jpg')
    
#%%
fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
ax1.imshow(image)
plt.show()