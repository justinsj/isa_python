from component_classifier_predict import ComponentClassifierPredict
from component_classifier_training import ComponentClassifierTraining
from extraction_labelling import ExtractionLabelling
from extraction_preprocessing import ExtractionPreprocessing
from helper_functions import calculate_accuracy
import numpy as np
import matplotlib.pyplot as plt
from random import sample
import gc
import random
import os.path
import time

class TestingClass(object):

    def __init__(self, PATH, wanted_w, wanted_h, export_w, export_h, max_piece_percent):
        self.PATH = PATH
        self.wanted_w = wanted_w
        self.wanted_h = wanted_h
        self.export_w = export_w
        self.export_h = export_h
        self.max_piece_percent = max_piece_percent
        gc.enable()
    
    def pick_random_training_samples(self, training_dataset_filename, n, seed):
        '''
        Inputs: training_data, n
        Outputs: randomly selected list of n training images
        '''
        print("loading training samples numpy")
        start = time.time()
        
        random.seed(seed)
        training_dataset = np.load(training_dataset_filename+'.npy')
        l = training_dataset.shape[0]
        random_indices = sample(range(l),n)
        random_data = np.copy(training_dataset[random_indices])
        training_dataset = None
        gc.collect()
        end = time.time()
        duration = end-start
        print("done loading training samples numpy... took " + str(duration)+ " seconds" )
#        for i in range(random_data.shape[0]):
#            print(str(i)+"'th training sample")
#            print(str(random_data[i,-1])+" = class")
#            fig,ax=plt.subplots(ncols=1,nrows=1,figsize = (5,5))
#            ax.imshow(np.reshape(random_data[i,0:100*100],(100,100)))
#            plt.show()
#            
        return random_data
        
    def load_all_gt(self,PATH,list_of_images):
        GT = []
        ground_truth_images = []
        ground_truth_data = []
        ground_truth_indices = []
        
        image_set = np.load(PATH+'all_training_images.npy')[list_of_images,:].astype(bool)
        gc.collect()
        for i in range(len(list_of_images)):
            print('loading image '+str(i))
            
            GT = []
            image = (image_set[i])
            gc.collect()
            ground_truth_filename = "all_" + str(i)
            labelling_obj=ExtractionLabelling(PATH,[],[],[],[],64, 100,100)
            labelling_obj.load_text(ground_truth_filename)
            
            GT = labelling_obj.gt #list of tuples
            if GT == []: continue

            GT_array = np.asarray(GT)

            GT_data = list(map(tuple,GT_array[:,:-1])) # change list of tuples of (x,y,w,h,c) in GT to array, truncate last column, then change back to list of tuples
            extraction_obj = ExtractionPreprocessing(image, '', GT_data)
            ext_images, ext_data = extraction_obj.preprocess_extractions(self.wanted_w, self.wanted_h, self.export_w, self.export_h,
                                                self.max_piece_percent)
            ext_images = np.asarray(ext_images).astype(bool)
            for j in range(len(ext_images)):
                ground_truth_images.append(ext_images[j])
                ground_truth_data.append(ext_data[j])
                ground_truth_indices.append(GT_array[j,-1])
        return ground_truth_images, ground_truth_data, ground_truth_indices
    def predict_from_gt_images_list(self,ground_truth_images, trained_model):
        
        prediction_obj = ComponentClassifierPredict(0.7,0.3)
        ext_class_index, ext_class_name, \
        ext_match_first_max_percent, ext_match_second_max_percent = prediction_obj.predict_classes(ground_truth_images,trained_model)
        return ext_class_index
    
    def predict_from_gt_image(self,ground_truth_image, trained_model_1, trained_model_2=None, trained_model_3=None):
        
        prediction_obj = ComponentClassifierPredict(0.7,0.3)
        if trained_model_2 != None and trained_model_3 != None:
            index_1, first_max_1, second_max_1, index_2, first_max_2, second_max_2, index_3, first_max_3, second_max_3 = prediction_obj.predict_class(ground_truth_image,trained_model_1, trained_model_2, trained_model_3)
            index = prediction_obj.select_most_common_prediction([index_1,index_2,index_3])
        else:
#            index, first_max, second_max = prediction_obj.predict_class(ground_truth_image,trained_model_1)
            index = prediction_obj.predict_class_with_rotations(ground_truth_image,trained_model_1)
        return index
    
    def load_gt_array(self,dataset_PATH, gt_image_num):
#        gt_image = np.load(self.PATH+'all_training_images.npy').astype(bool)

        gc.collect()
        GT = []
        gc.collect()
        ground_truth_filename = "all_" + str(gt_image_num)
        labelling_obj=ExtractionLabelling([],[],[],[],64, 100,100)
        labelling_obj.load_text(dataset_PATH, ground_truth_filename)
        GT = labelling_obj.gt #list of tuples
        GT_array = np.asarray(GT)
        
        return GT_array#, gt_image
    
    def process_gt_image_extractions(self, gt_image, GT_array):
        try:
            GT_data = list(map(tuple,GT_array[:,:-1])) # change list of tuples of (x,y,w,h,c) in GT to array, truncate last column, then change back to list of tuples
        except IndexError:
            x = GT_array[0]
            y = GT_array[1]
            w = GT_array[2]
            h = GT_array[3]
            GT_data =[(x,y,w,h)]
        extraction_obj = ExtractionPreprocessing(gt_image, '', GT_data)
        gt_extraction_list, gt_extraction_data = extraction_obj.preprocess_extractions(self.wanted_w, self.wanted_h, self.export_w, self.export_h,
                                                self.max_piece_percent)
#        print('gt_extraction_list[0]')
#        fig,ax=plt.subplots(ncols=1,nrows=1,figsize = (5,5))
#        ax.imshow(gt_extraction_list[0])
#        plt.show()
#        print(GT_array)
        try:
            gt_indices_list = list(map(int,GT_array[:,-1]))
        except IndexError:
            gt_indices_list = [GT_array[-1]]
#        print('gt_extraction_list')
#        print(gt_extraction_list)
#        print('gt_indices_list')
#        print(gt_indices_list)
        return gt_extraction_list, gt_indices_list
        
    def update_answers(self,dataset_PATH, gt_image_num):
        start = time.time()
        
        print('loading original training set...')
        
        data_all = np.load('Training_Samples_'+str(self.num_classes)+'_classes_'+str(self.img_rows)+'x'+str(self.img_cols)+'_all.npy')

        image = np.load("all_training_images.npy")[:,:,gt_image_num]
        print('finished loading original training set...')
        end = time.time()
        t = end-start
        print('time elapsed loading original training set = ' + str(t))
        
        GT_array = self.load_gt_array(dataset_PATH, gt_image_num)
        
        GT_data = GT_array[:,0:-1]
        GT_indices = GT_array[:,-1]
        extraction_to_be_preprocessed = ExtractionPreprocessing(image,'',GT_data)
        ext_images, ext_data, ext_class_index, ext_class_name = extraction_to_be_preprocessed.preprocess_extractions(self.wanted_w, self.wanted_h, self.export_w, self.export_h,
                                                                                                                    self.max_piece_percent)            
        GT_image_as_row = np.reshape(np.asarray(ext_images),(len(ext_images),self.img_rows*self.img_cols))
        GT_indices = np.reshape(GT_indices,(len(GT_indices),1))
        
        #Load image data as x array
        
        #Put together x and y as single array
        data_ans = np.hstack((GT_image_as_row, GT_indices))
        print('Adding ' + data_ans.shape[0] +' training samples to training set...')
        
        #Add new answers to old answers
        combined_data = list(data_all) #data_all in shape (number of training samples, img_rows*img_cols + 1)
        for i in range(data_ans.shape[0]):
            combined_data.append(list(data_ans[i,:]))
            
        data_all = np.asarray(combined_data)
        gc.collect()
        
        print('Final shape = '+ data_all.shape)
        #Save data
        np.save('Training_Samples_'+str(self.num_classes)+'_classes_'+str(self.img_rows)+'x'+str(self.img_cols)+'_all',data_all)
        end = time.time()
        duration = end-start
        print('time elapsed updating answers using list = ' + str(duration))
        
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
    def test_classifier_all(self, dataset_PATH, dataset_name, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL,iters,seed,max_problem_images): 
        #training_dataset_filename example: Training_Samples_64_classes_100x100_all
        # ground_truth_filename example: all_44
        #test n number of samples
        #for k times
        
        '''
        Inputs: image, ground_truth_data, ground_truth_index
        Outputs: accuracy of model classifier only (using same segmentations of ground truth)
        '''
        random.seed(seed)
#        x=[]
#        y=[]
        
#        n_was_list = True
        # load ground truth data        


        gc.collect()
        prediction_indices = []
        ground_truth_indices = []

        f = open(dataset_PATH+'testing_results_'+str(iters)+'.txt','a')
        # train model
        training_obj = ComponentClassifierTraining(dataset_PATH, "Training_Samples_64_classes_100x100_all", 64, 0, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL)
        training_obj.X_train, training_obj.y_train, training_obj.X_val, training_obj.y_val, training_obj.X_test, training_obj.y_test = training_obj.shuffle_data(np.load(dataset_PATH+dataset_name+'.npy'),seed)
        training_obj.model = training_obj.load_sketch_a_net_model(0, 64, training_obj.X_train.shape[1:])
        
#        training_obj.train(iters,seed)
        PATH = 'C:/Users/JustinSanJuan/Desktop/Workspace/python/Testing Folder/' #must have "/" at the end
        
        name = 'Sketch-a-Net_64_classes_100x100_0.0_all_100epochs'
        training_obj.model.load_weights(PATH+name+'.h5')

        
        trained_model = training_obj.model
#        del training_obj
        gc.collect()
        # test on all samples
        for gt_image_num in range(max_problem_images):
            gc.collect()
            print('testing on image number: ' + str(gt_image_num))
            gt_data_path_string = dataset_PATH+'GT/'+'GT_all_'+str(gt_image_num)+'.txt'
            
            if not(os.path.isfile(gt_data_path_string)): continue
            
            GT_array = self.load_gt_array(dataset_PATH, gt_image_num)
            if len(GT_array) == 0: continue
            if gt_image_num <400:
                gt_image = np.load(self.PATH+'all_training_images_1.npy').astype(bool)[:,:,gt_image_num]
            elif gt_image_num >=400 and gt_image_num <800:
                gt_image = np.load(self.PATH+'all_training_images_2.npy').astype(bool)[:,:,gt_image_num-400]

            gt_extraction_list_temp, gt_indices_list_temp = self.process_gt_image_extractions(gt_image, GT_array)
            
            gt_indices_list = []
            gt_extraction_list = []
            #remove class 23's from extractions
            for gt_list_index in range(len(gt_indices_list_temp)):
                if (int(gt_indices_list_temp[gt_list_index]) != 23):# and (int(gt_indices_list_temp[gt_list_index]) < 48):
                    gt_indices_list.append(gt_indices_list_temp[gt_list_index]) 
                    gt_extraction_list.append(gt_extraction_list_temp[gt_list_index])
            
            for gt_extraction_num in range(len(gt_indices_list)):
                gc.collect()
                prediction_index = (self.predict_from_gt_image(gt_extraction_list[gt_extraction_num], trained_model))
                
                ground_truth_index = gt_indices_list[gt_extraction_num]
                prediction_indices.append(prediction_index)
                ground_truth_indices.append(ground_truth_index)

#            print(ground_truth_indices)
#            print(prediction_indices)
#            
        accuracy = calculate_accuracy(prediction_indices, ground_truth_indices)

        gc.collect()
            
        f.writelines(str(accuracy)+'\n')
        f.writelines(str(prediction_indices)+'\n')
        f.writelines(str(ground_truth_indices)+'\n')
        f.close()
        del trained_model
        gc.collect()

        from helper_functions import plot_confusion_matrix
        from constants import target_names_all
        from sklearn.metrics import confusion_matrix
#        import itertools
        cnf_matrix = confusion_matrix(np.asarray(ground_truth_indices),
                    np.asarray(prediction_indices))
        plot_confusion_matrix(cnf_matrix, classes=target_names_all,
                      title='Confusion matrix')
        del prediction_indices
        del ground_truth_indices
        gc.collect()
        
        return 
    def test_classifier_multiple(self, dataset_PATH, dataset_name_list, num_classes,dropout, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL,iters,seed,start,end,weights_name = None): 
        #training_dataset_filename example: Training_Samples_64_classes_100x100_all
        # ground_truth_filename example: all_44
        #test n number of samples
        #for k times
        
        '''
        Inputs: image, ground_truth_data, ground_truth_index
        Outputs: accuracy of model classifier only (using same segmentations of ground truth)
        '''
        random.seed(seed)
#        x=[]
#        y=[]
        
#        n_was_list = True
        # load ground truth data        


        gc.collect()
        prediction_indices = []
        ground_truth_indices = []

        f = open(dataset_PATH+'testing_results_'+str(iters)+'.txt','a')

        seed = 1000
        training_obj = ComponentClassifierTraining(num_classes, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL)
        #Model is Sketch_a_net
        training_obj.model = training_obj.load_sketch_a_net_model(dropout, num_classes,(100,100,1))
        if weights_name == None:
            training_obj.train_from_multiple_files(100,seed,dataset_PATH,dataset_name_list,verbose = 1)
        else:
            training_obj.model.load_weights(dataset_PATH+weights_name+'.h5')

        trained_model = training_obj.model
#        del training_obj
        gc.collect()
        # test on all samples
        for gt_image_num in range(start,end):
            gc.collect()
            print('testing on image number: ' + str(gt_image_num))
            gt_data_path_string = dataset_PATH+'GT/'+'GT_all_'+str(gt_image_num)+'.txt'
            
            if not(os.path.isfile(gt_data_path_string)): continue
            
            GT_array = self.load_gt_array(gt_image_num)
            if len(GT_array) == 0: continue
            if gt_image_num <400:
                gt_image = np.load(self.PATH+'all_training_images_1.npy').astype(bool)[:,:,gt_image_num]
            elif gt_image_num >=400 and gt_image_num <800:
                gt_image = np.load(self.PATH+'all_training_images_2.npy').astype(bool)[:,:,gt_image_num-400]
            elif gt_image_num >=800 and gt_image_num <1200:
                gt_image = np.load(self.PATH+'all_training_images_3.npy').astype(bool)[:,:,gt_image_num-800]
            elif gt_image_num >=1200 and gt_image_num <1440:
                gt_image = np.load(self.PATH+'all_training_images_4.npy').astype(bool)[:,:,gt_image_num-1200]


            
            gt_extraction_list_temp, gt_indices_list_temp = self.process_gt_image_extractions(gt_image, GT_array)
            del gt_image
#            gt_indices_list = []
#            gt_extraction_list = []
            #remove class 23's from extractions
            for gt_list_index in range(len(gt_indices_list_temp)):
                if (int(gt_indices_list_temp[gt_list_index]) != 23):# and (int(gt_indices_list_temp[gt_list_index]) < 48):
#                    gt_indices_list.append(gt_indices_list_temp[gt_list_index]) 
#                    gt_extraction_list.append(gt_extraction_list_temp[gt_list_index])
            
                    gc.collect()
                    prediction_index = (self.predict_from_gt_image(gt_extraction_list_temp[gt_list_index], trained_model))
                
                    ground_truth_index = gt_indices_list_temp[gt_list_index]
                    prediction_indices.append(prediction_index)
                    ground_truth_indices.append(ground_truth_index)

#        print(ground_truth_indices)
#        print(prediction_indices)
        
        accuracy = calculate_accuracy(prediction_indices, ground_truth_indices)

        gc.collect()
            
        f.writelines(str(accuracy)+'\n')
        f.writelines(str(prediction_indices)+'\n')
        f.writelines(str(ground_truth_indices)+'\n')
        f.close()
        del trained_model
        gc.collect()

        from helper_functions import plot_confusion_matrix
        from constants import target_names_all
        from sklearn.metrics import confusion_matrix
#        import itertools
        cnf_matrix = confusion_matrix(np.asarray(ground_truth_indices),
                    np.asarray(prediction_indices))
        plot_confusion_matrix(cnf_matrix, classes=target_names_all,
                      title='Confusion matrix')
        del prediction_indices
        del ground_truth_indices
        gc.collect()
        
        return 
    def test_classifier_multiple_slow(self, dataset_PATH, dataset_name_list, num_classes,dropout, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL,iters,seed,start,end,weights_name = None): 
        #training_dataset_filename example: Training_Samples_64_classes_100x100_all
        # ground_truth_filename example: all_44
        #test n number of samples
        #for k times
        
        '''
        Inputs: image, ground_truth_data, ground_truth_index
        Outputs: accuracy of model classifier only (using same segmentations of ground truth)
        '''
        random.seed(seed)
#        x=[]
#        y=[]
        
#        n_was_list = True
        # load ground truth data        


        gc.collect()
        prediction_indices = []
        ground_truth_indices = []

        f = open(dataset_PATH+'testing_results_'+str(iters)+'.txt','a')

        seed = 1000
        training_obj = ComponentClassifierTraining(num_classes, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL)
        #Model is Sketch_a_net
        training_obj.model = training_obj.load_sketch_a_net_model(dropout, num_classes,(100,100,1))
        if weights_name == None:
            training_obj.train_from_multiple_files(100,seed,dataset_PATH,dataset_name_list,verbose = 1)
        else:
            training_obj.model.load_weights(dataset_PATH+weights_name+'.h5')

        trained_model = training_obj.model
#        del training_obj
        gc.collect()
        # test on all samples
        for gt_image_num in range(start,end):
            gc.collect()
            print('testing on image number: ' + str(gt_image_num))
            gt_data_path_string = dataset_PATH+'GT/'+'GT_all_'+str(gt_image_num)+'.txt'
            
            if not(os.path.isfile(gt_data_path_string)): continue
            
            GT_array = self.load_gt_array(dataset_PATH, gt_image_num)
            if len(GT_array) == 0: continue
            
            if gt_image_num < 400:
                gt_image = np.load(self.PATH+'all_training_images_1.npy')[:,:,gt_image_num]
            elif gt_image_num >=400 and gt_image_num <800:
                gt_image = np.load(self.PATH+'all_training_images_2.npy')[:,:,gt_image_num-400]
            elif gt_image_num >=800 and gt_image_num <1200:
                gt_image = np.load(self.PATH+'all_training_images_3.npy')[:,:,gt_image_num-800]
            elif gt_image_num >=1200 and gt_image_num <1440:
                gt_image = np.load(self.PATH+'all_training_images_4.npy')[:,:,gt_image_num-1200]
#            print(gt_image)  
            gc.collect()

            for GT_line in GT_array:
                gt_extraction_list_temp, gt_indices_list_temp = self.process_gt_image_extractions(gt_image, GT_line)
#                del gt_image
    #            gt_indices_list = []
    #            gt_extraction_list = []
                #remove class 23's from extractions
                if (int(gt_indices_list_temp[0]) != 23):# and (int(gt_indices_list_temp[gt_list_index]) < 48):
#                    gt_indices_list.append(gt_indices_list_temp[gt_list_index]) 
#                    gt_extraction_list.append(gt_extraction_list_temp[gt_list_index])
            
                    gc.collect()
                    prediction_index = (self.predict_from_gt_image(gt_extraction_list_temp[0], trained_model))
                
                    ground_truth_index = gt_indices_list_temp[0]
                    prediction_indices.append(prediction_index)
                    ground_truth_indices.append(ground_truth_index)

#            print(ground_truth_indices)
#            print(prediction_indices)
            
        accuracy = calculate_accuracy(prediction_indices, ground_truth_indices)

        gc.collect()
            
        f.writelines(str(accuracy)+'\n')
        f.writelines(str(prediction_indices)+'\n')
        f.writelines(str(ground_truth_indices)+'\n')
        f.close()
        del trained_model
        gc.collect()

        from helper_functions import plot_confusion_matrix
        from constants import target_names_all
        from sklearn.metrics import confusion_matrix
#        import itertools
        cnf_matrix = confusion_matrix(np.asarray(ground_truth_indices),
                    np.asarray(prediction_indices))
        plot_confusion_matrix(cnf_matrix, classes=target_names_all,
                      title='Confusion matrix')
#        del prediction_indices
#        del ground_truth_indices
        gc.collect()
        
        return ground_truth_indices, prediction_indices
    def test_classifier(self, training_dataset_filename, train_ratio, k,list_of_n,iters,seed): 
        #training_dataset_filename example: Training_Samples_64_classes_100x100_all
        # ground_truth_filename example: all_44
        #test n number of samples
        #for k times
        
        '''
        Inputs: image, ground_truth_data, ground_truth_index
        Outputs: accuracy of model classifier only (using same segmentations of ground truth)
        '''
        random.seed(seed)
#        x=[]
#        y=[]
        
#        n_was_list = True
        # load ground truth data        
        if (type(list_of_n) is int) or (type(list_of_n) is np.int64) or (type(list_of_n) is float): 
            print('n was int')
            list_of_n = [list_of_n]
#        step = 1
#        list_images = list(map(int,np.arange(0,150,step)))

        gc.collect()
        for n in list_of_n:
            for i in range(k): #number of iterations
                gc.collect()
                prediction_indices = []
                ground_truth_indices = []

                seed = int(random.random()*10000)
                random.seed(seed)
                f = open(self.PATH+'testing_results_'+str(iters)+'.txt','a')
                print('n = ' + str(n) + ', k = ' +str(i+1))
                # train model
                training_obj = ComponentClassifierTraining(self.PATH, "Training_Samples_64_classes_100x100_all", 64, 0, train_ratio, 1-train_ratio)
                training_obj.X_train, training_obj.y_train, training_obj.X_val, training_obj.y_val, training_obj.X_test, training_obj.y_test = training_obj.shuffle_data(self.pick_random_training_samples(training_dataset_filename, n,seed),seed)
                training_obj.model = training_obj.load_sketch_a_net_model(0, 64, training_obj.X_train.shape[1:])
                
                training_obj.train(iters,seed)
                
                trained_model = training_obj.model
                training_obj = None
                gc.collect()
                # test on all samples
                for gt_image_num in range(len(list_images)):
                    gc.collect()
                    print('testing on image number: ' + str(gt_image_num*step))
                    gt_data_path_string = self.PATH+'GT/'+'GT_all_'+str(gt_image_num*step)+'.txt'
                    
                    if not(os.path.isfile(gt_data_path_string)): continue
                    
                    GT_array = self.load_gt_array(gt_image_num*5)

                    gt_image = np.load(self.PATH+'all_training_images_1.npy').astype(bool)[:,:,gt_image_num*step]
                    
                    gt_extraction_list_temp, gt_indices_list_temp = self.process_gt_image_extractions(gt_image, GT_array)
                    
                    gt_indices_list = []
                    gt_extraction_list = []
                    #remove class 23's from extractions
                    for gt_list_index in range(len(gt_indices_list_temp)):
                        if (int(gt_indices_list_temp[gt_list_index]) != 23) and (int(gt_indices_list_temp[gt_list_index]) < 48):
                            gt_indices_list.append(gt_indices_list_temp[gt_list_index])
                            gt_extraction_list.append(gt_extraction_list_temp[gt_list_index])
                    
                    for gt_extraction_num in range(len(gt_indices_list)):
                        gc.collect()
                        prediction_index = (self.predict_from_gt_image(gt_extraction_list[gt_extraction_num], trained_model))
                        
                        ground_truth_index = gt_indices_list[gt_extraction_num]
                        prediction_indices.append(prediction_index)
                        ground_truth_indices.append(ground_truth_index)

                print(ground_truth_indices)
                print(prediction_indices)
                
                accuracy = calculate_accuracy(prediction_indices, ground_truth_indices)

                gc.collect()
                    
                f.writelines(str(n)+' '+str(accuracy)+'\n')
                f.close()
                del training_obj
                del trained_model
                gc.collect()

        del prediction_indices
        del ground_truth_indices
        gc.collect()
        
        return
    
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

    def test_classifier_remapped(self, training_dataset_filename, train_ratio, k,list_of_n,iters,seed): 
        #training_dataset_filename example: Training_Samples_64_classes_100x100_all
        # ground_truth_filename example: all_44
        #test n number of samples
        #for k times
        
        '''
        Inputs: image, ground_truth_data, ground_truth_index
        Outputs: accuracy of model classifier only (using same segmentations of ground truth)
        '''
        random.seed(seed)
#        x=[]
#        y=[]
        
#        n_was_list = True
        # load ground truth data        
        if (type(list_of_n) is int) or (type(list_of_n) is np.int64) or (type(list_of_n) is float): 
            print('n was int')
            list_of_n = [list_of_n]
#            n_was_list = False
        step = 5
        list_images = list(map(int,np.arange(0,150,step)))
#        gt_image_set = np.load(self.PATH+'all_training_images.npy').astype(bool)[:,:,list_images]
        
        gc.collect()
        for n in list_of_n:
            for i in range(k):
                gc.collect()
                prediction_indices = []
                ground_truth_indices = []
#                print(seed)
                seed = int(random.random()*10000)
                random.seed(seed)
                f = open(self.PATH+'testing_results_'+str(iters)+'.txt','a')
                print('n = ' + str(n) + ', k = ' +str(i+1))
                # pick random training samples
                # train model
                training_obj = ComponentClassifierTraining(self.PATH, "Training_Samples_64_classes_100x100_all", 64, 0, train_ratio, 1-train_ratio)
                training_obj.X_train, training_obj.y_train, training_obj.X_val, training_obj.y_val, training_obj.X_test, training_obj.y_test = training_obj.shuffle_data(self.pick_random_training_samples(training_dataset_filename, n,seed),seed)
                training_obj.model = training_obj.load_sketch_a_net_model(0, 64, training_obj.X_train.shape[1:])
                
                training_obj.train(iters,seed)
                
                trained_model = training_obj.model
                training_obj = None
                gc.collect()
                # test on all samples
                for gt_image_num in range(len(list_images)):
                    gc.collect()
                    print('testing on image number: ' + str(gt_image_num*step))
                    gt_data_path_string = self.PATH+'GT/'+'GT_all_'+str(gt_image_num*step)+'.txt'
                    
                    if not(os.path.isfile(gt_data_path_string)): continue
                    
                    GT_array = self.load_gt_array(gt_image_num*5)
#                        fig,ax=plt.subplots(ncols=1,nrows=1,figsize = (5,5))
#                        ax.imshow(gt_image)
#                        plt.show()
#                    gt_image = gt_image_set[:,:,gt_image_num].astype(bool)
                    gt_image = np.load(self.PATH+'all_training_images_1.npy').astype(bool)[:,:,gt_image_num*step]
                    
                    gt_extraction_list_temp, gt_indices_list_temp = self.process_gt_image_extractions(gt_image, GT_array)
                    
                    gt_indices_list = []
                    gt_extraction_list = []
                    #remove class 23's from extractions
                    for gt_list_index in range(len(gt_indices_list_temp)):
                        if (int(gt_indices_list_temp[gt_list_index]) != 23) and (int(gt_indices_list_temp[gt_list_index]) < 48):
                            gt_indices_list.append(gt_indices_list_temp[gt_list_index])
                            gt_extraction_list.append(gt_extraction_list_temp[gt_list_index])
                    
                    for gt_extraction_num in range(len(gt_indices_list)):
                        gc.collect()
                        prediction_index = self.map_index(self.predict_from_gt_image(gt_extraction_list[gt_extraction_num], trained_model))
                        ground_truth_index = self.map_index(gt_indices_list[gt_extraction_num])
                        
                        prediction_indices.append(prediction_index)
                        ground_truth_indices.append(ground_truth_index)
                        
#                        fig,ax=plt.subplots(ncols=1,nrows=1,figsize = (5,5))
#                        ax.imshow(gt_extraction_list[gt_extraction_num])
#                        plt.show()
#
#                        print('prediction_index')
#                        print(prediction_index)
#                        print('ground_truth_index')
#                        print(ground_truth_index)

                # calculate accuracy & return string
                print(ground_truth_indices)
                print(prediction_indices)
                
                accuracy = calculate_accuracy(prediction_indices, ground_truth_indices)
                
#                x.append(n)
#                y.append(accuracy)
#                print(accuracy)
#                print(y)
                gc.collect()
                    
                f.writelines(str(n)+' '+str(accuracy)+'\n')
                f.close()
                del training_obj
                del trained_model
                gc.collect()
#        if n_was_list == True:
#            #plot and save graph
#            x = np.asarray(x)
#            y = np.asarray(y)
#            fig,ax=plt.subplots(ncols=1,nrows=1,figsize = (15,15))
#            ax.scatter(x,y)
#            ax.set_xlabel("Training Samples Size")
#            ax.set_ylabel("Accuracy")
#            plt.show()
#            ax.figure.savefig("Accuracy Scatterplot for "+str(k*int(len(list_of_n)))+'_'+" samples")
        
#        max_accuracy = max(y)
        del prediction_indices
        del ground_truth_indices
        gc.collect()
        
        return
    def test_classifier_remapped_load_1_model(self, training_dataset_filename, train_ratio, k,list_of_n,iters,seed): 
        #training_dataset_filename example: Training_Samples_64_classes_100x100_all
        # ground_truth_filename example: all_44
        #test n number of samples
        #for k times
        
        '''
        Inputs: image, ground_truth_data, ground_truth_index
        Outputs: accuracy of model classifier only (using same segmentations of ground truth)
        '''
        random.seed(seed)
#        x=[]
#        y=[]
        
#        n_was_list = True
        # load ground truth data        
        if (type(list_of_n) is int) or (type(list_of_n) is np.int64) or (type(list_of_n) is float): 
            print('n was int')
            list_of_n = [list_of_n]
#            n_was_list = False
        step = 5
        list_images = list(map(int,np.arange(0,150,step)))
#        gt_image_set = np.load(self.PATH+'all_training_images.npy').astype(bool)[:,:,list_images]
        
        gc.collect()
        for n in list_of_n:
            gc.collect()
            prediction_indices = []
            ground_truth_indices = []
#                print(seed)
            seed = int(random.random()*10000)
            random.seed(seed)
            
            f = open(self.PATH+'testing_results_'+str(iters)+'.txt','a')
            print('n = ' + str(n) + ', k = ' +str(k+1))
            # pick random training samples
            # train model
            training_obj = ComponentClassifierTraining(self.PATH, "Training_Samples_64_classes_100x100_all", 64, 0, train_ratio, 1-train_ratio)
            training_obj.X_train, training_obj.y_train, training_obj.X_val, training_obj.y_val, training_obj.X_test, training_obj.y_test = training_obj.shuffle_data(self.pick_random_training_samples(training_dataset_filename, n,seed),seed)
            training_obj.model = training_obj.load_sketch_a_net_model(0, 64, training_obj.X_train.shape[1:])
            
            training_obj.model.load_weights('Sketch-a-Net_64_classes_100x100_0.0_all_100epochs_21.h5')
            trained_model_1 = training_obj.model
            
            training_obj = None
            gc.collect()
            # test on all samples
            for gt_image_num in range(len(list_images)):
                gc.collect()
                print('testing on image number: ' + str(gt_image_num*step))
                gt_data_path_string = self.PATH+'GT/'+'GT_all_'+str(gt_image_num*step)+'.txt'
                
                if not(os.path.isfile(gt_data_path_string)): continue
                
                GT_array = self.load_gt_array(gt_image_num*5)
#                        fig,ax=plt.subplots(ncols=1,nrows=1,figsize = (5,5))
#                        ax.imshow(gt_image)
#                        plt.show()
#                    gt_image = gt_image_set[:,:,gt_image_num].astype(bool)
                gt_image = np.load(self.PATH+'all_training_images_1.npy').astype(bool)[:,:,gt_image_num*step]
                
                gt_extraction_list_temp, gt_indices_list_temp = self.process_gt_image_extractions(gt_image, GT_array)
                
                gt_indices_list = []
                gt_extraction_list = []
                #remove class 23's from extractions
                for gt_list_index in range(len(gt_indices_list_temp)):
                    if (int(gt_indices_list_temp[gt_list_index]) != 23) and (int(gt_indices_list_temp[gt_list_index]) < 48):
                        gt_indices_list.append(gt_indices_list_temp[gt_list_index])
                        gt_extraction_list.append(gt_extraction_list_temp[gt_list_index])
                
                for gt_extraction_num in range(len(gt_indices_list)):
                    gc.collect()
                    prediction_index = self.map_index(self.predict_from_gt_image(gt_extraction_list[gt_extraction_num], trained_model_1))
                    ground_truth_index = self.map_index(gt_indices_list[gt_extraction_num])
                    
                    prediction_indices.append(prediction_index)
                    ground_truth_indices.append(ground_truth_index)
                    
#                        fig,ax=plt.subplots(ncols=1,nrows=1,figsize = (5,5))
#                        ax.imshow(gt_extraction_list[gt_extraction_num])
#                        plt.show()
#
#                        print('prediction_index')
#                        print(prediction_index)
#                        print('ground_truth_index')
#                        print(ground_truth_index)

            # calculate accuracy & return string
            print(ground_truth_indices)
            print(prediction_indices)
            
            accuracy = calculate_accuracy(prediction_indices, ground_truth_indices)
            
#                x.append(n)
#                y.append(accuracy)
#                print(accuracy)
#                print(y)
            gc.collect()
                
            f.writelines(str(n)+' '+str(accuracy)+' '+str(k)+'\n')
            f.close()

            gc.collect()
#        if n_was_list == True:
#            #plot and save graph
#            x = np.asarray(x)
#            y = np.asarray(y)
#            fig,ax=plt.subplots(ncols=1,nrows=1,figsize = (15,15))
#            ax.scatter(x,y)
#            ax.set_xlabel("Training Samples Size")
#            ax.set_ylabel("Accuracy")
#            plt.show()
#            ax.figure.savefig("Accuracy Scatterplot for "+str(k*int(len(list_of_n)))+'_'+" samples")
        
#        max_accuracy = max(y)

        gc.collect()
        
        return
    
    def test_classifier_remapped_load_3_models(self, training_dataset_filename, train_ratio, k,list_of_n,iters,list_images,seed): 
        #training_dataset_filename example: Training_Samples_64_classes_100x100_all
        # ground_truth_filename example: all_44
        #test n number of samples
        #for k times
        
        '''
        Inputs: image, ground_truth_data, ground_truth_index
        Outputs: accuracy of model classifier only (using same segmentations of ground truth)
        '''
        random.seed(seed)
#        x=[]
#        y=[]
        
#        n_was_list = True
        # load ground truth data        
        if (type(list_of_n) is int) or (type(list_of_n) is np.int64) or (type(list_of_n) is float): 
            print('n was int')
            list_of_n = [list_of_n]
#            n_was_list = False
#        gt_image_set = np.load(self.PATH+'all_training_images.npy').astype(bool)[:,:,list_images]
        
        gc.collect()
        for n in list_of_n:
            gc.collect()
            prediction_indices = []
            ground_truth_indices = []
#                print(seed)
            seed = int(random.random()*10000)
            random.seed(seed)
            
            f = open(self.PATH+'testing_results_'+str(iters)+'.txt','a')
            print('n = ' + str(n) + ', k = ' +str(k+1))
            # pick random training samples
            # train model
            training_obj = ComponentClassifierTraining(self.PATH, "Training_Samples_64_classes_100x100_all", 64, 0, train_ratio, 1-train_ratio)
            training_obj.X_train, training_obj.y_train, training_obj.X_val, training_obj.y_val, training_obj.X_test, training_obj.y_test = training_obj.shuffle_data(self.pick_random_training_samples(training_dataset_filename, n,seed),seed)
            training_obj.model = training_obj.load_sketch_a_net_model(0, 64, training_obj.X_train.shape[1:])
            
#            training_obj.model.load_weights('Sketch-a-Net_64_classes_100x100_0.0_all_100epochs_5.h5')
#            trained_model_1 = training_obj.model

            gc.collect()
            # test on all samples
            for gt_image_num in list_images:
                gc.collect()
                print('testing on image number: ' + str(gt_image_num))
                gt_data_path_string = self.PATH+'GT/'+'GT_all_'+str(gt_image_num)+'.txt'
                
                if not(os.path.isfile(gt_data_path_string)): continue
                
                GT_array = self.load_gt_array(gt_image_num)
#                        fig,ax=plt.subplots(ncols=1,nrows=1,figsize = (5,5))
#                        ax.imshow(gt_image)
#                        plt.show()
#                    gt_image = gt_image_set[:,:,gt_image_num].astype(bool)
                if gt_image_num <400:
                    gt_image = np.load(self.PATH+'all_training_images_1.npy').astype(bool)[:,:,gt_image_num]
                elif gt_image_num >=400 and gt_image_num <800:
                    gt_image = np.load(self.PATH+'all_training_images_2.npy').astype(bool)[:,:,gt_image_num-400]
                elif gt_image_num >=800 and gt_image_num <1200:
                    gt_image = np.load(self.PATH+'all_training_images_3.npy').astype(bool)[:,:,gt_image_num-800]
                elif gt_image_num >=1200 and gt_image_num <1600:
                    gt_image = np.load(self.PATH+'all_training_images_4.npy').astype(bool)[:,:,gt_image_num-1200]
                
                gt_extraction_list_temp, gt_indices_list_temp = self.process_gt_image_extractions(gt_image, GT_array)
                
                gt_indices_list = []
                gt_extraction_list = []
                #remove class 23's from extractions
                for gt_list_index in range(len(gt_indices_list_temp)):
                    if (int(gt_indices_list_temp[gt_list_index]) != 23) and (int(gt_indices_list_temp[gt_list_index]) < 48):
                        gt_indices_list.append(gt_indices_list_temp[gt_list_index])
                        gt_extraction_list.append(gt_extraction_list_temp[gt_list_index])
                
                for gt_extraction_num in range(len(gt_indices_list)):
                    gc.collect()
                    
                    training_obj.model.load_weights('Sketch-a-Net_64_classes_100x100_0.0_all_100epochs_5.h5')
                    trained_model_1 = training_obj.model
                    prediction_index_1 = self.map_index(self.predict_from_gt_image(gt_extraction_list[gt_extraction_num], trained_model_1))
                    
                    training_obj.model.load_weights('Sketch-a-Net_64_classes_100x100_0.0_all_100epochs_18.h5')
                    trained_model_2 = training_obj.model
                    prediction_index_2 = self.map_index(self.predict_from_gt_image(gt_extraction_list[gt_extraction_num], trained_model_2))
                    
                    training_obj.model.load_weights('Sketch-a-Net_64_classes_100x100_0.0_all_100epochs_21.h5')
                    trained_model_3 = training_obj.model
                    prediction_index_3 = self.map_index(self.predict_from_gt_image(gt_extraction_list[gt_extraction_num], trained_model_3))
                    
                    prediction_index = self.select_most_common_prediction([prediction_index_1,prediction_index_2,prediction_index_3])
                    
                    ground_truth_index = self.map_index(gt_indices_list[gt_extraction_num])
                    
                    prediction_indices.append(prediction_index)
                    ground_truth_indices.append(ground_truth_index)
                    
#                        fig,ax=plt.subplots(ncols=1,nrows=1,figsize = (5,5))
#                        ax.imshow(gt_extraction_list[gt_extraction_num])
#                        plt.show()
#
#                        print('prediction_index')
#                        print(prediction_index)
#                        print('ground_truth_index')
#                        print(ground_truth_index)

            # calculate accuracy & return string
            print(ground_truth_indices)
            print(prediction_indices)
            
            accuracy = calculate_accuracy(prediction_indices, ground_truth_indices)
            
#                x.append(n)
#                y.append(accuracy)
#                print(accuracy)
#                print(y)
            gc.collect()
                
            f.writelines(str(n)+' '+str(accuracy)+' '+str(k)+'\n')
            f.close()

            gc.collect()
#        if n_was_list == True:
#            #plot and save graph
#            x = np.asarray(x)
#            y = np.asarray(y)
#            fig,ax=plt.subplots(ncols=1,nrows=1,figsize = (15,15))
#            ax.scatter(x,y)
#            ax.set_xlabel("Training Samples Size")
#            ax.set_ylabel("Accuracy")
#            plt.show()
#            ax.figure.savefig("Accuracy Scatterplot for "+str(k*int(len(list_of_n)))+'_'+" samples")
        
#        max_accuracy = max(y)

        gc.collect()
        
        return
    def test_classifier_and_segmentation(self, ext_images, ext_data, ext_class_index, ext_class_names, ground_truth_data, ground_truth_index):
        '''
        Input: image, ext_images, ext_data, ext_class_index, ext_class_names
        Output: Accuracy calculation
        '''
        # Check overlap scores
        #overlap is calculated by (A & B) / (A or B), where A is the ground truth area, and B is the extracted image area
        
        # If not enough overlap, correct answer is 23, if enough overlap, take correct class from ground_truth_index
        #output: list of predicted classes vs correct classes
        
        
        #calculate accuracy