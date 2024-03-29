import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from extraction_preprocessing import ExtractionPreprocessing
import time
from keras import backend as K
import numpy as np
from component_classifier_predict import ComponentClassifierPredict
from constants import target_names_all, target_names
import os.path
import gc
from helper_functions import print_image_bw

class ExtractionLabelling(object):

    def __init__(self,ext_images,ext_data,ext_class_index,ext_class_name, num_classes, img_rows,img_cols, min_percent_match, min_confidence):
        self.ext_images = ext_images
        self.ext_data = ext_data
        self.ext_class_index = ext_class_index
        self.ext_class_name = ext_class_name
        self.num_classes = num_classes
        self.img_rows = img_rows
        print("img_rows =" + str(img_rows))
        print("img_cols =" + str(img_cols))
        self.img_cols = img_cols
        self.gt =[]
        self.extraction = []
        self.image = []
        self.min_percent_match = min_percent_match
        self.min_confidence = min_confidence

        #reset labeller answers
        self.answer = ''
        self.value = ''
        self.check = ''
        self.delete = ''
        
        self.prediction_index = ''
        self.x,self.y,self.w,self.h =0, 0, 0, 0
        


        #create empty set of ground truth lines
        self.lines=[]

        #create empty lists of exractions and data
        self.ext_class_index_temp=[]
        self.ext_class_name_temp=[]
        self.ext_images_temp=[]
        self.ext_data_temp=[]
        
        self.model = None
        gc.enable()

    def define_model(self,model):
        self.model = model
        
    def print_problem_image(self,image,k=None):
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
        ax.imshow(image,cmap = 'binary') #show problem image

        if k == 'correct':
            for k in range(len(self.ext_data_temp)):
                if self.ext_class_index_temp[k]!=23:
                    x, y, w, h = self.ext_data_temp[k]
                    rect = mpatches.Rectangle(
                        (x, y), w, h, fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)
                #need to add classes
            tick_spacing = 20
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            plt.xticks(rotation=70)
            plt.grid()

        elif k == 'review':
            print('printing ground truths for review...')
            for k in range(len(self.gt)):
                x, y, w, h, c = self.gt[k]
                if c != 23:
                    rect = mpatches.Rectangle(
                        (x, y), w, h, fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)
        else:
#            for k in range(len(self.ext_data)):
                #plot bounding box k on problem image
            x, y, w, h = self.ext_data[k]
            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect) #show bounding boxes from ext_data

        plt.show()

    def print_extraction_image(self,image,k=None, x=None, y=None,w=None,h=None):
        
        #plot only the extraction image (to show magnified)
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(1.5, 1.5))
        
        if k == None or k =='coords':
            rect = mpatches.Rectangle(
                    (x, y), w, h, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
        else:
            x1=self.ext_data[k][0]
            y1=self.ext_data[k][1]
            x2=x1+self.ext_data[k][2]
            y2=y1+self.ext_data[k][3]
    
            ext_image_from_image = image[y1:y2,x1:x2]
            ax.imshow(ext_image_from_image)
        plt.show()

    def request_answer(self,k):
        self.answer = ''
        self.answer = input(str(k+1)+'/'+str(len(self.ext_data))+' '
               'Prediction is '+ str(self.ext_class_name[k])+'\n'+
               'If it is correct, press enter, otherwise enter correct class index' + '\n'+
               'Else if it is a bad segmentation, enter "b"' +'\n'+ 
               'If you want to go back to the previous segmentation enter "r"'+'\n')

    def answer_is_enter(self,k):
        #create string based on answer as '' = enter
        print('Data was correct')
        self.ext_class_index_temp.append(self.ext_class_index[k])
        self.ext_class_name_temp.append(self.ext_class_name[k])
        self.ext_images_temp.append(self.ext_images[k])
        self.ext_data_temp.append(self.ext_data[k])
        print(len(self.ext_data_temp))
        print(len(self.ext_data))
        
        #create string to write as a single line in ground truth file
        string = str(self.ext_data[k][0])+' '+ str(self.ext_data[k][1])+' '+ str(self.ext_data[k][2])+' '+ str(self.ext_data[k][3])+' '+str(self.ext_class_index[k])
        print(string)
        return string

    def answer_is_b(self,k):
        #create string based on answer as b for bad extraction
        self.ext_class_index_temp.append(int(23))
        self.ext_class_name_temp.append(target_names_all[23])
        self.ext_images_temp.append(self.ext_images[k])
        self.ext_data_temp.append(self.ext_data[k])

        #create string for ground truth file with class = 23 for unknown
        string = str(self.ext_data[k][0])+' '+ str(self.ext_data[k][1])+' '+ str(self.ext_data[k][2])+' '+ str(self.ext_data[k][3])+' '+str(23)
        return string

    def answer_is_number(self,k):
        self.ext_class_index_temp.append(int(self.answer))
        self.ext_class_name_temp.append(target_names_all[int(self.answer)])
        self.ext_images_temp.append(self.ext_images[k])
        self.ext_data_temp.append(self.ext_data[k])
            
        string = str(self.ext_data[k][0])+' '+ str(self.ext_data[k][1])+' '+ str(self.ext_data[k][2])+' '+ str(self.ext_data[k][3])+' '+str(self.answer)
        return string

    def parse_answer(self,image,k):
        #ask labeller to confirm if prediction is correct
        '''
        Possible responses: 
        - '' (just press enter) --> save correct answers to lists
        - b --> bad segmentation labelled as class 23 (unknown)
        - r --> revert by removing previous answer from list and ask about previous answer again
        - c --> close the labelleing
        - [class number] --> save extraction but with new class number as indicated 

        '''
        continuecode = 1
        
        if self.answer =='' or self.answer ==' ':
            string = self.answer_is_enter(k)
            
        elif self.answer =='b' or self.answer =='"b"' or self.answer == "'b'":
            string = self.answer_is_b(k)
            
        elif self.answer =='r': #revert
            #remove previous answer from list
            self.ext_class_index_temp= self.ext_class_index_temp[0:-1]
            self.ext_class_name_temp= self.ext_class_name_temp[0:-1]
            self.ext_images_temp=self.ext_images_temp[0:-1]
            self.ext_data_temp=self.ext_data_temp[0:-1]
            self.lines = self.lines[:-1]
            
            self.print_problem_image(image, k=k-1)
            self.print_extraction_image(image,k=k-1)
            
            #ask for answer in previous k
            self.request_answer(k-1)
            continuecode = self.parse_answer(image,k-1)
            if continuecode == 0:
                return 0
            #ask for answer in current k again
            
            self.print_problem_image(image,k=k)
            self.print_extraction_image(image,k=k)
            
            self.request_answer(k)
            continuecode = self.parse_answer(image,k)
            if continuecode == 0:
                return 0
            string=''
        elif self.answer =='c':
            #write currently recorded ground truths and end process
            self.f.writelines([item for item in self.lines])
            self.f.close()
            print('saved and closed file')
            continuecode = 0
            return continuecode
        else:
            string = self.answer_is_number(k)
    
        if string != '':
            self.lines.append(string+'\n')
#        print(self.ext_data_temp)
        return continuecode
    
    def request_complete(self):
        self.complete = ''
        self.complete = input('If above figure captures all objects, press enter'+'\n'+'Otherwise, enter "a"'+'\n')

    def parse_complete(self,image):
        continuecode = 1
        if self.complete == '' or self.complete == ' ' or self.complete =='c':
            self.f.writelines([item for item in self.lines])
            self.f.close()
            continuecode = 0 #stop the loop
        else:
            x1=int(input('Enter the top left corner x coordinate'+'\n'))
            y1=int(input('Enter the top left corner y coordinate'+'\n'))
            x2=int(input('Enter the bottom right corner x coordinate'+'\n'))
            y2=int(input('Enter the bottom right corner y coordinate'+'\n'))
            
            self.x=x1
            self.y=y1
            self.w = x2-x1
            self.h = y2-y1
            #do prediction
            
            extraction=image[y1:y2,x1:x2]
            ExtractionPreprocessingObj = ExtractionPreprocessing(extraction, "", [])
            extraction = ExtractionPreprocessingObj.resize_extraction(extraction,x1,y1, x2-x1,y2-y1,self.img_cols,self.img_rows,self.img_cols,self.img_rows)
            self.extraction = extraction
            
            num_channel = 3 # since we need RGB, then expand dimensions of extraction
            prediction = ComponentClassifierPredict(self.min_percent_match, self.min_confidence)
            extraction = prediction.expand_dimension(extraction,num_channel)
            first_index, second_index, first_max, second_max = prediction.predict_class(extraction, self.model)
            self.prediction_index = int(prediction.use_entropy(first_index, first_max, second_max))
            self.ext_images.append(extraction)
            self.ext_data.append((x1,y1,x2-x1,y2-y1))
            self.ext_class_index.append(self.prediction_index)
            self.ext_class_name.append(target_names_all[self.prediction_index])
            """
            self.ext_images_temp.append(extraction)
            self.ext_data_temp.append((x1,y1,x2-x1,y2-y1))
            self.ext_class_index_temp.append(self.prediction_index)
            self.ext_class_name_temp.append(target_names_all[self.prediction_index])
            """

            self.print_extraction_image(image, k=len(self.ext_class_index)-1)
            
            #print('Prediction is ' +str(target_names_all[self.prediction_index]))
            #if labeller just pressed enter, then save predicted index
            #else, save the number entered as the class index
        
        return continuecode
    
    def request_check(self):
        self.check= input('Please enter the class index if prediciton is incorrect'+'\n'
                     + 'Otherwise, press enter to confirm prediction'+'\n')
            
    def parse_check(self,image):
        data=(self.x, self.y, self.w, self.h)
        self.ext_data_temp.append(data)
        
        if self.check == '' or self.check == ' ':
            string = str(self.x)+' '+str(self.y)+' '+str(self.w)+' '+str(self.h)+' '+str(self.prediction_index)
            self.lines.append(string+'\n')
            self.ext_images_temp.append(self.extraction)
            self.ext_class_index_temp.append(self.prediction_index)
            self.ext_class_name.append(target_names_all[self.prediction_index])
        else:
            class_index = int(self.check)
            string = str(self.x)+' '+str(self.y)+' '+str(self.w)+' '+str(self.h)+' '+str(class_index)
            self.lines.append(string+'\n')
            self.ext_images_temp.append(self.extraction)
            self.ext_class_index_temp.append(class_index)
            self.ext_class_name.append(target_names_all[class_index])
    
    def request_delete(self):
        self.delete = input('If you want to delete the previous input, enter "y"'+'\n'+'To close: enter "c"'+'\n'+'To continue: press enter'+'\n')
    def parse_delete(self,image):
        continuecode = 1
        if self.delete =='y' or self.delete=='"y"' or self.delete=="'y'":
            self.lines = self.lines[:-1]
        elif self.delete=='c'or self.delete=='"c"' or self.delete=="'c'":
            self.f.writelines([item for item in self.lines])
            self.f.close()
            continuecode = 0
        return continuecode

    def select_good_bounding_boxes(self,image,PATH, imagename):
        '''
        Supply a GUI to efficiently label ground truths.
        Give choices to confirm bounding box & class, and also to add more bounding boxes or delete incorrect labels.
        
        Inputs: Image, extraction data (image and coordinates from search function), extraction predictions (index and name), imagename
        Outputs: Saves text file of correct answers
        '''
        self.lines = []
        
        filename = PATH +'GT/' + 'GT_'+str(imagename)
        mode = 'w' #input('mode of file (w:write,a:append, include "+" to create if not there'+'\n'))
        overwrite = 'y'
        if os.path.isfile(PATH + 'GT_' + str(imagename)+ '.txt'):
            overwrite=input("A text file is already there under the name: " + str(imagename) +'\n' + "Do you want to overwrite it? (y/n)" + '\n')
        if overwrite != 'y':
            print('cancelling')
            return
        self.f = open(str(filename)+'.txt',str(mode)) #f = file where ground truths will be saved
        self.lines.append(str(imagename)+'\n')
        print(self.lines)
        
        continuecode = 1
        #for each image in list of extraction data
        for k in range(len(self.ext_data)):
        
            self.print_problem_image(image,k=k)
            self.print_extraction_image(image,k=k)
            
            #ask labeller to confirm if prediction is correct
            self.request_answer(k)
            continuecode = self.parse_answer(image,k)
            if continuecode == 0:
                return

        continuecode = 1
        while continuecode:
            self.print_problem_image(image,'correct')

            self.request_complete()
            continuecode = self.parse_complete(image)
            if continuecode == 0:
                break
            #print(self.ext_class_index[len(self.ext_class_index)-1])
            self.request_answer(len(self.ext_class_index)-1)
            continuecode = self.parse_answer(image,len(self.ext_class_index)-1)
            if continuecode == 0:
                break
            #print(self.ext_class_index)
            
            self.print_problem_image(image,'correct')
            
            self.request_delete()
            continuecode = self.parse_delete(image)
            if continuecode == 0:
                break
    def load_text(self,PATH,imagename): #imagename example = all_44
        filename = PATH +'GT/' + 'GT_'+str(imagename)
        if os.path.isfile(str(filename)+'.txt'):
            f = open(str(filename)+'.txt')
            lines = [line.rstrip('\n') for line in f]
            lines = lines[1::] #remove header
            self.gt = []
            for l in lines:
                if not(l == '' or l ==' ' or l =='\n'):
                    self.gt.append(tuple(map(int,l.split(" "))))
    def load_bounding_boxes(self,PATH,imagename): #imagename example = all_44
        filename = PATH +'GT/' + 'GT_'+str(imagename)
        if os.path.isfile(str(filename)+'.txt'):
            f = open(str(filename)+'.txt')
            lines = [line.rstrip('\n') for line in f]
            lines = lines[1::] #remove header
            gt = []
            for l in lines:
                if not(l == '' or l ==' ' or l =='\n'):
                    xywhc = tuple(map(int,l.split(" ")))
                    x,y,w,h,c = xywhc
                    if c != 23:
                        gt.append((x,y,w,h))
        return gt
            #print(self.gt)
        #return self.gt
    def plot_ground_truths(self,image,PATH, imagename):
        self.image = image
        self.load_text(PATH,imagename) # create groundtruth array, modifies self.gt
        self.print_problem_image(image,k='review')

    def update_answers_list(self, PATH, name, start, end, exclude = []):
        
        print('Updating answers...')
        #load gt_list
        
        start = int(start)
        end = int(end)
        
        #get ans
        data_ans = []
#        ans = []
#        ext_images = []
        
        self.gt = []
        loaded_image_name = ''
        for j in range(start, end):
            self.gt=[]
            if j < 400 and loaded_image_name != 'all_training_images_1':
                print('loading imageset')
                loaded_image_name = 'all_training_images_1'
                imageset = np.load(PATH+loaded_image_name+".npy")
                print('loaded imageset')
            elif j < 800 and j >=400 and loaded_image_name != 'all_training_images_2':
                print('loading imageset')
                loaded_image_name = 'all_training_images_2'
                imageset = np.load(PATH+loaded_image_name+".npy")
                print('loaded imageset')
            elif j < 1200 and j >= 800 and loaded_image_name != 'all_training_images_3':
                print('loading imageset')
                loaded_image_name = 'all_training_images_3'
                imageset = np.load(PATH+loaded_image_name+".npy")
                print('loaded imageset')
            elif j < 1440 and j >= 1200 and loaded_image_name != 'all_training_images_4':
                print('loading imageset')
                loaded_image_name = 'all_training_images_4'
                imageset = np.load(PATH+loaded_image_name+".npy")
                print('loaded imageset')
                
            if j < 400:
                image = imageset[:,:,j]
            elif j < 800 and j >= 400:
                image = imageset[:,:,j-400]
            elif j < 1200 and j >= 800:
                image = imageset[:,:,j-800]
            elif j < 1440 and j >= 1200:
                image = imageset[:,:,j-1200]

            imagename =  "all_"+str(int(j))
            print(PATH+'GT/GT_'+str(imagename)+'.txt')
            # don't add if missing GT data
            if not(os.path.isfile(PATH+'GT/GT_'+str(imagename)+'.txt')):
                continue
            print('imagename = '+ str(imagename))
            self.load_text(imagename)
#            print(self.gt)
            
#            self.plot_ground_truths(image,imagename)
            for i in range(len(self.gt)):
          #load ext_images
#                print_image_bw(image,5,5)
#                print(1)
                x, y, w, h, c = self.gt[i]
                if exclude != []:
                   if c in exclude: continue
#                print(c)
                x1 = x
                x2 = x+w
                y1 = y
                y2 = y+h
                extraction = image[y1:y2,x1:x2]
#                print_image_bw(extraction,5,5)
                extraction_obj = ExtractionPreprocessing(image, '', '')
                extraction = extraction_obj.preprocess_extraction(extraction, 100,100,100,100, 0.3, x, y, w, h)
#                extraction = ExtractionPreprocessing.resize_extraction(extraction)
#                print_image_bw(extraction,5,5)
    
                #preprocess
                data_line = np.reshape(extraction,(100*100))
                ans_line = np.asarray(int(c))
                new_data_line = np.hstack((data_line,ans_line))
                
                data_ans.append(new_data_line.astype(np.int).tolist())
#        print(data_ans)
        del image
        del imageset
        gc.collect()
        
        print('Adding ' + str(len(data_ans)) +' training samples to training set...')
        
        #Load current answers
        data_all=np.load(PATH+name+'.npy')
        print('Inital length = '+ str(data_all.shape[0]))
        
        #Add new answers to old answers
        combined_data = []
        for i in range(data_all.shape[0]):
            print(str(i)+' / ' +str(data_all.shape[0])+ ' of data_all')
            data_line = data_all[i].astype(np.int)
            combined_data.append(data_line.tolist())
        del data_all
        for i in range(len(data_ans)):
            print(str(i)+' / ' +str(len(data_ans))+' of data_ans')
            #if duplicate, skip
#            print(data_ans[i])
#            print(combined_data)
            if data_ans[i] in combined_data:
                continue

            data_line = data_ans[i]
            combined_data.append(data_line)
        del data_ans
        
        gc.collect()
        new_dataset_shape = len(combined_data)
        print('Final shape = '+ str(new_dataset_shape))
        #Save data
        new_dataset_name = 'Training_Samples_'+str(self.num_classes)+'_classes_'+str(self.img_rows)+'x'+\
            str(self.img_cols)+'_all_cleaned_updated_'+str(new_dataset_shape)+"_("+str(start)+"-"+str(end)+")"        
        np.save(PATH+new_dataset_name,np.asarray(combined_data))
        print('saved as: '+PATH+new_dataset_name+'.npy')
        
        return new_dataset_name
    def concatenate_datasets(self, dataset_PATH, dataset_name_list, start, end, suffix = None, exclude = []):
        
        print('Updating answers...')
        #load gt_list
        
        start = int(start)
        end = int(end)
        
        #get ans
        data_ans = []
#        ans = []
#        ext_images = []
        
        self.gt = []
        loaded_image_name = ''
        for j in range(start, end):
            self.gt=[]
            if j < 400 and loaded_image_name != 'all_training_images_1':
                print('loading imageset')
                loaded_image_name = 'all_training_images_1'
                imageset = np.load(dataset_PATH+loaded_image_name+".npy")
                print('loaded imageset')
            elif j < 800 and j >=400 and loaded_image_name != 'all_training_images_2':
                print('loading imageset')
                loaded_image_name = 'all_training_images_2'
                imageset = np.load(dataset_PATH+loaded_image_name+".npy")
                print('loaded imageset')
            elif j < 1200 and j >= 800 and loaded_image_name != 'all_training_images_3':
                print('loading imageset')
                loaded_image_name = 'all_training_images_3'
                imageset = np.load(dataset_PATH+loaded_image_name+".npy")
                print('loaded imageset')
            elif j < 1440 and j >= 1200 and loaded_image_name != 'all_training_images_4':
                print('loading imageset')
                loaded_image_name = 'all_training_images_4'
                imageset = np.load(dataset_PATH+loaded_image_name+".npy")
                print('loaded imageset')
                
            if j < 400:
                image = imageset[:,:,j]
            elif j < 800 and j >= 400:
                image = imageset[:,:,j-400]
            elif j < 1200 and j >= 800:
                image = imageset[:,:,j-800]
            elif j < 1440 and j >= 1200:
                image = imageset[:,:,j-1200]
            
#            #skip if image is emppty
#            labelled_array, max_label = measure.label(image, background=0, connectivity=2, return_num=True)
#            if max_label == 0:continue
            
            imagename =  "all_"+str(int(j))
            print(dataset_PATH+'GT/GT_'+str(imagename)+'.txt')
            # don't add if missing GT data
            if not(os.path.isfile(dataset_PATH+'GT/GT_'+str(imagename)+'.txt')):
                continue
#            print('imagename = '+ str(imagename))
            self.load_text(imagename)
#            print(self.gt)
            if len(self.gt) == 0: continue
#            self.plot_ground_truths(image,imagename)
            for i in range(len(self.gt)):
          #load ext_images
#                print_image_bw(image,5,5)
#                print(1)
                x, y, w, h, c = self.gt[i]
                if exclude != []:
                   if c in exclude: continue
#                print(c)
                x1 = x
                x2 = x+w
                y1 = y
                y2 = y+h
                extraction = image[y1:y2,x1:x2]
#                print_image_bw(extraction,5,5)
                extraction_obj = ExtractionPreprocessing(image, '', '')
                extraction = extraction_obj.preprocess_extraction(extraction, 100,100,100,100, 0.3, x, y, w, h)
#                extraction = ExtractionPreprocessing.resize_extraction(extraction)
#                print_image_bw(extraction,5,5)
    
                #preprocess
                data_line = np.reshape(extraction,(100*100))
                ans_line = np.asarray(int(c))
                new_data_line = np.hstack((data_line,ans_line))
                
                data_ans.append(new_data_line.astype(np.int).tolist())
#        print(data_ans)
        image = None
        imageset = None
        gc.collect()
        
        print('Adding ' + str(len(data_ans)) +' training samples to training set...')
        
        combined_data = []
        #Load current answers
        for dataset_name in dataset_name_list:
            data_all=np.load(dataset_PATH+dataset_name+'.npy')
            gc.collect()
            print('Inital length = '+ str(data_all.shape[0]))
        #Add new answers to old answers
            for i in range(data_all.shape[0]):
                print(str(i)+' / ' +str(data_all.shape[0])+ ' of data_all')
                #add rules here:
                if data_all[i].tolist() in combined_data: continue
                data_line = data_all[i].astype(np.int)
                combined_data.append(data_line.tolist())
            data_all = None
            gc.collect()
        
        gc.collect()
        new_dataset_shape = len(combined_data)
        print('Final shape = '+ str(new_dataset_shape))
        #Save data
        new_dataset_name = 'Training_Samples_'+str(self.num_classes)+'_classes_'+str(self.img_rows)+'x'+\
            str(self.img_cols)+'_all_cleaned_updated_'+str(new_dataset_shape)+"_("+str(start)+"-"+str(end)+")"        
        if suffix != None:
            new_dataset_name = new_dataset_name + '_'+str(suffix)
        np.save(dataset_PATH+new_dataset_name,np.asarray(combined_data))
        print('saved as: '+dataset_PATH+new_dataset_name+'.npy')
        
        return new_dataset_name
    def clean_datasets(self, PATH, name_list,suffix = None, delete_list=[], resize_list = [], swap_list = [],swap_index_list=[]):
        new_name_list = []
        for name in name_list:
            new_name = self.clean_dataset(PATH,name, suffix=suffix, delete_list=delete_list, resize_list = resize_list, swap_list = swap_list,swap_index_list=swap_index_list)
            new_name_list.append(new_name)
        return new_name_list
    def clean_dataset(self,PATH,name, suffix=None, delete_list=[], resize_list = [], swap_list = [],swap_index_list=[]):
#        from constants import target_names_all
        from skimage import measure
        complete_data_set = np.load(PATH+name+'.npy')
        label_set = complete_data_set[:,-1]#get correct answer labels
        data_set = complete_data_set[:,:-1] # remove correct answer labels
        temp_complete_data_set = []
        for i in range(data_set.shape[0]):
#            print(i)
            if i in delete_list:
                    continue
            elif i in resize_list:
                continue #originally fix it
            elif i in swap_list:
                list_index = swap_list.index(i)
                new_index = swap_index_list[list_index]
                #add data with new index
                data_set_line = data_set[i,:]
                label_set_line = np.asarray(new_index)
                new_data_line = np.hstack((data_set_line,label_set_line))
                temp_complete_data_set.append(new_data_line)
            else:
                #check if empty
                extraction = np.reshape(data_set[i,:],(100,100))
                labelled_array, max_label = measure.label(extraction, background=0, connectivity=2, return_num=True)
                if max_label == 0:
                    continue
                
                data_set_line = data_set[i,:]
                label_set_line = label_set[i]
                new_data_line = np.hstack((data_set_line,label_set_line))
                temp_complete_data_set.append(new_data_line)
        #change back to array
        complete_data_set = np.asarray(temp_complete_data_set)
        final_dataset_name = "Training_Samples_64_classes_100x100_all_cleaned_"+str(complete_data_set.shape[0])
        if suffix != None:
            final_dataset_name = final_dataset_name + '_'+str(suffix)
        np.save(PATH+final_dataset_name, complete_data_set)
        print('saved as :'+ str(PATH) + final_dataset_name)
        return final_dataset_name