import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from extraction_preprocessing import ExtractionPreprocessing
import time
from keras import backend as K
import numpy as np

class ExtractionLabelling(object):

    def __init__(self,PATH,ext_images,ext_data,ext_class_index,ext_class_name,target_names,target_names_all, num_classes, img_rows,img_cols):
        self.PATH = PATH
        self.ext_images = ext_images
        self.ext_data = ext_data
        self.ext_class_index = ext_class_index
        self.ext_class_name = ext_class_name
        self.target_names = target_names
        self.target_names_all = target_names_all
        self.num_classes = num_classes
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.gt =[]

        #reset labeller answers
        self.answer = ''
        self.value = ''
        self.delete = ''


        #create empty set of ground truth lines
        self.lines=[]

        #create empty lists of exractions and data
        self.ext_class_index_temp=[]
        self.ext_class_name_temp=[]
        self.ext_images_temp=[]
        self.ext_data_temp=[]

    def print_problem_image(self,k=None):
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
        ax.imshow(self.image) #show problem image

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
            for k in range(len(self.ext_data)):
                #plot bounding box k on problem image
                x, y, w, h = self.ext_data[k]
                rect = mpatches.Rectangle(
                    (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
                ax.add_patch(rect) #show bounding boxes from ext_data

        plt.show()

    def print_extraction_image(self,k):
        #plot only the extraction image (to show magnified)
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(1.5, 1.5))
        x1=self.ext_data[k][0]
        y1=self.ext_data[k][1]
        x2=x1+self.ext_data[k][2]
        y2=y1+self.ext_data[k][3]

        ext_image_from_image = self.image[y1:y2,x1:x2]
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
        
        #create string to write as a single line in ground truth file
        string = str(self.ext_data[k][0])+' '+ str(self.ext_data[k][1])+' '+ str(self.ext_data[k][2])+' '+ str(self.ext_data[k][3])+' '+str(self.ext_class_index[k])
        return string

    def answer_is_b(self,k):
        #create string based on answer as b for bad extraction
        self.ext_class_index_temp.append(int(23))
        self.ext_class_name_temp.append(self.target_names_all[23])
        self.ext_images_temp.append(self.ext_images[k])
        self.ext_data_temp.append(self.ext_data[k])

        #create string for ground truth file with class = 23 for unknown
        string = str(self.ext_data[k][0])+' '+ str(self.ext_data[k][1])+' '+ str(self.ext_data[k][2])+' '+ str(self.ext_data[k][3])+' '+str(23)
        return string

    def answer_is_number(self,k):
        self.ext_class_index_temp.append(int(self.answer))
        self.ext_class_name_temp.append(self.target_names_all[int(self.answer)])
        self.ext_images_temp.append(self.ext_images[k])
        self.ext_data_temp.append(self.ext_data[k])
            
        string = str(self.ext_data[k][0])+' '+ str(self.ext_data[k][1])+' '+ str(self.ext_data[k][2])+' '+ str(self.ext_data[k][3])+' '+str(self.answer)
        return string

    def parse_answer(self,k):
        #ask labeller to confirm if prediction is correct
        '''
        Possible responses: 
        - '' (just press enter) --> save correct answers to lists
        - b --> bad segmentation labelled as class 23 (unknown)
        - r --> revert by removing previous answer from list and ask about previous answer again
        - c --> close the labelleing
        - [class number] --> save extraction but with new class number as indicated 

        '''
        exitcode = 1

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
            
            #ask for answer in previous k
            self.request_answer(k-1)
            self.parse_answer(k-1)

            #ask for answer in current k again
            self.request_answer(k)
            self.parse_answer(k)
            
        elif self.answer =='c':
            #write currently recorded ground truths and end process
            self.f.writelines([item for item in self.lines])
            self.f.close()
            exitcode = 0
        else:
            string = self.answer_is_number(k)
    
        if string != '':
            self.lines.append(string+'\n')
        print(self.ext_data_temp)
        return exitcode
    
    def request_value(self):
        self.value = ''
        self.value = input('If above figure captures all objects, press enter'+'\n'+'Otherwise, enter "a"'+'\n')

    def parse_value(self):
        exitcode = 1
        if self.value == '' or self.value == ' ' or self.value =='c':
            self.f.writelines([item for item in self.lines])
            self.f.close()
            exitcode = 0 #stop the loop
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
            extraction=self.image[y1:y2,x1:x2]
            ext = self.resize_extraction(extraction)
            ext_im=ext[:,:]
            num_channel = 3 # since we need RGB, then expand dimensions of extraction
            
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

            '''
            prediction=model.predict(ext_im)[0]
            prediction_index = int(prediction.tolist().index(max(prediction)))
         
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
            ax.imshow(self.image)
            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            plt.show()
            
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(1.5, 1.5))
            ax.imshow(ext)
            plt.show()

            print('Prediction is ' +str(self.target_names_all[prediction_index]))
            #if labeller just pressed enter, then save predicted index
            #else, save the number entered as the class index
            check= input('Please enter the class index if prediciton is incorrect'+'\n'
                         + 'Otherwise, press enter to confirm prediction'+'\n')
            if check == '' or check == ' ':
                string = str(x1)+' '+str(y1)+' '+str(w)+' '+str(h)+' '+str(prediction_index)
                self.lines.append(string+'\n')
                self.ext_images_temp.append(ext)
                data=(x1,y1,w,h)
                self.ext_data_temp.append(data)
                self.ext_class_index_temp.append(prediction_index)
                self.ext_class_name.append(self.target_names_all[prediction_index])
            else:
                class_index = int(check)
                string = str(x1)+' '+str(y1)+' '+str(w)+' '+str(h)+' '+str(class_index)
                self.lines.append(string+'\n')
                self.ext_images_temp.append(ext)
                data=(x1,y1,w,h)
                self.ext_data_temp.append(data)
                self.ext_class_index_temp.append(class_index)
                self.ext_class_name.append(self.target_names_all[class_index])
                '''
        return exitcode
    def request_delete(self):
        self.delete = input('If you want to delete the previous input, enter "y"'+'\n'+'To close: enter "c"'+'\n'+'To continue: press enter'+'\n')
    def parse_delete(self):
        exitcode = 1
        if self.delete =='y' or self.delete=='"y"' or self.delete=="'y'":
            self.lines = self.lines[:-1]
        elif self.delete=='c'or self.delete=='"c"' or self.delete=="'c'":
            self.f.writelines([item for item in self.lines])
            self.f.close()
            exitcode = 0
        return exitcode

    def select_good_bounding_boxes(self,image,imagename):
        '''
        Supply a GUI to efficiently label ground truths.
        Give choices to confirm bounding box & class, and also to add more bounding boxes or delete incorrect labels.
        
        Inputs: Image, extraction data (image and coordinates from search function), extraction predictions (index and name), imagename
        Outputs: Saves text file of correct answers
        '''
        
        self.image = image
        self.imagename = imagename
        
        filename = self.PATH +'GT/' + 'GT_'+str(imagename)
        mode = 'w+' #input('mode of file (w:write,a:append, include "+" to create if not there'+'\n'))
        self.f = open(str(filename)+'.txt',str(mode)) #f = file where ground truths will be saved
        self.lines.append(str(imagename)+'\n')


        #for each image in list of extraction data
        for k in range(len(self.ext_data)):

            self.print_problem_image(k)
            self.print_extraction_image(k)
            
            #ask labeller to confirm if prediction is correct
            self.request_answer(k)
            self.parse_answer(k)


        exitcode = 1
        while exitcode:
            self.print_problem_image('correct')

            self.request_value()
            exitcode = self.parse_value()
            if exitcode == 0:
                break
            
            self.print_problem_image('correct')
            
            self.request_delete()
            exitcode = self.parse_delete()
            if exitcode == 0:
                break
    def load_text(self,imagename):
        filename = self.PATH +'GT/' + 'GT_'+str(imagename)
        f = open(str(filename)+'.txt')
        lines = [line.rstrip('\n') for line in f]
        lines = lines[1::] #remove header
        self.gt = []
        for l in lines:
            self.gt.append(tuple(map(int,l.split(" "))))
        print(self.gt)
    def plot_ground_truths(self,image,imagename):
        self.image = image
        self.load_text(imagename) # create groundtruth array, modifies self.gt
        self.print_problem_image('review')

    #Stack new images and answers then add to training samples
    def update_training_set(self,image, ext_images, ext_data, ans):

        start = time.time()
        print('Updating answers...')
        
        #Load answers as a vertical column array
        y_ans = np.transpose(np.asarray(ans).reshape(1,len(ans)))
        print('y array has shape: ' + str(y_ans.shape))
        
        #Load image data as x array
        x_ans = np.asarray(ext_images).reshape(len(ext_images),self.img_rows*self.img_cols)
        print('x array has shape: ' + str(x_ans.shape))
        
        #Put together x and y as single array
        data_ans = np.hstack((x_ans,y_ans))
        print('Adding ' + data_ans.shape[0] +' training samples to training set...')
        
        #Load current answers
        data_all=np.load('Training_Samples_'+str(self.num_classes)+'_classes_'+str(self.img_rows)+'x'+str(self.img_cols)+'_all.npy')
        print('Inital length = '+ len(data_all))

        data_all = np.vstack((data_all,data_ans))
        print('Final shape = '+ data_all.shape)
        #Save data
        np.save('Training_Samples_'+str(self.num_classes)+'_classes_'+str(self.img_rows)+'x'+str(self.img_cols)+'_all',data_all)
        end = time.time()
        duration = end-start
        print('time elapsed updating answers vstack = ' + str(duration))
        
    def update_training_set_image(self,image, imagename, max_piece_percent, wanted_w,wanted_h, export_w,export_h):
        start = time.time()
        print('Updating answers...')
        
        self.image = image
        self.load_text(imagename)
        data_all = np.load(self.PATH+'Training_Samples_'+str(self.num_classes)+'_classes_'+str(self.img_rows)+'x'+str(self.img_cols)+'_all.npy')

        answer_set = [(i[0], i[1], i[2], i[3]) for i in self.gt]
        extraction_to_be_preprocessed = ExtractionPreprocessing(image,'',answer_set)
        ext_images, ext_data, ext_class_index, ext_class_name = extraction_to_be_preprocessed.preprocess_extractions(wanted_w, wanted_h, export_w, export_h,
                                                                                                                    max_piece_percent)            
        #Load answers as a vertical column array
        answer_list = [i[4] for i in self.gt]
        y_ans = np.asarray(answer_list).reshape(len(answer_list),1)
        print('y array has shape: ' + str(y_ans.shape))
        
        #Load image data as x array
        x_ans = np.asarray(ext_images).reshape(len(ext_images),self.img_rows*self.img_cols)
        print('x array has shape: ' + str(x_ans.shape))
        
        #Put together x and y as single array
        data_ans = np.hstack((x_ans,y_ans))
        print('Adding ' + data_ans.shape[0] +' training samples to training set...')
        
        data_all = np.vstack((data_all,data_ans))
        print('Final shape = '+ data_all.shape)
        #Save data
        np.save('Training_Samples_'+str(self.num_classes)+'_classes_'+str(self.img_rows)+'x'+str(self.img_cols)+'_all',data_all)
        end = time.time()
        duration = end-start
        print('time elapsed updating answers vstack = ' + str(duration))
    
    def update_training_set_image_as_list(self,image, imagename, max_piece_percent, wanted_w,wanted_h, export_w,export_h):
        start = time.time()
        print('Updating answers...')
        
        self.image = image
        self.load_text(imagename)
        data_all = np.load(self.PATH+'Training_Samples_'+str(self.num_classes)+'_classes_'+str(self.img_rows)+'x'+str(self.img_cols)+'_all.npy')

        answer_set = [(i[0], i[1], i[2], i[3]) for i in self.gt]
        extraction_to_be_preprocessed = ExtractionPreprocessing(image,'',answer_set)
        ext_images, ext_data, ext_class_index, ext_class_name = extraction_to_be_preprocessed.preprocess_extractions(wanted_w, wanted_h, export_w, export_h,
                                                                                                                    max_piece_percent)            
        #Load answers as a vertical column array
        answer_list = [i[4] for i in self.gt]
        y_ans = np.asarray(answer_list).reshape(len(answer_list),1)
        print('y array has shape: ' + str(y_ans.shape))
        
        #Load image data as x array
        x_ans = np.asarray(ext_images).reshape(len(ext_images),self.img_rows*self.img_cols)
        print('x array has shape: ' + str(x_ans.shape))
        
        #Put together x and y as single array
        data_ans = np.hstack((x_ans,y_ans))
        print('Adding ' + data_ans.shape[0] +' training samples to training set...')
        
        #Add new answers to old answers
        combined_data = []
        for i in range(data_all.shape[0]):
            data_line = data_all[i,:]
            combined_data.append(data_line)
        for i in range(data_ans.shape[0]):
            data_line = data_ans[i,:]
            combined_data.append(data_line)
        data_all = np.asarray(combined_data)
        
        print('Final shape = '+ data_all.shape)
        #Save data
        np.save('Training_Samples_'+str(self.num_classes)+'_classes_'+str(self.img_rows)+'x'+str(self.img_cols)+'_all',data_all)
        end = time.time()
        duration = end-start
        print('time elapsed updating answers vstack = ' + str(duration))
        
    def update_answers_list(self,ext_images, ext_data, ans):
        start = time.time()

        print('Updating answers...')
        
        #Load answers as a vertical column array
        y_ans = np.transpose(np.asarray(ans).reshape(1,len(ans)))
        print('y array has shape: ' + str(y_ans.shape))
        
        #Load image data as x array
        x_ans = np.asarray(ext_images).reshape(len(ext_images),self.img_rows*self.img_cols)
        print('x array has shape: ' + str(x_ans.shape))
        
        #Put together x and y as single array
        data_ans = np.hstack((x_ans,y_ans))
        print('Adding ' + data_ans.shape[0] +' training samples to training set...')
        
        #Load current answers
        data_all=np.load('Training_Samples_'+str(self.num_classes)+'_classes_'+str(self.img_rows)+'x'+str(self.img_cols)+'_all.npy')
        print('Inital length = '+ len(data_all))

        #Add new answers to old answers
        combined_data = []
        for i in range(data_all.shape[0]):
            data_line = data_all[i,:]
            combined_data.append(data_line)
        for i in range(data_ans.shape[0]):
            data_line = data_ans[i,:]
            combined_data.append(data_line)
        data_all = np.asarray(combined_data)
        print('Final shape = '+ data_all.shape)
        #Save data
        np.save('Training_Samples_'+str(self.num_classes)+'_classes_'+str(self.img_rows)+'x'+str(self.img_cols)+'_all',data_all)
        end = time.time()
        duration = end-start
        print('time elapsed updating answers list = ' + str(duration))