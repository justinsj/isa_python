import matplotlib.ticker as ticker

class ExtractionLabelling(object):
    
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