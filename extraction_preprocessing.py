from image_preprocessing import ImagePreprocessing

class ExtractionPreprocessing(object):
    """
    
    """



    def __init__():
        self.___ = ___
        pass
        # define padding function, to be used in preprocessing

    def padwithzeros(self, vector, pad_width, iaxis, kwargs):
            # pad vector (image) in all directions with 0's for a length of pad_width
        vector[:pad_width[0]] = 0
        vector[-pad_width[1]:] = 0
        return vector

     # Edge-trim, remove unconnected edge pieces, resize, then predict classes
    def preprocess_extractions(self, image,candidates, wanted_w, wanted_h, export_w, export_h):
        for x,y,w,h in candidates:
            # Given x,y,w,h, store each extraction and coordinates in lists    
            extraction = np.copy(image[y:y+h,x:x+w])
            
            labelled_array, max_label = measure.label(extraction, background=0, connectivity=1, return_num=True)
            
    def trim_extraction(self,extraction,)
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