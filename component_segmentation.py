class ComponentSegmentation(object):
            
    def search(image, good_candidates, bad_candidates, _scale, _sigma, _min_size):
        """
        #function: use connectivity and selective search to create candidate bounding boxes for classification
        """

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


    def plot_bounding_boxes_with_names(image, candidates, name):
        """
        # create fig1, ax1, create single subplot, then draw bounding boxes x, y, w, and save figure with name of model
        """
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
