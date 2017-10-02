class ComponentSegmentation(object):
    
    def __init__(self,image,name):
        self.name = name
        self.image = image
        self.temp_set=[]
        self.premerged_set=[]
        self.merged_set=[]
        self.used_set=set()

    def merge_set(self):
        # Loop merging algorithm
        # Declare empty set for merging
        merged_set = set()
        # List is necessary for looping
        merged_set_list=[]
        # Declare empty set for used rows
        used_set = set()

        # convert list of lists into matrix to do column swaps
        self.premerged_set = np.asmatrix(self.premerged_set)

        if premerged_set.shape[1] < 1:
            continue

        # modify matrix from [x,y,w,h] into [x1,x2,y1,y2]
        x_col = self.premerged_set[:,0])
        y_col = self.premerged_set[:,1])
        w_col = self.premerged_set[:,2])
        h_col = self.premerged_set[:,3])

        x1_col = x_col
        x2_col = x1_col + w_col
        y1_col = y_col
        y2_col = y_col + h_col
        premerged_set = np.asmatrix(np.array([x1_col, x2_col, y1_col, y2_col]))

        for loop_count in range(overlap_repeats + 1):
            
            # set skip variable to false. When true, it will stop using row u to check overlap with other rows.
            skip = False
            
            # cycle through row u of [x1,x2,y1,y2] matrix
            for u in range(premerged_set.shape[0]):
                
                # When skip == True, it means row u has been merged with another row, so 'continue' to stop checking row u with other rows.
                if skip==True:
                    skip=False
                    continue
                
                # Set test_row as [x1,x2,y1,y2] data in row u
                test_row=premerged_set[u,:]
                
                # Cycle through each row n of [x1,x2,y1,y2] matrix and
                # Check if row u overlaps with other rows n in matrix
                for n in range(premerged_set.shape[0]):
                    
                    # Skip checking of row n with row u if u ==n. If checking different row, check for overlap
                    if u != n:
                        
                        # Check if region n is outside of region u
                        # xi' is the xi of the fixed test [x1,x2,y1,y2]
                        # If x1 is greater than x2', or x2 is less than x1' ,then same with y direction
                        # If this statement is passed, it means u does not overlap with n
                        if premerged_set[n,0] > test_row[0,1] or premerged_set[n,1] < test_row[0,0] or premerged_set[n,2] > test_row[0,3] or premerged_set[n,3] < test_row[0,2]:
                            
                            # Change test back into (x,y,w,h) as tuple called test1 to be put in good_candidates
                            test_row=tuple([test_row.tolist()[0][0],test_row.tolist()[0][2],test_row.tolist()[0][1]-test_row.tolist()[0][0],test_row.tolist()[0][3]-test_row.tolist()[0][2]])
                                # If test_row is not yet in good_candidates, add test_row. (to prevent duplication & error message)
                            if not ((test_row in merged_set) or (test_row in used_set)):
                                merged_set.add(test_row)
                                # Good_candidates_list is a list version of good_candidates, which has tuples; the list version is necessary to loop the process
                                # list() used for tuple instead of array.tolist()
                                merged_set_list.append(list(test_row))
                                # also add test1 into used_set
                                used_set.add(test_row)
                            
                            # else, if test is inside temp_set row n data, calculate overlap percentage
                        else: 
                            # Overlap percentage of regions defined as (A1 intersect with A2)/(A1 union A2) * 100%
                            # Create 2x4 matrix called overlap set, which has test row 'u' and temp_set row 'n' data
                            overlap_set = []
                            overlap_set.append(test_row.tolist()[0])
                            overlap_set.append(premerged_set[n,:].tolist()[0])
                            overlap_set = np.asmatrix(overlap_set)
                            
                                # get overlap area
                            overlap_x1 = max(overlap_set[:,0])
                            overlap_x2 = min(overlap_set[:,1])
                            overlap_y1 = max(overlap_set[:,2])
                            overlap_y2 = min(overlap_set[:,3])
                            
                            overlap_w = overlap_x2 - overlap_x1
                            overlap_h = overlap_y2 - overlap_y1
                            
                            area_overlap = int(overlap_w * overlap_h)
                     
                                # get union area
                            outer_x1 = min(overlap_set[:,0])
                            outer_x2 = max(overlap_set[:,1])
                            outer_y1 = min(overlap_set[:,2])
                            outer_y2 = max(overlap_set[:,3])
                            
                            union_w = outer_x2 - outer_x1
                            union_h = outer_y2 - outer_y1
                            
                            gaps = 0
                                # get max possible area - gaps, since there are at most 4 possible squares of gaps from the largest square area created by the two overlapping bounding boxes
                                
                                # if x1 smaller than x1', and y1 greater than y1' (top left corner)
                            if (overlap_set[1,0] < overlap_set[0,0] and overlap_set[1,2] > overlap_set[0,2]) or (overlap_set[1,0] > overlap_set[0,0] and overlap_set[1,2] < overlap_set[0,2]):
                                gaps += int(abs((overlap_set[0,0] - overlap_set[1,0])*(overlap_set[1,2] - verlap_set[0,2])))
                                # if x2 greater than x2' and y1 greater than y1' (top right corner)
                            if (overlap_set[1,1] > overlap_set[0,1] and overlap_set[1,2] > overlap_set[0,2]) or (overlap_set[1,1] < overlap_set[0,1] and overlap_set[1,2] < overlap_set[0,2]):
                                gaps += int(abs((overlap_set[1,1] - overlap_set[0,1])*(overlap_set[1,2] - overlap_set[0,2])))
                                # if x1 greater than x1' and y2 greater than y2' (bottom left corner)
                            if (overlap_set[1,0] > overlap_set[0,0] and overlap_set[1,3] > overlap_set[0,3]) or (overlap_set[1,0] < overlap_set[0,0] and overlap_set[1,3] < overlap_set[0,3]):
                                gaps += int(abs((overlap_set[1,0] - overlap_set[0,0])*(overlap_set[1,3] - overlap_set[0,3])))
                                # if x2 greater than x2' and y2 less than y2' (bottom right corner)
                            if (overlap_set[1,1] > overlap_set[0,1] and overlap_set[1,3] < overlap_set[0,3]) or (overlap_set[1,1] < overlap_set[0,1] and overlap_set[1,3] > overlap_set[0,3]):
                                gaps += int(abs((overlap_set[1,1] - overlap_set[0,1])*(overlap_set[1,3] - overlap_set[0,3])))

                            area_union = int(union_w * union_h) - gaps
                            percent_overlap = area_overlap / area_union

                                # Merge bounding boxes if overlap percent is large enough
                            if percent_overlap > overlap_threshold:
                                
                                # Convert back to x,y,w,h and save as good_segment
                                good_segment = [int(outer_x1),int(outer_y1),int(outer_x2-outer_x1), int(outer_y2-outer_y1)]
                                good_segment = tuple(good_segment)
                                
                                # Remove check temp_set row 'n' if inside good_candidates since it is now merged
                                if tuple(premerged_set[n,:].tolist()) in merged_set:
                                    premerged_set.remove(tuple(premerged_set[n,:].tolist()))
                                    
                                # Also remove from list if needed
                                if premerged_set[n,:].tolist() in merged_set_list:
                                    merged_set_list.remove(premerged_set[n,:].tolist())
                                    
                                # Change test back into (x,y,w,h) as tuple called test_row to be put in good_candidates
                                test_row=tuple([test_row.tolist()[0][0],test_row.tolist()[0][2],test_row.tolist()[0][1]-test_row.tolist()[0][0],test_row.tolist()[0][3]-test_row.tolist()[0][2]])
                                    
                                # Remove test_row from good_candidates if needed since it is now merged
                                if tuple(test_row) in merged_set:
                                    merged_set.remove(test_row)
                                    
                                # Also remove from list version
                                if list(test_row) in merged_set_list:
                                    merged_set_list.remove(list(test_row))
                                    
                                # If merged bounding box not yet in good_candidates, add it
                                if not(good_segment in merged_set):
                                    merged_set.add(good_segment)
                                if not(list(good_segment) in merged_set_list):
                                    merged_set_list.append(list(good_segment))
                                    
                                    # also record used data
                                used_set.add(good_segment)
                                used_set.add(tuple(premerged_set[n,:].tolist()))
                                used_set.add(test_row)
                                skip=True
                            else:
                                if not ((test_row in merged_set)):
                                    merged_set.add(test_row)
                                    # merged_set_list is a list version of merged_set, which has tuples; the list version is necessary to loop the process
                                if not(test_row.tolist() in merged_set_list):
                                    merged_set_list.append(temp_set1[n,:].tolist())

                # rest temp_set to be what is inside good_candidates_list for looping process
            premerged_set = merged_set
            self.merged_set = merged_set

    def plot_bounding_boxes_with_name(self):
        """
        Create fig1, ax1, create single subplot, then draw bounding boxes x, y, w, h and save figure with name of model
        """
        # Draw rectangles on the original image
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(25, 25))
        ax.imshow(self.image)
        for x, y, w, h in self.merged_set: #or in candidates
            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)
        plt.show()

        fig.savefig('/home/chloong/Desktop/Justin San Juan/Testing Folder/Output/'+str(self.name)+'.jpg')

    def search(self, scale, sigma, min_size):
        """
        1. Separate unconnected segments.
        2. Apply minimum requirements.
        3. Use selective search to find bounding boxes.
        4. Merge highly overlapping bounding boxes.
        5. Print bounding boxes on image.

        Inputs:
        Problem image as numpy array [height,width]

        Returns:
        List of bounding boxes & extraction arrays
        """
        image = self.image
        # Label the matrix with different connected components
        labeled_matrix, num_cropped = measure.label(image, background=0, connectivity=1, return_num=True)

        # decleare premerged_set as an empty list

        
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

            # add clopped
            premerged_set.append(tuple(cropped_ls))
      
            # perform selective search
            img_lbl, regions = selectivesearch.selective_search(cropped, scale=scale, sigma=sigma, min_size=min_size)
            
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
                    
                    #add ls as list into premerged_set, which will be merged based on overlap
                self.premerged_set.append(ls)
        
        self.merge_set()
        self.print_bounding_boxes_with_name()
