import numpy as np
from skimage import measure
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class ComponentSegmentation(object):
    
    def __init__(self, image, name, 
                 min_shape, min_height, min_width, 
                 buffer_zone, min_area, min_black, min_black_ratio,
                 overlap_repeats, overlap_threshold):
        
        self.name = name
        self.image = image
        self.temp_set = []
        self.premerged_set = []
        self.merged_set = []
        self.used_set = set()
        self.min_shape = min_shape 
        self.min_height = min_height
        self.min_width = min_width
        self.buffer_zone = buffer_zone
        self.min_area = min_area
        self.min_black = min_black
        self.min_black_ratio = min_black_ratio
        self.overlap_repeats = overlap_repeats
        self.overlap_threshold = overlap_threshold

        

    def swap_columns(self):
        """
        Swap columns from [x,y,w,h] to [x1, x2, y1, y2]
        Input: [x, y, w, h] array
        Output: [x1, x2, y1, y2] array
        """

        premerged_set1 = np.asmatrix(np.asarray(self.premerged_set))
        
        x_col = premerged_set1[:,0]
        y_col = premerged_set1[:,1]
        w_col = premerged_set1[:,2]
        h_col = premerged_set1[:,3]
        
        x1_col = np.asarray(x_col)
        x2_col = np.asarray(x1_col + w_col)
        y1_col = np.asarray(y_col)
        y2_col = np.asarray(y_col + h_col)
        
        premerged_set = np.column_stack((x1_col, x2_col, y1_col, y2_col))

        return premerged_set

    def merge_set(self):
        """
        1. Change [x, y, w, h] to [x1, x2, y1, y2] using self.swap_columns()
        2. Compare each row to each other row and merge if overlap is over threshold
        3. Loop merging algorithm until convergence

        Inputs: [x, y, w, h] array of unmerged bounding box coordinates
        Outputs: [x, y, w, h] array of merged bounding box coordinates
        """

        # Declare empty set for merging
        merged_set = set()
        # List is necessary for looping
        merged_set_list=[]
        
        # loop merging algorithm
        for loop_count in range(self.overlap_repeats + 1):
            # Declare empty set for used rows
            used_set = set()
            
            # modify matrix from [x,y,w,h] into [x1,x2,y1,y2] matrix using swap_columns
            premerged_set = self.swap_columns()
            
            # set skip variable to false. When true, it will stop using row u to check overlap with other rows.
            skip = False
            
            # cycle through row u of [x1,x2,y1,y2] matrix
            for u in range(premerged_set.shape[0]):
                
                # Set test_row as [x1,x2,y1,y2] data in row u
                
                test_row=premerged_set[u,:]

                
                # Cycle through each row n of [x1,x2,y1,y2] matrix and
                # Check if row u overlaps with other rows n in matrix
                for n in range(premerged_set.shape[0]):
                    # When skip == True, it means row u has been merged with another row, so 'continue' to stop checking row u with other rows.
                    if skip==True:
                        skip = False
                        continue
                    
                    # Skip checking of row n with row u if u ==n. If checking different row, check for overlap
                    if u != n:
                        #row u as (x, y, w, h)
                        test_row1=tuple([test_row.tolist()[0],test_row.tolist()[2],test_row.tolist()[1]-test_row.tolist()[0],test_row.tolist()[3]-test_row.tolist()[2]])
                        #row n as (x, y, w, h)
                        test_row2=tuple([premerged_set[n,:].tolist()[0],premerged_set[n,:].tolist()[2],premerged_set[n,:][1]-premerged_set[n,:][0],premerged_set[n,:][3]-premerged_set[n,:][2]])
                        
                        # Check if region n is outside of region u
                        # xi' is the xi of the fixed test [x1,x2,y1,y2]
                        # If x1 is greater than x2', or x2 is less than x1' ,then same with y direction
                        # If this statement is passed, it means u does not overlap with n
                        if premerged_set[n,0] > test_row[1] or premerged_set[n,1] < test_row[0] or premerged_set[n,2] > test_row[3] or premerged_set[n,3] < test_row[2]:
                            
                            #if not yet used to merge, add to merged set first
                            if not (test_row1 in used_set):
                                # If test_row is not yet in good_candidates, add test_row. (to prevent duplication & error message)
                                if not (test_row1 in merged_set):
                                    merged_set.add(test_row1)
                                # Good_candidates_list is a list version of good_candidates, which has tuples; the list version is necessary to loop the process
                                if not (list(test_row1) in merged_set_list):
                                    # list() used for tuple instead of array.tolist()
                                    merged_set_list.append(list(test_row1))

                        # else, calculate overlap percentage
                        else: 
                            # Overlap percentage of regions defined as (A1 intersect with A2)/(A1 union A2) * 100%
                            # Create 2x4 matrix called overlap set, which has test row 'u' and temp_set row 'n' data
                            overlap_set = []
                            overlap_set.append(test_row.tolist())
                            overlap_set.append(premerged_set[n,:].tolist())
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
                                gaps += int(abs((overlap_set[0,0] - overlap_set[1,0])*(overlap_set[1,2] - overlap_set[0,2])))
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
                            if percent_overlap > self.overlap_threshold:

                                # Convert back to x,y,w,h and save as good_segment
                                good_segment = [int(outer_x1),int(outer_y1),int(outer_x2-outer_x1), int(outer_y2-outer_y1)]
                                good_segment = tuple(good_segment)
                                
                                # Remove test_row row 'n' if inside merged_set since it is now merged
                                if test_row2 in merged_set:
                                    merged_set.remove(test_row2)
                                    
                                # Also remove from list if needed
                                if list(test_row2) in merged_set_list:
                                    merged_set_list.remove(list(test_row2))
                                    
                                # Remove test_row1 from good_candidates if needed since it is now merged
                                if test_row1 in merged_set:
                                    merged_set.remove(test_row1)
                                    
                                # Also remove from list version
                                if list(test_row1) in merged_set_list:
                                    merged_set_list.remove(list(test_row1))
                                    
                                # If merged bounding box not yet in good_candidates, add it
                                if not(good_segment in merged_set):
                                    merged_set.add(good_segment)
                                    
                                if not(list(good_segment) in merged_set_list):
                                    merged_set_list.append(list(good_segment))
                                
                                if not((test_row1) in used_set):
                                    used_set.add(test_row1)
                                if not((test_row2) in  used_set):
                                    used_set.add(test_row2)
                                if not((good_segment) in used_set):
                                    used_set.add(good_segment)
                                
                                skip=True
                            else:
                                
                                #only remove from data if not yet used in used set
                                if not(test_row1 in used_set):
                                    #if not yet in merged_set, add it
                                    if not (test_row1 in merged_set):
                                        merged_set.add(test_row1)
                                    # merged_set_list is a list version of merged_set, which has tuples; the list version is necessary to loop the process
                                    if not(list(test_row1) in merged_set_list):
                                        merged_set_list.append(list(test_row1))
                                        
                                if not(test_row2 in used_set):
                                    # if row n is not yet in merged_set, add it
                                    if not(test_row2 in merged_set):
                                        merged_set.add(test_row2)
                                    #list version of above
                                    if not(list(test_row2) in merged_set_list):
                                        merged_set_list.append(list(test_row2))
                                

                # reset premerged_set to be what is inside merged_set for looping process
                self.premerged_set = list(merged_set)
            self.merged_set = list(merged_set)

    def separate_unconnected_segments(self, scale_input, sigma_input, min_size_input):
        """
        a. Separate unconnected segments using measure.label.
        b. Ignore segment if it does not pass minimum requirements, record segment if it passes, as [x,y,w,h]
        c. Perform selective search on unconnected segment if it passes.

        Input: Image
        Output: List of separated segmentations coordinates 
        """


        image = self.image
        # Label the matrix with different connected components
        labeled_matrix, num_cropped = measure.label(image, background=0, connectivity=1, return_num=True)

        # decleare premerged_set as an empty list
        self.premerged_set = []
        
        # Loop through all connected components
        # range list is different in python. numbers from 1 to 5 is range(1, 5+1), (last number is not included)
        for i in range(1, num_cropped + 1):
        
            # Get the coordinates of current labels
            x = np.array(np.where(labeled_matrix == i))

            # Eliminate case of noise, tuneable
            # 'continue' skips everything under the if statement in the for loop
            if x.shape[1] < self.min_shape: continue
           
            # We have down > up and right > left # To find corners of image
            up = x[0][0]
            down = x[0][-1]
            left = np.amin(x[1])
            right = np.amax(x[1])

            # Essential if there is noise, because it will be counted as one conencted component
            if down - up < self.min_height or right - left < self.min_width: continue

            # Buffering zone: 2 (To exapnd image), tuneable
            # Crop the image of current connected component with buffer
            cropped = image[up-self.buffer_zone:down+self.buffer_zone, left-self.buffer_zone:right+self.buffer_zone]

            # Convert to RGB --> selective search requires RGB
            temp = np.zeros([cropped.shape[0], cropped.shape[1], 3])
            temp[:, :, 0] = cropped
            temp[:, :, 1] = cropped
            temp[:, :, 2] = cropped
            cropped = temp

            # perform selective search on cropped region
            self.selective_search(cropped,left,up, scale_input, sigma_input, min_size_input)

    def selective_search(self,cropped,left,up, scale_input, sigma_input, min_size_input):
        """
        1. Perform selective search on cropped region
        2. Apply minimum requirements
        3. Record bounding box coordinates as [x, y, w, h] in self.premerged_set

        Inputs:
        Cropped region as image matrix (RGB)

        Returns:
        Bounding boxes in self.premerged_set
        """


        # perform selective search
        img_lbl, regions = selectivesearch.selective_search(cropped, scale=scale_input, sigma=sigma_input, min_size=min_size_input)
        
        # each r in regions is a dictionary (rect: x, y, w, h; size: n ...)
        for r in regions:

            # exclude regions smaller than min_area pixels, tuneable
            if r['size'] < self.min_area:
                continue
            
            
            # change relative coordinates to absolute coordinates
            x, y, w, h = r['rect']
            x1 = x + left
            x2 = x + w + left - 1
            y1 = y + up
            y2 = y + h + up - 1

            cropped_region = self.image[y1:y2, x1:x2] 

            # get number of pixels in each connected component and store in black_spot
            black_spot = np.array(np.where(cropped_region == 1))
            
            # filter those with very few black dots (noise)
            if black_spot.shape[1] < self.min_black: continue
            if float(black_spot.shape[1]) / (w * h) < self.min_black_ratio: continue
            

            ls = list(r['rect'])
            ls[0] = ls[0] + left
            ls[1] = ls[1] + up
            
                #add ls as list into self.premerged_set as tuple of (x,y,w,h) , which will be merged based on overlap
            self.premerged_set.append(tuple(ls))
            

    def search(self, scale_input, sigma_input, min_size_input):
        """
        1.  a) Separate unconnected segments
            b) Apply minimum requirements
            c) Use selective search to find bounding boxes.
        2. Merge highly overlapping bounding boxes.
        3. Print bounding boxes on image.

        Inputs:
        Problem image as numpy array [height,width]

        Returns:
        List of bounding boxes & extraction arrays self.merged_set
        """
        self.separate_unconnected_segments(scale_input, sigma_input, min_size_input) 
        self.merge_set()
