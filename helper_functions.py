from adjustText import adjust_text
import matplotlib.patheffects as PathEffects
from scipy import interpolate
from keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from constants import subset_dictionary
from constants import specific_dictionary

def print_time_string(index,time_cost_string_list,time_cost_time_list):
    print(str(time_cost_string_list[index]) + ' done... Time Elapsed : '+ str(time_cost_time_list[index]) + ' seconds...')
    return
def store_time(index,time_cost_time_list,time_count):
    time_cost_time_list[index] = time_count
    return time_cost_time_list
def print_image_bw(image,l,w):
    fig,ax=plt.subplots(ncols=1,nrows=1,figsize = (l,w))
    ax.imshow(image,cmap = 'binary')
    plt.show()


def dropout_search(dropout_ls):
    """
    Hyperparameter search for dropout
    """
    pass

def plot_model_results_and_save(image,name, ext_data_list, ext_class_index_list, ext_class_name_list, ground_truth_index_list):
    fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(25, 25))
    together = []
    for i in range(len(ext_data_list)):
        x, y, w, h = ext_data_list[i]
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax1.add_patch(rect)
        color='b' #not compared to any answer
        try:
            if ext_class_index_list[i] != ground_truth_index_list[i]:
                color='r' # incorrect
            else:
                color = 'g' # correct
                
        except:
            color = 'b' #not compared to any answer
#        ax1.annotate(str(i) + ' : ' + str(ext_class_name_list[i]),xy=(x, y-2),fontsize=12,color=color)
        string = str(i+1) + ' : ' + str(ext_class_name_list[i])
        
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

def calculate_accuracy(prediction_list, ground_truth_list):
    if len(prediction_list) != len(ground_truth_list):
        return 'invalid'
    
    ground_truth_list = np.array(ground_truth_list)
    prediction_list = np.array(prediction_list)
    
    score_matrix = (ground_truth_list == prediction_list)

    ##### Calculate Overall Accuracy #####
    if len(score_matrix) == 0:
        accuracy = 'N/A'
    else:
        accuracy= (sum(score_matrix)/len(score_matrix))*100

    print('Accuracy is '+str(accuracy)+" %")

    return accuracy

def create_string_from_dict(accuracies_dict):
    string = ''
    for subset in accuracies_dict:
        string = string + subset +' : '+ str(accuracies_dict[subset]) + '\n'
            
    return string


def calculate_subset_accuracies(prediction_list, ground_truth_list):
    
    #test
    
    #initialize empty lists for each subset for predictions
    subset_predictions_list_dict = {}
    for subset in subset_dictionary:
        extra = {subset:[]}
        subset_predictions_list_dict.update(extra)
            
    print(subset_predictions_list_dict)
    print(len(subset_predictions_list_dict))
    #initialize empty lists for each subset for ground truths
    subset_ground_truths_list_dict = subset_predictions_list_dict.copy()
    
    
    #parse through each prediciton and ground_truth and save them into lists      
    for index in range(len(prediction_list)):
        prediction_index = prediction_list[index]
        ground_truth_index = ground_truth_list[index]
        
        for subset in subset_dictionary:
            if ground_truth_index in subset_dictionary[subset]:
                subset_ground_truths_list_dict[subset].append(ground_truth_index)
                subset_predictions_list_dict[subset].append(prediction_index)
                        
    #for each subset, calculate accuracy and add to subset_accuracies_dict
    subset_accuracies_dict = {}
    for subset in subset_dictionary:
        subset_prediction_list = subset_predictions_list_dict[subset]
        subset_ground_truth_list = subset_ground_truths_list_dict[subset]
        subset_accuracy = (calculate_accuracy(subset_prediction_list,subset_ground_truth_list))
        
        extra = {subset:subset_accuracy}
        
        subset_accuracies_dict.update(extra)
        
    print(subset_accuracies_dict)
        
    return subset_accuracies_dict, subset_predictions_list_dict, subset_ground_truths_list_dict

def calculate_specific_accuracies(prediction_list, ground_truth_list):
    
    
    #initialize empty lists for each subset for predictions
    specific_predictions_list_dict = {}
    for specific in specific_dictionary:
        extra = {specific:[]}
        specific_predictions_list_dict.update(extra)
            
    print(specific_predictions_list_dict)
    print(len(specific_predictions_list_dict))
    #initialize empty lists for each specific for ground truths
    specific_ground_truths_list_dict = specific_predictions_list_dict.copy()
    
    
    #parse through each prediciton and ground_truth and save them into lists      
    for index in range(len(prediction_list)):
        prediction_index = prediction_list[index]
        ground_truth_index = ground_truth_list[index]
        
        for specific in specific_dictionary:
            if ground_truth_index in specific_dictionary[specific]:
                specific_ground_truths_list_dict[specific].append(ground_truth_index)
                specific_predictions_list_dict[specific].append(prediction_index)
                    
    
    #for each specific, calculate accuracy and add to specific_accuracies_dict
    specific_accuracies_dict = {}
    for specific in specific_dictionary:
        specific_prediction_list = specific_predictions_list_dict[specific]
        specific_ground_truth_list = specific_ground_truths_list_dict[specific]
        specific_accuracy = (calculate_accuracy(specific_prediction_list,specific_ground_truth_list))
        
        extra = {specific:specific_accuracy}
        
        specific_accuracies_dict.update(extra)
        
    print(specific_accuracies_dict)
        
    return specific_accuracies_dict, specific_predictions_list_dict, specific_ground_truths_list_dict

def calculate_accuracies(prediction_list, ground_truth_list):
    '''
    Inputs: ground_truth_list, prediciton_list
    Returns: overall_accuracy, subset_accuracies, specific_accuracies, string
    '''
    overall_accuracy = calculate_accuracy(prediction_list, ground_truth_list)
    
    subset_accuracies_dict, subset_prediciton_list_dict, subset_ground_truth_list_dict = calculate_subset_accuracies(prediction_list, ground_truth_list)
    subset_string = create_string_from_dict(subset_accuracies_dict)
    
    specific_accuracies_dict, specific_prediction_list_dict, specific_ground_truth_list_dict = calculate_specific_accuracies(prediction_list, ground_truth_list)
    specific_string = create_string_from_dict(specific_accuracies_dict)
    string = subset_string + '\n' + '\n' + specific_string
    
    return overall_accuracy, subset_accuracies_dict, specific_accuracies_dict, string


# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, PATH='', name='confusion_matrix', verbose = False,save = False):
    
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(25, 25))
    
    import itertools
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if verbose:
            print("Normalized confusion matrix")
    else:
        if verbose:
            print('Confusion matrix, without normalization')
    
#    print(cm)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    if save:
        fig.savefig(PATH+name)
        print('Figure is saved as: ' +PATH+name+'.png')

def confusion_matrix_analysis(cm, dataset_PATH, name, min_count, verbose = False):
    count = 0
    string_counts_list = []
    string_entries_list = [] #list of entries in dictionary
    
    f = open(dataset_PATH+name+'.txt','w+')
    from constants import target_names_all
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            string = ''
            if i == j: continue
            count = cm[i][j]
            if count == 0: continue
            if count < min_count: continue
            true_label = target_names_all[i]
            predicted_label = target_names_all[j]
            string = str(true_label) + ' --> ' + str(predicted_label)
            
            string_entries_list.append(string)
            string_counts_list.append(count)
    
    string_list = [] #list of strings to be added
    for k in range(len(string_counts_list)):
        if verbose:
            print(k)
        #get max count
        max_count = max(string_counts_list)
        #get index of max_count
        max_index = string_counts_list.index(max_count)
        #create string
        string_line = '['+str(string_counts_list[max_index])+'] '+ str(string_entries_list[max_index])
        #add string to string_list
        string_list.append(string_line)
        #remove max entries
        string_counts_list.remove(string_counts_list[max_index])
        string_entries_list.remove(string_entries_list[max_index])
        
    #write each string
    for line in string_list:
        f.writelines(line + '\n')
    f.close()
    if verbose:
        return string_list
    print('Data saved as: ' + dataset_PATH+name+'.txt')
    return

def separate_correct(list_to_be_separated, prediction_list, ground_truth_list):
    ground_truth_array = np.asarray(ground_truth_list)
    prediction_array = np.asarray(prediction_list)
    correct_list = (ground_truth_array == prediction_array)
    correct_output_list = []
    wrong_output_list = []
    for i in range(len(correct_list)):
        if correct_list[i] == True:
            correct_output_list.append(list_to_be_separated[i])
        else:
            wrong_output_list.append(list_to_be_separated[i])
    return correct_output_list, wrong_output_list
def print_all_entropy_parameters(prediction_list,
                                    ground_truth_list,
                                    ext_class_first_max_index,
                                    ext_class_second_max_index,
                                    ext_match_first_max_percent,
                                    ext_match_second_max_percent,
                                    resolution, min_percent_match_start,
                                    min_percent_match_end,
                                    min_confidence_start,
                                    min_confidence_end):
    import numpy as np
    min_percent_match_list = []
    min_confidence_list = []
    accuracy_list = []
    for min_percent_match in np.arange(min_percent_match_start,min_percent_match_end+resolution,resolution):
        temp_min_percent_match_list = []
        temp_min_confidence_list = []
        temp_accuracy_list = []
        for min_confidence in np.arange(min_confidence_start,min_confidence_end+resolution,resolution):
            accuracy = test_accuracy(prediction_list,ground_truth_list,
                    ext_class_first_max_index,ext_class_second_max_index,
                    ext_match_first_max_percent,ext_match_second_max_percent,
                    min_percent_match, min_confidence)
            temp_accuracy_list.append(accuracy*100)
            temp_min_confidence_list.append(min_confidence)
            temp_min_percent_match_list.append(min_percent_match)
            print(str(min_percent_match) + ' ' + str(min_confidence) + ' ' +str(accuracy))
        
        min_percent_match_list.append(np.asarray(temp_min_percent_match_list))
        min_confidence_list.append(np.asarray(temp_min_confidence_list))
        accuracy_list.append(np.asarray(temp_accuracy_list))
    min_percent_match_list = np.asarray(min_percent_match_list)
    min_confidence_list = np.asarray(min_confidence_list)
    accuracy_list = np.asarray(accuracy_list)
    return min_percent_match_list, min_confidence_list, accuracy_list

def get_gradient_entropy_parameters(prediction_list,
                                    ground_truth_list,
                                    ext_class_first_max_index_1,
                                    ext_class_second_max_index_1,
                                    ext_match_first_max_percent_1,
                                    ext_match_second_max_percent_1,
                                    min_percent_match, min_confidence, resolution):
    
    
    accuracy_base = test_accuracy(prediction_list,ground_truth_list,
                    ext_class_first_max_index_1,ext_class_second_max_index_1,
                    ext_match_first_max_percent_1,ext_match_second_max_percent_1,
                    min_percent_match, min_confidence)
    
    #try increasing min_percent_match
    inc_min_percent_match = min_percent_match
    accuracy_inc_min_percent_match = accuracy_base
    while accuracy_inc_min_percent_match == accuracy_base and inc_min_percent_match <= 1.0 and inc_min_percent_match - min_percent_match <0.5:
        print('inc_min_percent_match = ' + str(inc_min_percent_match))
        inc_min_percent_match +=resolution
        accuracy_inc_min_percent_match = test_accuracy(prediction_list,ground_truth_list,
                        ext_class_first_max_index_1,ext_class_second_max_index_1,
                        ext_match_first_max_percent_1,ext_match_second_max_percent_1,
                        inc_min_percent_match, min_confidence)
        
    if inc_min_percent_match >=1.0: inc_min_percent_match = min_percent_match
    
    #try decreasing min_percent_match
    dec_min_percent_match = min_percent_match
    accuracy_dec_min_percent_match = accuracy_base
    while accuracy_dec_min_percent_match == accuracy_base and dec_min_percent_match >= 0 and min_percent_match - dec_min_percent_match <0.5:
        print('dec_min_percent_match = ' + str(dec_min_percent_match))
        dec_min_percent_match -=resolution
        accuracy_dec_min_percent_match = test_accuracy(prediction_list,ground_truth_list,
                        ext_class_first_max_index_1,ext_class_second_max_index_1,
                        ext_match_first_max_percent_1,ext_match_second_max_percent_1,
                        dec_min_percent_match, min_confidence)
    if dec_min_percent_match <=0: dec_min_percent_match = min_percent_match
    
    if accuracy_inc_min_percent_match > accuracy_base and accuracy_inc_min_percent_match > accuracy_dec_min_percent_match:
        dmin_percent_match = inc_min_percent_match-min_percent_match
    elif accuracy_dec_min_percent_match > accuracy_base:
        dmin_percent_match = dec_min_percent_match-min_percent_match
    else: dmin_percent_match = 0
    
    #try increasing min_confidence
    inc_min_confidence = min_confidence
    accuracy_inc_min_confidence = accuracy_base
    while accuracy_inc_min_confidence == accuracy_base and inc_min_confidence <= 1.0 and inc_min_confidence - min_confidence <0.5:
        print('inc_min_confidence = ' + str(inc_min_confidence))
        inc_min_confidence +=resolution
        accuracy_inc_min_confidence = test_accuracy(prediction_list,ground_truth_list,
                        ext_class_first_max_index_1,ext_class_second_max_index_1,
                        ext_match_first_max_percent_1,ext_match_second_max_percent_1,
                        min_percent_match, inc_min_confidence)
        
    if inc_min_confidence >=1.0: inc_min_confidence = min_confidence
    #try decreasing min_confidence
    dec_min_confidence = min_confidence
    accuracy_dec_min_confidence = accuracy_base
    while accuracy_dec_min_confidence == accuracy_base and dec_min_confidence >= 0 and min_confidence - dec_min_confidence <0.5:
        print('dec_min_confidence= ' + str(dec_min_confidence))
        dec_min_confidence -=resolution
        accuracy_dec_min_confidence = test_accuracy(prediction_list,ground_truth_list,
                        ext_class_first_max_index_1,ext_class_second_max_index_1,
                        ext_match_first_max_percent_1,ext_match_second_max_percent_1,
                        min_percent_match, dec_min_confidence)
    if dec_min_confidence <=0: dec_min_confidence = min_confidence
    
    if accuracy_inc_min_confidence > accuracy_base and accuracy_inc_min_confidence > accuracy_dec_min_confidence:
        dmin_confidence = inc_min_confidence-min_confidence
    elif accuracy_dec_min_confidence > accuracy_base:
        dmin_confidence = dec_min_confidence-min_confidence
    else: dmin_confidence = 0
    
    return dmin_percent_match, dmin_confidence



def test_accuracy(prediction_list,
                    ground_truth_list,
                    ext_class_first_max_index_1,
                    ext_class_second_max_index_1,
                    ext_match_first_max_percent_1,
                    ext_match_second_max_percent_1,
                    min_percent_match, min_confidence):
    temp_prediction_list = []
    for i in range(len(prediction_list)):
        index = ext_class_first_max_index_1[i]
        if not(ext_match_first_max_percent_1[i] > min_percent_match and (ext_match_first_max_percent_1[i]-ext_match_second_max_percent_1[i]) > min_confidence):
            index = 23
        
        temp_prediction_list.append(index)
        
    score_matrix = (np.asarray(temp_prediction_list) == np.asarray(ground_truth_list))
    accuracy = sum(score_matrix)/len(score_matrix)
    return accuracy
def get_optimal_entropy_parameters(prediction_list,
                                    ground_truth_list,
                                    ext_class_first_max_index_1,
                                    ext_class_second_max_index_1,
                                    ext_match_first_max_percent_1,
                                    ext_match_second_max_percent_1,iters,seed, resolution):
    import random
    from helper_functions import get_gradient_entropy_parameters
    from helper_functions import test_accuracy
    #initialize with seed
    min_percent_match = int(random.random()/resolution)*resolution #from 0 to 1
    min_confidence = int(random.random()/resolution)*resolution # from 0 to 1
    for a in range(iters):
        print(a)
        #get gradient
        dmin_percent_match, dmin_confidence = get_gradient_entropy_parameters(prediction_list,
                                    ground_truth_list,
                                    ext_class_first_max_index_1,
                                    ext_class_second_max_index_1,
                                    ext_match_first_max_percent_1,
                                    ext_match_second_max_percent_1, min_percent_match, min_confidence, resolution)
        
        #apply gradient update
        min_percent_match += dmin_percent_match
        min_confidence += dmin_confidence
        
        #test accuracy
        accuracy = test_accuracy(prediction_list,
                        ground_truth_list,
                        ext_class_first_max_index_1,
                        ext_class_second_max_index_1,
                        ext_match_first_max_percent_1,
                        ext_match_second_max_percent_1,
                        min_percent_match, min_confidence)
        print(accuracy)
    return min_percent_match, min_confidence
#%%
################## OLD CODE #########################
"""
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
"""
