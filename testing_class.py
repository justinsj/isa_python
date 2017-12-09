from component_classifier_predict import ComponentClassifierPredict
from component_classifier_training import ComponentClassifierTraining
from extraction_labelling import ExtractionLabelling
from helper_functions import calculate_accuracy
import numpy as np
import matplotlib.pyplot as plt

class TestingClass(object):

    def __init__(self, PATH):
        self.PATH = PATH
    
    def pick_random_training_samples(self, training_dataset_filename, n):
        '''
        Inputs: training_data, n
        Outputs: randomly selected list of n training images
        '''
        training_dataset = np.load(training_dataset_filename+'.npy')
        l = training_dataset.shape[0]
        random_indices = sample(range(l),n)
        random_data = training_dataset[random_indices]
        return random_data
        
    def load_all_gt(PATH):
        GT = []
        ground_truth_images = []
        imageset = np.load(PATH+all_training_images.npy)
        for i in range(400):
            image = imageset[i]
            
            ground_truth_filename = "all_" + str(i)
            labelling_obj=ExtractionLabelling(PATH,[],[],[],[],64, 100,100)
            labelling_obj.load_text(ground_truth_filename)
            GT_temp = labelling_obj.gt
            for j in range(len(GT_temp)):
                GT.append(GT_temp[j])
                x,y,w,h = GT_temp[j][0:-1]
                x1 = x
                x2 = x+w
                y1 = y
                y2 = y+h
                ground_truth_images.append(image[y1:y2,x1:x2])
        GT = np.asmatrix(GT)
        ground_truth_data = GT[:,0:-1]
        ground_truth_indices = GT[:,-1]
        return ground_truth_images, ground_truth_data, ground_truth_indices
    def predict_all_from_gt_data(ground_truth_images, trained_model):
        prediction_obj = ComponentClassifierPredict(0.7,0.3)
        ext_class_index, ext_class_name, \
        ext_match_first_max_percent, ext_match_second_max_percent = prediction_obj.predict_classes(ground_truth_images,trained_model)
        return ext_class_index
    def test_classifier(self, image, training_dataset_filename, train_ratio, n): 
        #training_dataset_filename example: Training_Samples_64_classes_100x100_all
        # ground_truth_filename example: all_44
        '''
        Inputs: image, ground_truth_data, ground_truth_index
        Outputs: accuracy of model classifier only (using same segmentations of ground truth)
        '''
        # load ground truth data
        ground_truth_images, ground_truth_data, ground_truth_indices = self.load_all_gt(PATH)

        # pick random training samples
        random_data = self.pick_random_training_samples(training_dataset_filename, n)
        # train model
        training_obj = ComponentClassifierTraining(self.PATH, "Training_Samples_64_classes_100x100_all", 64, 0, train_ratio, 1-train_ratio)
        training_obj.train(100)
        trained_model = training_obj.model
        # test on all samples
        prediction_indices = self.predict_all_from_gt_data(ground_truth_images, trained_model)
        # calculate accuracy & return string
        accuracy = calculate_accuracy(prediction_indices, ground_truth_indices)
        return accuracy
    def test_classifier_multiple_n(self, image, training_dataset_filename, train_ratio, k, list_n):
        #get accuracies for k tries of n, and store in 2xlen(ns) matrix
        x=[]
        y=[]
        for n in list_of_n:
            for i in range(k):
                accuracy = self.test_classifier(image, training_dataset_filename, train_ratio, n)
                x.append(n)
                y.append(accuracy)
        #plot and save graph
        x = np.asarray(x)
        y = np.asarray(y)
        plt.scatter(x,y)
        plt.xlabel("Training Samples Size")
        plt.ylabel("Accuracy")
        plt.show()
        
        plt.savefig("Accuracy Scatterplot for "+str(k)+" samples")
        
    def test_classifier_and_segmentation(self, image, ext_images, ext_data, ext_class_index, ext_class_names, ground_truth_data, ground_truth_index):
        '''
        Input: image, ext_images, ext_data, ext_class_index, ext_class_names
        Output: Accuracy calculation
        '''
        # Check overlap scores
        #overlap is calculated by (A & B) / (A or B), where A is the ground truth area, and B is the extracted image area
        
        # If not enough overlap, correct answer is 23, if enough overlap, take correct class from ground_truth_index
        #output: list of predicted classes vs correct classes
        
        
        #calculate accuracy