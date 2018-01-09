import collections
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import callbacks
from keras.optimizers import Adadelta
from random import sample
import gc
import random
gc.enable()

class ModelNotTrained(Exception):
    """
    When certain class attributes are accessed but but requires the model to be
    trained first
    """
    pass


class ComponentClassifierTraining(object):
    """
    Train component classifier, current model is adopted from sketched-a-net paper [1],
    which achieved 98% in classifying 63 classes with 50 epochs and zero dropout
    (1 epoch takes around 15 seconds on Nvidia GeForce GTX 1060)

    [1]: https://arxiv.org/pdf/1501.07873.pdf
    """

    def __init__(self,PATH,name, num_classes, dropout, TRAINING_RATIO_TRAIN, TRAINING_RATIO_VAL):
        """
        Input:
        - training_data is in shape(num_train, 100*100+1), where the last values in
          second dimension are the labels
        - val_data is in shape(num_val, 100*100+1), where the last values in
          second dimension are the labels 
        - test_data is in shape(num_test, 100*100+1), where the last values in
          second dimension are the labels 
        """
        self.is_trained = False  # To initialize that the model is not trained yet
        self.TRAINING_RATIO_TRAIN = TRAINING_RATIO_TRAIN
        self.TRAINING_RATIO_VAL = TRAINING_RATIO_VAL
        self.num_classes = num_classes
        self.batch_size = 25
        self.img_rows,self.img_cols = 100,100

    def load_data(self,PATH,name):
        
        """ Load data set from path and filename """
        data_set = np.load(PATH+name+'.npy')
        return data_set
    
    def shuffle_data(self, data_set,seed):
        """
        Split data set randomly
        Further optimization: Convert it into numpy
        """
        random.seed(seed)
        l = data_set.shape[0]
        f = int(l * self.TRAINING_RATIO_TRAIN)
        train_indices = sample(range(l),f)
        val_and_test_indices = np.delete(np.array(range(0, l)), train_indices)
        train_data = np.copy(data_set[train_indices])
        val_and_test_data = np.copy(data_set[val_and_test_indices])
        data_set = None #clear data_set
        gc.collect()
        
        l = val_and_test_data.shape[0]
        f = int(l * (self.TRAINING_RATIO_VAL/(1-self.TRAINING_RATIO_TRAIN)))
        val_indices = sample(range(l),f)
        test_indices = np.delete(np.array(range(0, l)), val_indices)
        val_data = np.copy(val_and_test_data[val_indices])
        test_data = np.copy(val_and_test_data[test_indices])
        val_and_test_data = None
        gc.collect()
        
#        train_data = np.random.shuffle(np.asarray(train_data))
#        val_data = np.random.shuffle(np.asarray(val_data))
#        test_data = np.random.shuffle(np.asarray(test_data))
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_shuffled_data(train_data, val_data, test_data)
    
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def load_shuffled_data(self, train_data, val_data, test_data):
        """ Load all data to be directly passed into the model """

        # Convert input data to 2D float data
        # The last values in second dimension are the labels
        
        X_train, y_train = train_data[:, 0:-1].astype('float32'), train_data[:, -1]
        X_val, y_val = val_data[:, 0:-1].astype('float32'), val_data[:, -1]
        X_test, y_test = test_data[:, 0:-1].astype('float32'), test_data[:, -1]

        # Reshape back to 3D matrix to be passed into CNN
        X_train = X_train.reshape(X_train.shape[0], self.img_rows, self.img_cols)
        X_val = X_val.reshape(X_val.shape[0], self.img_rows, self.img_cols)
        X_test = X_test.reshape(X_test.shape[0], self.img_rows, self.img_cols)

        # Necessary transformation
        if K.image_data_format() == 'channels_first':
            X_train = X_train.reshape(X_train.shape[0], 1, self.img_rows, self.img_cols)
            X_val = X_val.reshape(X_val.shape[0], 1, self.img_rows, self.img_cols)
            X_test = X_test.reshape(X_test.shape[0], 1, self.img_rows, self.img_cols)
        else:
            X_train = X_train.reshape(X_train.shape[0], self.img_rows, self.img_cols, 1)
            X_val = X_val.reshape(X_val.shape[0], self.img_rows, self.img_cols, 1)
            X_test = X_test.reshape(X_test.shape[0], self.img_rows, self.img_cols, 1)

        # Convert class vectors to one-hot matrices
        y_train = to_categorical(y_train, self.num_classes)
        y_val = to_categorical(y_val, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)

        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = X_train, y_train, X_val, y_val, X_test, y_test
        return X_train, y_train, X_val, y_val, X_test, y_test

    @staticmethod
    def load_sketch_a_net_model(dropout, num_classes, input_shape):
        """ Load Sketch-A-Net keras model layers """
        model = Sequential()

        # Layer 1
        model.add(Conv2D(64, (15, 15), strides=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=1))
        model.add(Dropout(dropout))
        # Layer 2
        model.add(Conv2D(128, (5, 5), strides=1, activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
        model.add(Dropout(dropout))
        # Layer 3
        model.add(Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(Dropout(dropout))
        # Layer 4
        model.add(Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(Dropout(dropout))
        # Layer 5
        model.add(Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
        model.add(Dropout(dropout))
        # Layer 6
        model.add(Conv2D(512, (5, 5), strides=1, activation='relu'))
        model.add(Dropout(0.5))
        # Layer 7
        model.add(Conv2D(512, (1,1), strides=1, activation='relu'))
        model.add(Dropout(0.5))
        # Layer 8
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))
        
        
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adadelta(),
                      metrics=['accuracy'])
        model.summary()
        return model

    def train(self, epochs, seed, verbose=1):
        """ Train model with input number of epochs """
        np.random.seed(seed)  # For reproducibility
        
        self.history = self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size,
                                      epochs=epochs, verbose=verbose,
                                      validation_data=(self.X_val, self.y_val))
        self.is_trained = True
        
    def train_with_callbacks(self, epochs, seed, verbose=1):
        """ Train model with input number of epochs """
        np.random.seed(seed)  # For reproducibility


        filename='model_train_new.csv'
        
        csv_log=callbacks.CSVLogger(filename, separator=',', append=False)
        
        early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')
        
        filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"
        
        checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        
        callbacks_list = [csv_log,early_stopping,checkpoint]
        
        self.history = self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size,
                                      epochs=epochs, verbose=verbose,
                                      validation_data=(self.X_val, self.y_val), callbacks=callbacks_list)
        self.is_trained = True

    def get_train_stats(self):
        """ Get training loss and accuracy from history """
        train_loss = self.history.history['loss'][-1]
        train_accuracy = self.history.history['acc'][-1]

        return train_loss, train_accuracy

    def get_validation_stats(self):
        """ Get validation loss and accuracy from history """
        validation_loss = self.history.history['val_loss'][-1]
        validation_accuracy = self.history.history['val_acc'][-1]

        return validation_loss, validation_accuracy

    def get_test_stats(self):
        """ Use model to evaluate test loss and accuracy """
        [test_loss, test_accuracy] = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        return test_loss, test_accuracy

    def get_stats(self):
        if not self.is_trained:
            raise ModelNotTrained('Model is not trained yet!')
        else:
            # User ordered dict here to preserve order of training, val, test
            stats = collections.OrderedDict()

            stats['train_loss'], stats['train_accuracy'] \
                = self.get_train_stats()
            stats['validation_loss'], stats['validation_accuracy'] \
                = self.get_validation_stats()
            stats['test_loss'], stats['test_accuracy'] \
                = self.get_test_stats()

            return stats

    def print_model_summary(self):
        if not self.is_trained:
            raise ModelNotTrained('Model is not trained yet!')
        else:
            return self.model.summary() 

    def plot_learning_curve(self):
        """ Plot the train/validation loss and accuracies """
        if not self.is_trained:
            raise ModelNotTrained('Model is not trained yet!')
        else:
            # Plot history for accuracy
            plt.subplot(2, 1, 1)
            plt.plot(self.history.history['acc'])
            plt.plot(self.history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.show()
            # Plot history for loss
            plt.subplot(2, 1, 2)
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.show()

    def save(self, name):
        if not self.is_trained:
            raise ModelNotTrained('Model is not trained yet!')
        else:
            self.model.save_weights(name+'.h5')
    def plot_training_data_all(self,PATH,name, start):
        from constants import target_names_all
        complete_data_set = self.load_data(PATH,name)
        label_set = complete_data_set[:,-1]#get correct answer labels
        data_set = complete_data_set[:,:-1] # remove correct answer labels
        
        print(data_set.shape)
        print(data_set.shape[0])
        
        num_of_samples=data_set.shape[0]
        
        divisor = 100
        total_figures = int(np.ceil(num_of_samples/divisor))
        
#        test_num = 25
#        num_of_samples = test_num
        #include all input data in title
        subplot_num=int(np.ceil(np.sqrt(divisor)))
        print(subplot_num)
        multiplier = 5
        
        for j in range(start,total_figures):
            fig=plt.figure(figsize=(subplot_num*multiplier,subplot_num*multiplier))
            print(j)
            for i in range(divisor): 
                print(i)
                ax = fig.add_subplot(subplot_num, subplot_num, i+1)
                try:
                    temp_image =np.reshape(data_set[j*divisor+i],(100,100))
                    temp_image =np.reshape(data_set[j*divisor+i],(100,100))
                    temp_ans = int(label_set[j*divisor+i])
                    temp_name = target_names_all[temp_ans]
                        #plot extractions with matching percentages and corresponding color of text
                    ax.set_title(str(j*divisor+i)+": prediction is : '"+ str(temp_ans)+ "'"+'\n'+ str(temp_name))
                    ax.imshow(temp_image,cmap = 'binary')
                except IndexError:
                    continue
                plt.tight_layout()
            plt.show()
        
                #save the figure by name
            fig.savefig(PATH + 'complete_data_set'+str(name)+'_'+str(j)+'.jpg')
            print('Saved extractions figure as' + PATH+ 'complete_data_set_'+str(name)+'_'+str(j)+'.jpg')
#            fig.close()
    def clean_dataset(self,PATH,name):
#        from constants import target_names_all
        from skimage import measure

        delete_list = [1724, 1816, 1836, 2403,
        2499, 2648, 2731, 3153, 3283, 3359, 3517, 3587, 
        3601, 3678, 3698, 3722, 3725, 3726, 3728, 3729, 3730,
        3819, 4776, 4888,
        5757, 5761, 6024, 6106, 6797, 7735, 7739, 7774, 7825, 7924, 7975, 7976, 7977, 7980, 7984, 8066,
        8075, 8142, 8174, 8178, 8196, 8197, 8205, 8270, 8274, 8553, 8567, 8898, 9015, 9116, 9119, 9186, 9187, 9197, 9205, 
        9235, 9237, 9270, 9361, 9523, 9640, 9741, 9744,
        9811, 9812, 9822, 9830, 9860, 9862, 9895, 
        9986, 10148, 10265, 10366, 10369, 
        10436, 10437, 10447, 10455, 10485, 10487, 10520, 10611, 10773, 10890, 10991, 10994, 11061, 11062, 11072, 11080, 
        11236, 11398, 11515, 11616, 11619, 11686, 11687, 11697, 11705, 11735, 11770, 11861, 12023, 12140, 12241, 12244, 12311, 12312, 12322, 12330, 12360, 12362, 12395, 12486,
        12648, 12765, 12866, 12869, 12936, 12937, 12947, 12955, 12985, 12987, 13020, 13111, 13273, 13390, 13491, 13494, 13561, 13562, 13572, 13580, 13610, 13612, 13645, 13736, 
        #14395, 14446, 14558, 14651, 14656, 14938,
        17675, 17849, 18300, 18474, 18837, 18852, 18919, 19008, 19099, 19180, 19239, 19462, 19477, 19544, 19633, 19724, 19805, 19864, 20000, 20046, 20072, 20117, 20149, 20199, 20271, 20275, 20386, 20400, 20424, 20464, 20509, 20568, 20572, 20576, 20594, 20625, 20671, 20697,
        20742, 20774, 20824, 20896, 20900, 21011, 21025, 21049, 21089, 21134, 21193, 21197, 21201, 21219, 21250, 21290, 21322, 21367, 21396, 21399, 21449, 21521, 21525, 21636, 21650, 21674, 21714, 21759, 21818, 21822, 21826, 21844, 21875, 21915, 21947, 21992, 22021, 22024, 22074, 22146, 22150, 22261, 22275, 22299, 22339, 22384, 22442, 22447, 22451, 22469, 22529, 22852, 22857, 23017, 23040, 23049, 23081, 23105, 23154, 23477, 23482, 23642, 23665, 23674, 23706, 23730, 23756, 23769, 23779, 23774, 23790, 23808, 23816, 24102
        ]
    
        resize_list = [2350, 2375, 2383, 3600, 3602, 3607, 3629, 
        3770, 3796, 3847, 3849, 3896,
        3924, 4125, 4150, 4180, 4200, 4201, 4203, 4206, 4225, 4226, 4227, 4234, 4250, 4257, 4275, 4276, 4277, 4278, 4279,
        4285, 4303, 4350, 4351, 4355, 4394, 4395, 4396, 4397, 4398, 4399,
        4416, 4421, 4422, 4423, 4424, 4441, 4446, 4447, 4448, 4449, 4473, 4474, 4524,
        4802, 4825, 4829, 4854, 4903, 4925, 4926, 4952, 5021, 5022, 5023, 5024, 5048, 5049, 5074, 5099,
        5121, 5426, 5504, 5645, 5670, 6050, 6259, 6267, 6268, 6269, 6270, 6271, 6272, 6273, 6274, 6289, 6296, 6297, 6298, 6299, 6316, 6321, 6323, 6324, 7301, 7329, 7356, 7378, 7406, 7429, 7454, 7520, 7574, 7623, 7674, 7954, 8000,
        8079, 8144, 8145, 8146, 8147, 8148, 8149, 8162, 8165, 8171, 8172, 8173, 8195, 8198, 8199, 8215, 8271, 8272, 8273, 8298,
        8789, 8823, 8843, 8844, 8866, 9014, 9121, 9232, 9414, 9468, 9469, 9554, 9639, 9746, 9857, 10073, 10094, 10179, 10264, 10371, 10482, 10698, 10664, 10718, 10719, 10741, 10996, 11107, 11110, 11112, 11145, 
        11289, 11323, 11343, 11344, 11366, 11429, 11514, 11621, 11732, 11914, 11969, 11948, 12054, 12139, 12246, 12357, 12539, 12573, 12593, 12594,
        12679, 12764, 12871, 12982, 13151, 13164, 13198, 13218, 13219, 13241, 13304, 13389, 13496, 13607
        ] 
        
        swap_list = [3874, 29, 266, 282, 283, 30068, 30127, 30197, 30177, 30195
        ]
        
#        swap_list_index = [5, 6, 6, 9, 9, 53, 23, 23, 23, 23
#        ]
        
        complete_data_set = self.load_data(PATH,name)
        label_set = complete_data_set[:,-1]#get correct answer labels
        data_set = complete_data_set[:,:-1] # remove correct answer labels
        temp_complete_data_set = []
        for i in range(data_set.shape[0]):
            print(i)
            if i in delete_list:
                continue
            elif i in resize_list:
                continue #originally fix it
            elif i in swap_list:
                if i == 3874:
                    new_index = 5
                elif i == 29 or i == 266:
                    new_index = 6
                elif i == 282 or i == 283:
                    new_index = 9
                elif i == 30068:
                    new_index = 53
                elif i == 30127 or i == 30197 or i == 30177 or i == 30195:
                    new_index = 23
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
                #check if duplicate
                
                
                data_set_line = data_set[i,:]
                label_set_line = label_set[i]
                new_data_line = np.hstack((data_set_line,label_set_line))
                temp_complete_data_set.append(new_data_line)
        #change back to array
        complete_data_set = np.asarray(temp_complete_data_set)
        np.save(PATH+"Training_Samples_64_classes_100x100_all_cleaned_"+str(complete_data_set.shape[0]), complete_data_set)
        print('saved as :'+ str(PATH) + "Training_Samples_64_classes_100x100_all_cleaned_"+str(complete_data_set.shape[0]))
    def count_dataset(self, dataset_PATH, dataset_name, num_classes):
        data_all = self.load_data(dataset_PATH, dataset_name)
        label_list = data_all[:,-1]
        count_list = np.zeros(num_classes).astype(np.int).tolist()
        x_list = np.arange(num_classes).tolist()
        for i in label_list:
            count_list[i] += 1
            
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(25, 25))
        plt.bar(x_list,count_list)
        plt.title("Dataset Count")
        plt.xlabel("Class index")
        plt.ylabel("Count")
        plt.show()
        fig.savefig(dataset_PATH+dataset_name+'dataset_count')
        print(count_list)
        return count_list
        