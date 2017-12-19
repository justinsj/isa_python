import collections
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import callbacks
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
                      optimizer='adam',
                      metrics=['accuracy'])

        return model

    def train(self, epochs, seed, verbose=1):
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
        """ name should end with .h5 """
        if not self.is_trained:
            raise ModelNotTrained('Model is not trained yet!')
        else:
            self.model.save_weights(name)
