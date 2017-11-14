import collections
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical


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
    batch_size = 200
    img_rows, img_cols = 100, 100

    def __init__(self, training_data, val_data, test_data, dropout=0):
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
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, \
            self.num_classes = self.load_data(training_data, val_data, test_data)
        self.model = self.load_sketch_a_net_model(dropout)

    def load_data(self, training_data, val_data, test_data):
        """ Load all data to be directly passed into the model """

        # Convert input data to 2D float data
        # The last values in second dimension are the labels
        X_train, y_train = training_data[:, 0:-1].astype('float32'), training_data[:, -1]
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
        num_classes = len(np.unique([y_train, y_val, y_test]))  # Obtain number of classes
        y_train = to_categorical(y_train, num_classes)
        y_val = to_categorical(y_val, num_classes)
        y_test = to_categorical(y_test, num_classes)

        return X_train, y_train, X_val, y_val, X_test, y_test, num_classes

    def load_sketch_a_net_model(self, dropout):
        """ Load Sketch-A-Net keras model layers """
        input_shape = self.X_train.shape[1:]
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
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model

    def train(self, epochs, verbose=0):
        """ Train model with input number of epochs """
        np.random.seed(2017)  # For reproducibility
        self.history = self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size,
                                      epochs=epochs, verbose=verbose,
                                      validation_data=(self.X_test, self.y_test))
        self.is_trained = True

    def get_train_stats(self):
        """ Get training loss and ccuracy from history """
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
