# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 01:32:43 2018

@author: JustinSanJuan
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import callbacks
from keras.optimizers import Adadelta
dropout = 0
input_shape = (100,100,1)
num_classes = 64


    @staticmethod
    def load_sketch_a_net_model_1(dropout, num_classes, input_shape,verbose = False):
        """ Load Sketch-A-Net keras model layers """
        #model_1 inc size of first filter, 2.3m parameters
        model = Sequential()
        
        # Layer 1
        model.add(Conv2D(64, (20, 20), strides=5, activation='relu', input_shape=input_shape))
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
        model.add(Conv2D(512, (2, 2), strides=1, activation='relu'))
        model.add(Dropout(0.5))
        
        # Layer 7
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))
        
        
        model.compile(loss='categorical_crossentropy',
                  optimizer=Adadelta(),
                  metrics=['accuracy'])
        if verbose:
            model.summary()
        return model
    @staticmethod
    def load_sketch_a_net_model_2(dropout, num_classes, input_shape,verbose = False):
        """ Load Sketch-A-Net keras model layers """
        #model_2 inc size of first filter and inc number, 2.5m parameters
        model = Sequential()
        
        # Layer 1
        model.add(Conv2D(128, (20, 20), strides=5, activation='relu', input_shape=input_shape))
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
        model.add(Conv2D(512, (2, 2), strides=1, activation='relu'))
        model.add(Dropout(0.5))
        
        # Layer 7
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))
        
        
        model.compile(loss='categorical_crossentropy',
                  optimizer=Adadelta(),
                  metrics=['accuracy'])
        if verbose:
            model.summary()
        return model
    
    @staticmethod
    def load_sketch_a_net_model_3(dropout, num_classes, input_shape,verbose = False):
        """ Load Sketch-A-Net keras model layers """
        #model_3 inc number of first filter only, 5m parameters
        model = Sequential()
        
        # Layer 1
        model.add(Conv2D(128, (15, 15), strides=3, activation='relu', input_shape=input_shape))
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
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))
        
        
        model.compile(loss='categorical_crossentropy',
                  optimizer=Adadelta(),
                  metrics=['accuracy'])
        if verbose:
            model.summary()
        return model
    @staticmethod
    def load_sketch_a_net_model_4(dropout, num_classes, input_shape,verbose = False):
        """ Load Sketch-A-Net keras model layers """        #model_4 decrease first filter size, 5m parameters
        model = Sequential()
        
        # Layer 1
        model.add(Conv2D(64, (10, 10), strides=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=1))
        model.add(Dropout(dropout))
        # Layer 2
        model.add(Conv2D(128, (7, 7), strides=1, activation='relu'))
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
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))
        
        
        model.compile(loss='categorical_crossentropy',
                  optimizer=Adadelta(),
                  metrics=['accuracy'])
        if verbose:
            model.summary()
        return model
    @staticmethod
    def load_sketch_a_net_model_5(dropout, num_classes, input_shape,verbose = False):
        """ Load Sketch-A-Net keras model layers """
        #model_5 double all filters, 20m parameters
        model = Sequential()
        
        # Layer 1
        model.add(Conv2D(128, (15, 15), strides=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=1))
        model.add(Dropout(dropout))
        # Layer 2
        model.add(Conv2D(256, (5, 5), strides=1, activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
        model.add(Dropout(dropout))
        # Layer 3
        model.add(Conv2D(512, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(Dropout(dropout))
        # Layer 4
        model.add(Conv2D(512, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(Dropout(dropout))
        # Layer 5
        model.add(Conv2D(512, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
        model.add(Dropout(dropout))
        # Layer 6
        model.add(Conv2D(1024, (5, 5), strides=1, activation='relu'))
        model.add(Dropout(0.5))
        
        # Layer 7
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy',
                  optimizer=Adadelta(),
                  metrics=['accuracy'])
        if verbose:
            model.summary()
        return model