
class ComponentClassification(object):
    
    # Hyperparameters
    BATCH_SIZE = 200
    num_classes = 64
    epochs = 300
    training_ratio = 0.7
    img_rows, img_cols = 100, 100
    min_percent_match = 0 # set to 0.7 possiby set to 0.5
    min_confidence = 0 # set to 0.6 possibly set to 0.3


        #function: split dataset randomly
    def random_split_dataset(data_set, training_ratio):
        """
        Split data set randomly
        Further optimization: Convert it into numpy
        """
        l = data_set.shape[0]
        f = int(l * training_ratio)
        train_indices = sample(range(l),f)
        test_indices = np.delete(np.array(range(0, l)), train_indices)
        train_data = data_set[train_indices]
        test_data = data_set[test_indices]
        x_train = train_data[:,:-1]
        y_train = train_data[:,(-1)]
        y_train=y_train.reshape(y_train.shape[0],1)
        print(x_train.shape)
        x_test = test_data[:,:-1]
        y_test = test_data[:,(-1)]
        y_test = y_test.reshape(y_test.shape[0],1)
        return x_train, y_train, x_test, y_test

    def load_data():
        """ Load pre-shuffled data """
        data_all = np.load('/home/chloong/Desktop/Justin San Juan/Testing Folder/'+'Training_Samples_'+str(num_classes)+'_classes_'+str(img_rows)+'x'+str(img_cols)+'_all'+'.npy')
        train_data=np.load('/home/chloong/Desktop/Justin San Juan/Testing Folder/'+'Training_Samples_'+str(num_classes)+'_classes_'+str(img_rows)+'x'+str(img_cols)+'_'+'train_data.npy')
        test_data=np.load('/home/chloong/Desktop/Justin San Juan/Testing Folder/'+'Training_Samples_'+str(num_classes)+'_classes_'+str(img_rows)+'x'+str(img_cols)+'_'+'test_data.npy')
        x_train = train_data[:,:-1]
        y_train = train_data[:,(-1)]
        y_train=y_train.reshape(y_train.shape[0],1)
        x_test = test_data[:,:-1]
        y_test = test_data[:,(-1)]
        y_test = y_test.reshape(y_test.shape[0],1)
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
            
            # Reshape back to 3D matrix to be passed into CNN
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)
        
            # Necessary transformation
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)
            
            #change data type    
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        
            # Preparation and training of neural network\n",
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
            
            # Convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        print('Loaded dataset')
        
        return x_train, y_train, x_test, y_test, input_shape, data_all

    def load_model_layers(d):
        """
        Load Sketch-A-Net keras model layers ***model must have been declared as a global variable
        """
        #L1
        model.add(Conv2D(64, (15,15),strides=3, activation='relu',input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(3, 3),strides=1))
        model.add(Dropout(d))
        #L2
        model.add(Conv2D(128, (5,5),strides=1, activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
        model.add(Dropout(d))
        #L3
        model.add(Conv2D(256, (3,3),strides=1,padding='same', activation='relu'))
        model.add(Dropout(d))
        #L4
        model.add(Conv2D(256, (3,3),strides=1,padding='same', activation='relu'))
        model.add(Dropout(d))
        #L5
        model.add(Conv2D(256, (3,3),strides=1,padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
        model.add(Dropout(d))
        #L6
        model.add(Conv2D(512, (5,5),strides=1, activation='relu'))
        model.add(Dropout(0.5))
        #L7
        model.add(Conv2D(512, (1,1),strides=1, activation='relu'))
        model.add(Dropout(0.5))
        #L8
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))
        
        model.summary()
        model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])

        #function: train model with input number of epochs ***model must have been declared as a global variable
    def train_model(epochs):
        model.fit(x_train, y_train, batch_size=batch_size,
                                    epochs=epochs,
                                    verbose=1,
                                    validation_data=(x_test, y_test))
        
        #function: save model weights ***model must have been declared as a global variable
    def save_model_weights(name,epochs):
        model.save_weights(name + '_all_'+str(epochs)+"epochs"+".h5")
        print('saved model weights as '+name + "_" + str(epochs)+"epochs"+".h5")
        
        #function: load model weights from file ***model must have been declared as a global variable
    def load_model_weights(name,epochs):
        model.load_weights(name + "_all_"+ str(epochs)+"epochs"+".h5")
        print('loaded model weights from '+name + "_" + str(epochs)+"epochs"+".h5")

    def predict_classes(ext_images,group,ext_class_index,ext_class_name,ext_next_round,ext_next_round_index):
        if group =='numbers' or group =='all': 
            indices = range(len(ext_images))
        else: 
            indices = ext_next_round_index[:]
        if indices != []:
            for i in indices:
                image = ext_images[i]
        ########### ADD DIMENSIONS TO MATCH CLASSIFIER DIMENSIONS ################
                num_channel = 3 # since we need RGB
                
                if num_channel==1: # if classifier only needs 1 channel
                    if K.image_dim_ordering()=='th': # modify data if using theano instead of tensorflow
                        image = np.expand_dims(image, axis=0)
                        image = np.expand_dims(image, axis=0)
                    else:
                        image = np.expand_dims(image, axis=3) 
                        image = np.expand_dims(image, axis=0)
                        
                else:
                    if K.image_dim_ordering()=='th': # modify data if using theano instead of tensorflow
                        image=np.rollaxis(image,2,0)
                        image = np.expand_dims(image, axis=0)
                    else:
                        # expand dimensions as needed in classifier
                        image = np.expand_dims(image, axis=3)
                        image = np.expand_dims(image, axis=0)
        
        ########### PREDICT OBJECT CLASS W/ ENTROPY THEORY & RECORD DATA ############## 
        
                    # get match percentages for each class from classifier
                prediction=model.predict(image)
                
                second_max=list(prediction[0])
                second_max.remove(max(second_max))
               
                    # get first, second, and third maximum percentage matches, to be used for entropy calculations
                first_max=max(prediction[0])
                second_max=max(second_max)
                
                    # attach percentages to lists (in range of 0 to 1.0, ex: 91% is recorded as 0.91)
                ext_match_percent.append(first_max)
                ext_match_percent2.append(second_max)
                
                    # if prediction is not confident or if confidence, as calculated by the difference top two predictions is too hight, or if another third prediction is close to the second prediction
                    # discard =raction as an 'unknown' class
                if first_max < min_percent_match or first_max-second_max < min_confidence:
                    index=17 # index 17 is class 18, the unknown class
                    
                    # otherwise, if prediciton is confident, record the index and class name
                else:
                    index=((prediction[0]).tolist().index(first_max))
                
                    #save extractions
                ext_class_index[i] = index
                ext_class_name[i] = target_names_all[index]

    def hyperparameter_search():
        pass

    def calculate_recall():
        pass

    def calculate_precision():
        pass

    def calculate_accuracy():
        pass

    def calculate_F1_score():
        pass

    def plot_confusion_matrix():
        from sklearn.metrics import confusion_matrix
        import itertools

        Y_pred = model.predict(x_test)
        y_pred = np.argmax(Y_pred, axis=1)

                            
        # Plotting the confusion matrix
        def plot_confusion_matrix(cm, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')

            print(cm)

            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, cm[i, j],
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            
        # Compute confusion matrix
        cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), y_pred))

        np.set_printoptions(precision=2)

        plt.figure(figsize=(30,30))

        # Plot non-normalized confusion matrix
        plot_confusion_matrix(cnf_matrix, classes=target_names,
                              title='Confusion matrix')

        plt.show()
                #*******************saving does not work properly
        try:
            plt.savefig('/home/chloong/Desktop/Justin San Juan/Testing Folder/Output/cnf_matrix_'+name+'_'+str(epochs)+'epochs.jpg')
        except:
            plt.savefig('C:/Users/JustinSanJuan/Desktop/HKUST/UROP Deep Learning Image-based Structural Analysis/Code/Python/Testing Folder/cnf_matrix_'+name+'_'+str(epochs)+'epochs.jpg')



    def class_mapping():
        try:
            adjusted_ans = ans[:]
            for r in range(0,int(len(ans))):
                if ans[r] >= 14 and ans[r] <= 17:
                    adjusted_ans[r] = 14
                elif ans[r] >=18 and ans[r] <=21:
                    adjusted_ans[r] = 15
                elif ans[r] ==22:
                    adjusted_ans[r] = 16
                elif ans[r] ==23:
                    adjusted_ans[r] = 17
                
                elif ans[r] >= 24 and ans[r] <=31:
                    adjusted_ans[r] = 18
                elif ans[r] >=32 and ans[r] <=39:
                    adjusted_ans[r] = 19
                elif (ans[r] >=40 and ans[r] <=41) or (ans[r] >=44 and ans[r]<=45):
                    adjusted_ans[r] = 20
                elif (ans[r] >=42 and ans[r] <=43) or (ans[r] >=46 and ans[r]<=47):
                    adjusted_ans[r] = 21
                elif ans[r] >=48:
                    adjusted_ans[r] = adjusted_ans[r] - 26
        except:
            adjusted_ans=''
            #create figure with all extractions and percent matches if no answers

    def adjust_predictions(ext_class_index,ext_class_name):
        for k in range(len(ext_class_index)):
            index= ext_class_index[k]
            
            if index >=14 and index <=17:
                index = 14
                
                #adjust predictions to merge clockwise moments
            elif index >=18 and index <=21:
                index = 15
                #adjust index of noise
            elif index == 22:
                index = 16
                
                #adjust index of random letters
            elif index ==23:
                index = 17
                
                #adjust predicitons to merge fixed supports
            elif index >= 24 and index <=31:
                index = 18
                
                #adjust predicitons to merge pinned supports
            elif index >=32 and index <=39:
                index = 19
                
                #adjust predictions to merge vertical roller supports
            elif (index >=40 and index <=41) or (index >=44 and index<=45):
                index = 20
                
                #adjust predicitons to merge horizontal roller supports
            elif (index >=42 and index <=43) or (index >=46 and index<=47):
                index = 21
                
                #adjust index of last 12 classes
            elif index >=48 and index <=63:
                index = index - 26
            
            ext_class_index[k] = index
            ext_class_name[k] = target_names[index]