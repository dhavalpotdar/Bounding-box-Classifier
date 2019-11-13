import os
from collections import Counter
import random
import configparser
import cv2
import pandas as pd
import numpy as np
import keras
from keras.layers import Dense, BatchNormalization, Dropout, Flatten
from keras import Model
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model, model_from_json

class ContainerClassifier():
    def __init__(self):
        self.label_map = None
        self.train_data = None
        self.train_labels = None
        self.label_binarizer = None
        self.test_data = None
        self.test_labels = None
        self.model = None
        self.class_dict = None
        self.original_sizes = None

    ########################## private methods #################################
    def _read_paths(self, folder_path):
        
        '''
        Reads both yolo txt and jpg files in given path
        
        Args:
            path to folder
        
        Returns:
            list of tuples: [(path/img_name.txt, path/img_name.jpg)]
        '''
        
        for root, dirs, files in os.walk(folder_path):
            root_path = root
            file_list = files
        
        # separate out txt file names and acquire txt file names accordingly
        jpg_files = [file for file in file_list if file.endswith('.jpg')]
        txt_files = [jpg_file_name[:-4] + '.txt' for jpg_file_name in jpg_files]    
        
        # add root path in front of file name
        txt_files = [root_path + '/' + file_name for file_name in txt_files]
        jpg_files = [root_path + '/' + file_name for file_name in jpg_files]
        
        data_path_listoftuples = []
        for item in zip(txt_files, jpg_files):
            data_path_listoftuples.append(item)

        random.shuffle(data_path_listoftuples)

        return data_path_listoftuples
    

    def _denormalize(self, size, box):
        # size list[width, height] of image, box list[x,y,w,h] from LabelImg
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        x = x*size[0]
        y = y*size[1]
        w = w*size[0]
        h = h*size[1]


        x1 = (2*x-w)/2.0
        y1 = (2*y-h)/2.0
        x3 = (2*x+w)/2.0
        y3 = (2*y+h)/2.0
        
        return [int(x1),int(y1), int(x3),int(y3)]

    def _create_data(self, data_path_listoftuples, resize_to, 
                     use_labels, training=True): 
        
        ''' 
        Function to implement the logic for creating dataset.

        Args:
            data_path_listoftuples: list of tuples - 
                                    [(path/img_name.txt, path/img_name.jpg)]

            resize_to: tuple

        Returns:
            data_array

            label_array
        '''
        data_list = []
        label_list = []
        original_sizes = []

        # if training:
            # dict to reindex arbitrary labels to sequential mapping starting 
            # from 0 to len(use_labels)
        mapping_dict = {i: k for k, i in enumerate(use_labels)}
        self.label_map = mapping_dict

        for txt_file_path, img_file_path in data_path_listoftuples:

            img = cv2.imread(img_file_path)
            height, width = img.shape[:-1] # 3rd dimension is number of channels

            with open(txt_file_path) as fh:         #fh - file handle
                for line in fh:

                    # 1st item of the line is the index of label,
                    # the rest are coordinates

                    float_list = [float(coordinate) for coordinate in \
                                    line[1:].split()]

                    label = float(line[0])

                    xmin, ymin, xmax, ymax = self._denormalize((width, height), 
                                                                float_list)

                    if label in use_labels:
                        
                        img_crop = img[ymin:ymax, xmin:xmax]
                        
                        if not training:
                            original_sizes.append(img_crop)

                        #copyMakeBorder(image, top, bottom, left, right, color):
                        # top, bottom, left, right are the number of pixels on 
                        # respective side to add blank border to
                        WHITE = [255, 255, 255]
                        # image = cv2.copyMakeBorder(img_crop, ymin, height-ymax, 
                        #                             xmin, width-xmax,
                        #                             cv2.BORDER_CONSTANT,
                        #                             value=WHITE)

                        # cv2.imwrite('./img.jpg', image)
                       
                        image = cv2.resize(img_crop, tuple(resize_to))
                        
                        # cv2.imwrite('./resized.jpg', image)
                        
                        data_list.append(image)
                        label_list.append(self.label_map[label])
        
        label_arr = np.array(label_list)
        data_arr = np.array(data_list)
        
        # convert labels to one hot and save binarizer for testint
        if training:
            lb = LabelBinarizer()
            lb.fit(list(self.label_map.values()))
            self.label_binarizer = lb
            label_arr = self.label_binarizer.transform(label_arr)

        if training:
            return data_arr, label_arr

        else:
            return data_arr, label_arr, np.array(original_sizes)


    ########################### public methods #################################
    
    def read_train_data(self, path, resize_to=(512, 512),  
                        class_dict={0: 'Vertical Relational', 
                                    1: 'Horizontal Entity', 
                                    2: 'Multiline Text', 
                                    3: 'Other Tabular', 
                                    4: 'Header', 
                                    5: 'Other', 
                                    6: 'Logo'}, 
                        use_labels=[0, 1, 2, 3, 5, 6]):
        # takes path to training data folder 
        '''
        Args:
            path: 
                path to training data folder having:
                    - img_name.txt, containing labels and coordinates of 
                      containers 
                    - img_name.jpg

            resize_to:
                shape tuple of size for each image

            use_labels:
                the labels to use from img_name.txt

                lables as defined in predefined_classes.txt in labelImg 
                       
                |      class         |   label    |
                |--------------------|------------|
                | vertical relational|      0     |
                | horizontal entity  |      1     |
                | multiline text     |      2     |  
                | other tabular      |      3     |  
                | header             |      4     |  
                | other              |      5     |  
                | logo               |      6     |
    
        
        Returns:
            None. Saves the train data and labels array as properties of the 
            class.
        '''
        print('Loading training data...')
        self.class_dict = class_dict

        # read the paths
        data_path_listoftuples = self._read_paths(path)

        # create training data and save it as class property
        data_arr, label_arr = self._create_data(data_path_listoftuples, 
                                                      resize_to, 
                                                      use_labels)

        data_arr = data_arr/255.0                                                      
        self.train_data = data_arr
        self.train_labels = label_arr
        self.resize_to = resize_to


    def read_test_data(self, path, resize_to=(512, 512),
                                   class_dict={0: 'Vertical Relational', 
                                               1: 'Horizontal Entity', 
                                               2: 'Multiline Text', 
                                               3: 'Other Tabular', 
                                               4: 'Header', 
                                               5: 'Other', 
                                               6: 'Logo'}, 
                                    use_labels=[0, 1, 2, 3, 5, 6]):

        '''
        Args:
            path: 
                path to training data folder having:
                    - img_name.txt, containing labels and coordinates of 
                      containers 
                    - img_name.jpg 

        Returns:
            None, saves the test data and labels as class properties
        '''
        print('Loading testing data...')
        self.class_dict = class_dict
        self.resize_to = resize_to

        # read paths
        data_path_listoftuples = self._read_paths(path)

        # create testing data and save it as class property
        data_arr, label_arr, original_sizes = self._create_data(
                                                        data_path_listoftuples, 
                                                        self.resize_to,
                                                        use_labels=use_labels, 
                                                        training=False)
        
        data_arr = data_arr/255.0
        self.test_data = data_arr
        self.test_labels = label_arr
        self.original_sizes = original_sizes


    def train(self, model, epochs=5, batch_size=10, optimizer='adam', 
                summary=True):
        
        '''
        Args:
            model: 
                None if using pretrained model
                'model_name' if training

            summary:
                bool, whether to print out model summary
        '''

        # get input shape of resized images and add the channels dimension
        input_shape = list(self.resize_to)
        input_shape.append(3)
        shape_tuple = tuple(input_shape)

        if model == 'pretrained':

            model = self.model

        elif model == 'resnet101':
            model = keras.applications.resnet.ResNet101(include_top=True, 
                                                        weights=None, 
                                                        input_tensor=None, 
                                                        input_shape=shape_tuple, 
                                                        pooling=None, 
                                                        classes=\
                                                            len(self.label_map))

        elif model == 'densenet121':
            model = keras.applications.densenet.DenseNet121(include_top=True, 
                                                            weights=None, 
                                                            input_tensor=None, 
                                                            input_shape=\
                                                                shape_tuple, 
                                                            pooling=None, 
                                                            classes=\
                                                            len(self.label_map))

        elif model == 'resnet50':
            model = keras.applications.resnet.ResNet50(include_top=True, 
                                                       weights=None, 
                                                       input_tensor=None, 
                                                       input_shape=shape_tuple, 
                                                       pooling=None, 
                                                       classes=\
                                                           len(self.label_map))

        elif model == 'resnet152':
            model = keras.applications.resnet.ResNet152(include_top=True, 
                                                        weights=None, 
                                                        input_tensor=None, 
                                                        input_shape=shape_tuple, 
                                                        pooling=None, 
                                                        classes=\
                                                            len(self.label_map))

        else: 
            raise ValueError('Please Enter valid model name one of:\
                "resnet50", "resnet101", "resnet152", "densenet121"')

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, 
                        metrics=['accuracy'])

        if summary:
            print(model.summary())                                                                                            

        print(' ')                                                                                                        
        print('Training on {} examples...'.format(self.train_labels.shape[0]))
        print('-'*100)

        model.fit(self.train_data, self.train_labels, batch_size=batch_size, 
                    epochs=epochs, verbose=1, shuffle=True)

        print(' ')
        print('Training Finished.')

        # save model as property
        self.model = model
        
    def save_model(self, name, path='./'):
        
        print(' ')
        print('Saving Model...')
        print(' ')
        # serialize model to JSON
        model_json = self.model.to_json()
        
        with open(path + name + '.json', "w") as json_file:
            json_file.write(model_json)
            
        # serialize weights to HDF5
        self.model.save_weights(path + name + '.h5')
        print("Saved model to disk")
        

    def load_model(self, json_path, weights_path):
        
        # load json and create model
        json_file = open(json_path, 'r')
        
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json)
        
        # load weights into new model
        loaded_model.load_weights(weights_path)
        
        print("Loaded model from disk")

        self.model = loaded_model


    def test_model(self, output_dir='test_output', top_k=3, plot_outputs=False):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model = self.model
        data = self.test_data
        labels = self.test_labels
        og_sizes = self.original_sizes

        print(' ')
        print('Predicting on {} samples...'.format(self.test_labels.shape[0]))
        print('-'*100)

        preds = model.predict(data)

        # for top-k predictions
        for pred_arr in preds:
            ind = np.argpartition(pred_arr, -top_k)[-top_k:]
            top_k_probs = pred_arr[ind]
            top_k_classes = ind
            top_k_dict = dict(zip(top_k_classes, top_k_probs))
            print(top_k_dict)

        # for most confident prediction, the metrics are calculated for these
        preds = np.argmax(preds, axis=1)

        print(' ')
        print('Classification Report:')
        print('-'*50)
        print(classification_report(labels, preds))
        print(' ')
        print('Confusion Matrix:')
        print('-'*50)
        print(confusion_matrix(labels, preds))

        # invert label dictionary
        inv_label_map = {v: k for k, v in self.label_map.items()}

        if plot_outputs:
            count = 1
            for img, pred in zip(og_sizes, preds):
                cv2.putText(img, self.class_dict[inv_label_map[pred]], 
                            (0, int(img.shape[0])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                cv2.imwrite('./' + output_dir + '/' + str(count) + '.jpg', img)
                count += 1        



class Config(configparser.ConfigParser):

    '''
        Class that inherits from ConfigParser and makes accessing parameters
        from the config file easy.

        Methods:
            create_config(path):
                Takes the path of the `config.ini` file and saves parameters
                as class objects

            _test_confit():
                Private method to ensure validity of config parameters.
    '''

    def __init__(self):
        configparser.ConfigParser.__init__(self)
        self.classes = None
        self.use_labels = None
        self.resize_to = None
        self.architecture = None
        self.epochs = None
        self.batch_size = None
        self.optimizer = None
        self.print_model_summary = None
        self.model_name = None
        self.save_to_dir = None
        self.load_from_dir = None
        self.test_data_folder_path = None
        self.test_output_dir = None
        self.top_k = None
        self.plot_outputs = None
        self.class_dict = None

    def create_config(self, path):
        self.read(path)
        
        # train specs
        self.train_data_folder_path = self['train specs']['TrainDataFolderPath']
        self.classes = \
            list(map(str.lstrip, self['train specs']['Classes'].split(',')))
        self.use_labels = \
            list(map(str.lstrip, self['train specs']['UseLabels'].split(',')))
        self.resize_to = \
            list(map(int, self['train specs']['ResizeTo'].split(',')))    
        self.epochs = int(self['train specs']['Epochs'])
        self.batch_size = int(self['train specs']['BatchSize'])
        
        # model params
        self.architecture = self['model params']['ModelArchitecture']

        self.optimizer = self['model params']['Optimizer']
        self.print_model_summary = bool(self['model params']\
            ['PrintModelSummary'])

        # load model
        self.model_name = self['save/load model']['ModelName']
        self.save_to_dir = self['save/load model']['SaveToDir']
        self.load_from_dir = self['save/load model']['LoadFromDir']

        # test specs
        self.test_data_folder_path = self['test specs']['TestDataFolderPath']
        self.test_output_dir = self['test specs']['TestOutputDir']
        self.top_k = int(self['test specs']['TopK'])
        self.plot_outputs = bool(self['test specs']['PlotOutputs'])
        
        # convert the classes list to dict
        self.class_dict = {k:v for k,v in enumerate(config.classes)}

        # call the tests to ensure validity of parameters passed
        self._test_config()

        # we map the class labels in self.use_labels to class indices using the
        # self.class_dict
        inv_class_dict = self._invert_dict(self.class_dict)
        self.use_labels = [inv_class_dict[i] for i in self.use_labels]


    def _invert_dict(self, dct):
        return {v:k for k,v in dct.items()}

    def _test_config(self):
        
        # test train specs
        assert len(set(self.use_labels) - set(self.classes)) == 0,\
            '''`UseLabels` must be a subset of `Classes`. 
        Received UseLabels: {}
        Received Classes: {}'''.format(self.use_labels, self.classes)

        assert len(self.use_labels) != 0, \
                'UseLabels cannot be empty.'

        assert self.resize_to[0] == self.resize_to[1], \
            '''The shape tuple passed to ResizeTo should be square.
            Received shape: {}'''.format(self.resize_to)

        # test model params
        possible_models = ['resnet101', 'resnet50', 'densenet121', 'resnet152']
        assert self.architecture in possible_models, \
            '''Model Architecture should be one of:
            {}
            Received Architecture: {}'''.format(possible_models, 
                                                self.architecture)

        assert self.batch_size <= 4, \
                '''BatchSize should be less than 4. Having a larger batch size
                results in Memory Error of GPU. Received BatchSize: {}'''.\
                    format(self.batch_size)

        assert self.top_k <= len(self.use_labels), \
                '''TopK must be less than length of UseLabels.
                Received TopK : {}'''.format(self.top_k)
            
        print('All tests passed. All Config parameters are valid.')

        


if __name__ == '__main__':

    ########################## get config
    config = Config()
    config.create_config('config.ini')
    
    cc = ContainerClassifier()
    
    
    ######################### training 

    cc.read_train_data(path=config.train_data_folder_path, 
                    resize_to=config.resize_to, 
                    class_dict=config.class_dict, 
                    use_labels=config.use_labels)

    cc.train(model=config.architecture, 
             epochs=config.epochs, 
             batch_size=config.batch_size, 
             optimizer=config.optimizer, 
             summary=config.print_model_summary)
    
    cc.save_model(name=config.model_name, 
                  path=config.save_to_dir)
    
    
    
    ######################### load pretrained
    cc.load_model(json_path=config.load_from_dir+'.json', 
                  weights_path=config.load_from_dir+'.h5')

    
    
    ######################### testing
    cc.read_test_data(path=config.test_data_folder_path, 
                      resize_to=config.resize_to, 
                      class_dict=config.class_dict, 
                      use_labels=config.use_labels)

    cc.test_model(output_dir=config.test_output_dir, 
                  top_k=config.top_k, 
                  plot_outputs=config.plot_outputs)

    