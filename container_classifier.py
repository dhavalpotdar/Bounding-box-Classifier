import os
from collections import Counter
import random
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
    '''
        This class reads in .jpg images of scanned text documents along with
        .txt files having the coordinates of bounding boxes along with labels.
        The labeling can be performed using annotation tools like LabelImg which
        can be found here: https://github.com/tzutalin/labelImg. The .txt files
        should be output as YOLO files from the LabelImg tool.

        This class further trains a Deep Learning model of choice and tests it.
    '''
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
        '''
            Method to denormalize the coordinates output by LabelImg.

            Args:
                size: list, [width, height] of image

                box_list: list, [x, y, w, h] from LabelImg

            Returns:
                list, denormalized [xmin, ymin, xmax, ymax]
        '''
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
                     use_labels=None, training=True):
        '''
        Function to implement the logic for creating dataset.

        Args:
            data_path_listoftuples: list of tuples -
                                    [(path/img_name.txt, path/img_name.jpg)]

            resize_to: tuple, shape tuple to resize training images to

            use_labels: list, labels to be used during training, None by default
                        (this option can be used to disregard labels with very
                         few training examples)

             training: bool, flag for training; true by default

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

                        WHITE = [255, 255, 255]

                        # copyMakeBorder(image, top, bottom, left, right, color):
                        # top, bottom, left, right are the number of pixels on
                        # respective side to add blank border to

                        # uncomment the following line to pad crops of bounding-
                        # boxes such that position of crop in the image is
                        # retained:

                        # image = cv2.copyMakeBorder(img_crop, ymin, height-ymax, 
                        #                             xmin, width-xmax,
                        #                             cv2.BORDER_CONSTANT,
                        #                             value=WHITE)
                        image = cv2.resize(img_crop, resize_to)

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

            Rest of the arguments are same as read_train_data() method.

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
                Pass as None if using pretrained model.
                'model_name' if training from scratch. One of:
                                                        - resnet101
                                                        - resnet50
                                                        - resnet152
                                                        - densenet121

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
                                                        input_shape=None,
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
        '''
            Saves trained model at given path
        '''
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
        '''
            Args:
                json_path: path to json file comprising the model architecture

                weights_path: path to h5 file having the weights of a
                                previously trained model

            Returns:
                None, attaches the loaded model as the class property
                'self.model' on which the self.test_model() method can be called

        '''
        # load json and create model
        json_file = open(json_path, 'r')

        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(weights_path)

        print("Loaded model from disk")

        self.model = loaded_model


    def test_model(self, output_dir='test_output'):

        '''
            Function to print classification metrics and output crops of
            classified bounding-boxes along with label written on them.

            Args:
                output_dir: '/path/to/folder', where crops are output along with
                            labeling
        '''
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

        count = 1
        for img, pred in zip(og_sizes, preds):
            cv2.putText(img, self.class_dict[inv_label_map[pred]],
                        (0, int(img.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            cv2.imwrite('./' + output_dir + '/' + str(count) + '.jpg', img)
            count += 1



# =========================== example use ==================================== #
if __name__ == '__main__':

    cc = ContainerClassifier()
    cc.read_train_data(path='./annotated_data', resize_to=(224, 224))

    training from scratch
    cc.train(model='resnet101', epochs=30, batch_size=4)
    cc.save_model(name='resnet101_nopretrained_30epochs_224x224')

    # uncomment if loading previously trained model
    # cc.load_model('./models/container_classifier_v1_0_0.json',
    #                 './models/container_classifier_v1_0_0.h5')

    # testing
    cc.read_test_data(path='./test', resize_to=(224, 224))

    cc.test_model(output_dir='test_output')
