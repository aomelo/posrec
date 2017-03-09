import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Recurrent, Activation, Dropout, MaxPooling2D, Convolution2D, Dense, Flatten
from keras.layers.recurrent import SimpleRNN
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.image import ImageDataGenerator, Iterator, K
from create_input import load_sequences_with_paths
from PIL import Image
import random
from keras.applications.vgg16 import VGG16


class PHImageIterator(Iterator):
    def __init__(self, directory, image_data_generator, target_size=(90, 160), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=88,
                 train=True, test_size=0.2, labels_dict=None):
        seq_img_paths, seq_labels, instances, labels_list = load_sequences_with_paths(directory, labels_dict)
        n_test_instances = int(test_size*len(seq_labels))
        n_train_instances = len(seq_labels) - n_test_instances
        train_test = [1]*n_train_instances + [0]*n_test_instances
        random.shuffle(train_test)
        if train:
            idx = [i for i,v in enumerate(train_test) if v == 1]
        else:
            idx = [i for i,v in enumerate(train_test) if v == 0]

        seq_img_paths = [seq_img_paths[i] for i in idx]
        seq_labels = [seq_labels[i] for i in idx]


        sample = [np.array(Image.open(paths[random.randint(0, len(paths) - 1)])) / 255.0 for paths in seq_img_paths]
        sample = [img for img in sample if len(img.shape)==3]
        sample = np.array(sample)
        print(sample.shape)
        image_data_generator.fit(sample)

        self.paths, self.labels = [],[]
        for i in range(len(seq_labels)):
            self.paths += seq_img_paths[i]
            self.labels += seq_labels[i]

        instances = [instances[i] for i in idx]

        n_instances = len(instances)
        self.n_labels = len(labels_list)
        super(PHImageIterator, self).__init__(n_instances, batch_size, shuffle, seed)
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = target_size
        self.image_size = self.target_size + (3,)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        batch_x = np.zeros((current_batch_size, ) + self.image_size, dtype=K.floatx())
        batch_y = np.zeros((current_batch_size, self.n_labels), dtype=int)
        for i, idx in enumerate(index_array):
            path = self.paths[idx]
            label = self.labels[idx]
            img = np.array(Image.open(path))
            #print(img.shape, batch_x[i, j].shape)
            batch_x[i,] = img/255.0
            batch_y[i, label] = 1
        return batch_x, batch_y

class PHImageDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory,
                            target_size=(90, 160), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=88,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='jpeg',
                            follow_links=False,
                            train=True, labels_dict=None
                            ):
        return PHImageIterator(directory, self, target_size=target_size, color_mode=color_mode, classes=classes,
                                       class_mode=color_mode, batch_size=batch_size, shuffle=shuffle, seed=seed,
                                       train=train, labels_dict=labels_dict)


class CNN(object):
    def __init__(self, n_classes, batch_size=64, n_epochs=100, optimizer="rmsprop", learning_rate=0.001):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.model = None
        self.n_classes = n_classes

    def create_model(self):
        self.model = Sequential()
        self.model.add(Convolution2D(64, 3, 3, input_shape=(90, 160, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())

        self.model.add(Dense(64))  # returns a sequence of vectors of dimension 32
        self.model.add(Dense(self.n_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.optimizer,
                           metrics=['accuracy'],
                           learning_rate=self.learning_rate)

    def fit(self, X, Y):
        self.create_model()
        self.model.fit(X, Y,
                       batch_size=self.batch_size,
                       nb_epoch=self.n_epochs,
                       shuffle=True)

    def fit_generator(self, train_generator, validation_generator=None, samples_per_epoch=4096, nb_epoch=50,
                      nb_val_samples=400):
        self.create_model()
        self.model.fit_generator(
            train_generator,
            samples_per_epoch=samples_per_epoch,
            nb_epoch=nb_epoch,
            validation_data=validation_generator,
            nb_val_samples=nb_val_samples)

    def predict_proba(self, X):
        self.model.predict_proba(X)

    def predict(self, X):
        self.model.predict_proba(X)


class CNNVGG16(CNN):
    def create_model(self):

        self.model = Sequential()
        self.model.add(VGG16(weights='imagenet', include_top=False, input_shape=(90,160,3)))
        self.model.add(Flatten())
        self.model.add(Dense(64))  # returns a sequence of vectors of dimension 32
        self.model.add(Dense(self.n_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.optimizer,
                           metrics=['accuracy'],
                           learning_rate=self.learning_rate)