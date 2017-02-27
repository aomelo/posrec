import numpy as np
import random
from keras.models import Sequential
from keras.layers import LSTM, Recurrent, Activation, Dropout, MaxPooling2D, Convolution2D, Dense, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.image import ImageDataGenerator, Iterator, K
from create_input import load_sequences_with_paths
from numpy.random import choice
from PIL import Image
from keras.utils.np_utils import to_categorical

import random


class PHImageDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory,
                            target_size=(90, 160), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=88,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='jpeg',
                            follow_links=False,
                            seq_len=10, train=True
                            ):
        pass


class PHImageSequenceIterator(Iterator):
    def __init__(self, directory, image_data_generator, target_size=(90, 160), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=88,
                 seq_len=10, train=True, test_size=0.2):
        self.seq_img_paths, self.seq_labels, instances, labels = load_sequences_with_paths(directory)
        idx_with_min_len = [i for i, seq in enumerate(self.seq_labels) if len(seq) >= seq_len]

        n_test_instances = int(test_size*len(idx_with_min_len))
        n_train_instances = len(idx_with_min_len) - n_test_instances
        train_test = [1]*n_train_instances + [0]*n_test_instances
        random.shuffle(train_test)
        if train:
            idx_with_min_len = [idx_with_min_len[i] for i,v in enumerate(train_test) if v == 1]
        else:
            idx_with_min_len = [idx_with_min_len[i] for i,v in enumerate(train_test) if v == 0]

        self.seq_img_paths = [self.seq_img_paths[i] for i in idx_with_min_len]
        self.seq_labels = [self.seq_labels[i] for i in idx_with_min_len]
        instances = [instances[i] for i in idx_with_min_len]

        n_instances = len(instances)
        self.n_labels = len(labels)
        super(PHImageSequenceIterator, self).__init__(n_instances, batch_size, shuffle, seed)
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.seq_len = seq_len
        self.target_size = target_size
        self.image_size = self.target_size + (3,)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        batch_x = np.zeros((current_batch_size, self.seq_len,) + self.image_size, dtype=K.floatx())
        batch_y = np.zeros((current_batch_size, self.seq_len, self.n_labels), dtype=int)
        for i, idx in enumerate(index_array):
            path_seq = self.seq_img_paths[idx]
            label_seq = self.seq_labels[idx]
            offset = random.randint(0, len(path_seq) - self.seq_len)
            for j, path in enumerate(path_seq[offset:offset + self.seq_len]):
                img = np.array(Image.open(path))
                #print(img.shape, batch_x[i, j].shape)
                batch_x[i, j] = img/255.0
                batch_y[i, j, label_seq[j + offset]] = 1
        return batch_x, batch_y


class PHImageSequenceDataGenerator(object):
    def flow_from_directory(self, directory,
                            target_size=(90, 160), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='jpeg',
                            follow_links=False,
                            seq_len=10, train=True
                            ):
        return PHImageSequenceIterator(directory, self, target_size=target_size, color_mode=color_mode, classes=classes,
                                       class_mode=color_mode, batch_size=batch_size, shuffle=shuffle, seed=seed,
                                       seq_len=seq_len, train=train)


class LRCN(object):
    def __init__(self, n_classes, seq_len=10, batch_size=64, n_epochs=100, optimizer="rmsprop", learning_rate=0.001):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.model = None
        self.n_classes = n_classes
        self.seq_len = seq_len

    def create_model(self):
        self.model = Sequential()
        self.model.add(TimeDistributed(Convolution2D(64, 3, 3), input_shape=(self.seq_len, 90, 160, 3)))
        self.model.add(TimeDistributed(Activation('relu')))
        self.model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
        self.model.add(TimeDistributed(Dropout(0.25)))
        #self.model.add(TimeDistributed(Convolution2D(64, 3, 3)))
        #self.model.add(TimeDistributed(Activation('relu')))
        #self.model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
        #self.model.add(TimeDistributed(Dropout(0.25)))
        self.model.add(TimeDistributed(Flatten()))

        self.model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
        self.model.add(TimeDistributed(Dense(self.n_classes, activation='softmax')))

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

    def fit_generator(self, train_generator, validation_generator=None, samples_per_epoch=2000, nb_epoch=50,
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
