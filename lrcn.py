import numpy as np
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, Recurrent, Activation, Dropout, MaxPooling2D, Convolution2D, Dense, Flatten, Input
from keras.layers.recurrent import SimpleRNN
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.image import ImageDataGenerator, Iterator, K
from create_input import load_sequences_with_paths
from PIL import Image
import random
from keras.applications.vgg16 import VGG16


class PHImageSequenceIterator(Iterator):
    def __init__(self, directory, image_data_generator, target_size=(90, 160), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=88,
                 seq_len=10, train=True, test_size=0.2, labels_dict=None):
        self.seq_img_paths, self.seq_labels, instances, labels = load_sequences_with_paths(directory, labels_dict)
        if labels_dict is not None:
            labels = list(range(max(labels_dict.values())+1))
        idx_with_min_len = [i for i, seq in enumerate(self.seq_labels) if len(seq) >= seq_len]
        print("%d out of %d sequences have length >= %d" % (len(idx_with_min_len), len(self.seq_labels), seq_len))

        n_test_instances = int(test_size * len(idx_with_min_len))
        n_train_instances = len(idx_with_min_len) - n_test_instances
        train_test = [1] * n_train_instances + [0] * n_test_instances
        random.shuffle(train_test)
        if train:
            idx_with_min_len = [idx_with_min_len[i] for i, v in enumerate(train_test) if v == 1]
        else:
            idx_with_min_len = [idx_with_min_len[i] for i, v in enumerate(train_test) if v == 0]

        self.seq_img_paths = [self.seq_img_paths[i] for i in idx_with_min_len]
        self.seq_labels = [self.seq_labels[i] for i in idx_with_min_len]
        instances = [instances[i] for i in idx_with_min_len]

        sample = [np.array(Image.open(paths[random.randint(0, len(paths) - 1)])) / 255.0 for paths in
                  self.seq_img_paths]
        sample = [img for img in sample if len(img.shape) == 3]
        sample = np.array(sample)
        print(sample.shape)
        image_data_generator.fit(sample)

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
                # print(img.shape, batch_x[i, j].shape)
                batch_x[i, j] = img / 255.0
                batch_y[i, j, label_seq[j + offset]] = 1
        return batch_x, batch_y


class PHImageSequenceDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory,
                            target_size=(90, 160), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='jpeg',
                            follow_links=False,
                            seq_len=10, train=True, labels_dict=None,
                            ):
        return PHImageSequenceIterator(directory, self, target_size=target_size, color_mode=color_mode, classes=classes,
                                       class_mode=color_mode, batch_size=batch_size, shuffle=shuffle, seed=seed,
                                       seq_len=seq_len, train=train, labels_dict=labels_dict)


class LRCN(object):
    def __init__(self, n_classes, nb_filters=[128,96,64], seq_len=10, batch_size=64, n_epochs=100, optimizer="rmsprop", dropout=0.25, learning_rate=0.001, saved_model=None, activation="sigmoid"):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.model = None
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.saved_model = saved_model
        self.nb_filters = nb_filters
        self.activation = activation
        self.dropout = dropout
        self.create_model()


    def create_model(self):
        if self.saved_model is not None:
            self.model = load_model(self.saved_model)
        else:
            input_cnn = Input(shape=(90, 160, 3))
            input_seq = Input(shape=(self.seq_len, 90, 160, 3))

            x = Convolution2D(self.nb_filters[0], 3, 3, border_mode="same")(input_cnn)
            #x = Activation(self.activation)(x)
            x = MaxPooling2D((2, 2))(x)
            #if self.dropout > 0:
            #    x = Dropout(self.dropout)(x)
            for nb_filter in self.nb_filters[1:]:
                x = Convolution2D(nb_filter, 3, 3, border_mode="same")(x)
                #x = Activation(self.activation)(x)
                x = MaxPooling2D((2, 2))(x)
                #if self.dropout > 0:
                #    x = Dropout(self.dropout)(x)
            x = Flatten()(x)

            cnn_model = Model(input=input_cnn,output=x)

            self.model = Sequential()

            self.model.add(TimeDistributed(cnn_model,input_shape=(self.seq_len, 90, 160, 3)))
            self.model.add(SimpleRNN(self.n_classes, return_sequences=True, activation='softmax'))
            #x = Dense(self.n_classes, activation='softmax')(x)


            self.model.compile(loss='categorical_crossentropy',
                               optimizer=self.optimizer,
                               metrics=['accuracy', 'fmeasure', 'categorical_accuracy'],
                               learning_rate=self.learning_rate)

            self.model.summary()

            # self.model = Sequential()
            # self.model.add(TimeDistributed(Convolution2D(64, 3, 3), input_shape=(self.seq_len, 90, 160, 3)))
            # self.model.add(TimeDistributed(Activation('relu')))
            # self.model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
            # self.model.add(TimeDistributed(Dropout(0.25)))
            # self.model.add(TimeDistributed(Convolution2D(64, 3, 3)))
            # self.model.add(TimeDistributed(Activation('relu')))
            # self.model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
            # self.model.add(TimeDistributed(Dropout(0.25)))
            # self.model.add(TimeDistributed(Flatten()))
            #
            # self.model.add(SimpleRNN(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
            # self.model.add(TimeDistributed(Dense(self.n_classes, activation='softmax')))
            #
            # self.model.compile(loss='categorical_crossentropy',
            #                    optimizer=self.optimizer,
            #                    metrics=['accuracy', 'fmeasure', 'categorical_accuracy'],
            #                    learning_rate=self.learning_rate)

    def fit(self, X, Y):

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


class LRCNVGG16(LRCN):
    def create_model(self):
        if self.saved_model is not None:
            self.model = load_model(self.saved_model)
        else:
            vgg_model = Sequential()
            vgg_model.add(VGG16(weights='imagenet', include_top=False, input_shape=(90, 160, 3)))
            vgg_model.add(Flatten())
            self.model = Sequential()
            self.model.add(TimeDistributed(vgg_model, input_shape=(self.seq_len, 90, 160, 3)))
            self.model.add(SimpleRNN(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
            self.model.add(TimeDistributed(Dense(self.n_classes, activation='softmax')))

            self.model.compile(loss='categorical_crossentropy',
                               optimizer=self.optimizer,
                               metrics=['accuracy', 'fmeasure', 'categorical_accuracy'],
                               learning_rate=self.learning_rate)
