from argparse import ArgumentParser
from create_input import load_sequences, to_no_seq
from lrcn import LRCN, PHImageSequenceDataGenerator, PHImageSequenceIterator


if __name__ == '__main__':
    parser = ArgumentParser("Transforms thumbnails folder into input sequences")
    parser.add_argument("input", type=str, default=None, help="path to folder containing labels folder")
    parser.add_argument("-m", "--model", type=str, default=None, help="path to model checkpointr")

    args = parser.parse_args()

    # (X, Y), (X_test, Y_test) = cifar10.load_data(dirname=".")
    # print(X.shape,Y.shape,X_test.shape,Y_test.shape)

    data_gen = PHImageSequenceDataGenerator()
    train_generator = data_gen.flow_from_directory(args.input,batch_size=64,train=True)
    test_generator = data_gen.flow_from_directory(args.input,batch_size=64,train=False)
    #train_generator = PHImageSequenceDataGenerator(args.input,batch_size=128)
    n_classes = train_generator.n_labels
    batch_size = train_generator.batch_size

    clf = LRCN(n_classes=n_classes,batch_size=batch_size)
    clf.fit_generator(train_generator=train_generator)
