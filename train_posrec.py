from argparse import ArgumentParser
from create_input import pos_hier, get_labels_dict
from lrcn import LRCN, PHImageSequenceDataGenerator, LRCNVGG16
from cnn import CNN, CNNVGG16, PHImageDataGenerator

from collections import defaultdict
import numpy as np

if __name__ == '__main__':
    parser = ArgumentParser("Transforms thumbnails folder into input sequences")
    parser.add_argument("input", type=str, default=None, help="path to folder containing labels folder")
    parser.add_argument("-m", "--model", type=str, default="lrcn", help="name of model to be trained [cnn,lrcn,cnnvgg16,lrcnvgg16]")
    parser.add_argument("-sm", "--saved-model", type=str, default=None, help="path to saved model")
    parser.add_argument("-ls", "--layers", type=int, nargs="+", default=[64,32], help="number of filters on each layer")
    parser.add_argument("-ne", "--n-epochs", type=int, default=50, help="number of epochs")

    labels = ["Sex","Oral","Outercourse","Noncourse"]
    labels_dict, _ = get_labels_dict(labels,pos_hier)

    n_classes = len(labels)
    batch_size = 64

    args = parser.parse_args()

    # (X, Y), (X_test, Y_test) = cifar10.load_data(dirname=".")
    # print(X.shape,Y.shape,X_test.shape,Y_test.shape)
    print({l:i for i,l in enumerate(labels)})


    print(labels_dict)

    if args.model.startswith("lrcn"):
        if args.model == "lrcn":
            clf = LRCN(n_classes=n_classes, nb_filters=args.layers, batch_size=batch_size, saved_model=args.saved_model)
        elif args.model == "lrcnvgg16":
            clf = LRCNVGG16(n_classes=n_classes, nb_filters=args.layers, batch_size=batch_size,
                            saved_model=args.saved_model)
        else:
            raise ("Model %s is not supported" % args.model)
        data_gen = PHImageSequenceDataGenerator()
        train_generator = data_gen.flow_from_directory(args.input, batch_size=batch_size, train=True, labels_dict=labels_dict)
        test_generator = data_gen.flow_from_directory(args.input, batch_size=batch_size, train=False, labels_dict=labels_dict)
    elif args.model.startswith("cnn"):
        if args.model == "cnn":
            clf = CNN(n_classes=n_classes, nb_filters=args.layers, batch_size=batch_size, saved_model=args.saved_model)
        elif args.model == "cnnvgg16":
            clf = CNNVGG16(n_classes=n_classes, nb_filters=args.layers, batch_size=batch_size, saved_model=args.saved_model)
        else:
            raise ("Model %s is not supported" % args.model)
        data_gen = PHImageDataGenerator()
        train_generator = data_gen.flow_from_directory(args.input, batch_size=batch_size, train=True, labels_dict=labels_dict)
        test_generator = data_gen.flow_from_directory(args.input, batch_size=batch_size, train=False, labels_dict=labels_dict)

    else:
        raise ("Model %s is not supported" % args.model)

    clf.fit_generator(train_generator=train_generator)
    clf.model.save(args.model+str(args.layers)+".h5")
