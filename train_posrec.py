from argparse import ArgumentParser
from create_input import load_sequences, to_no_seq, pos_hier
from lrcn import LRCN, PHImageSequenceDataGenerator, PHImageSequenceIterator, LRCNVGG16
from treelib import Tree, Node
from collections import defaultdict
import numpy as np

if __name__ == '__main__':
    parser = ArgumentParser("Transforms thumbnails folder into input sequences")
    parser.add_argument("input", type=str, default=None, help="path to folder containing labels folder")
    parser.add_argument("-m", "--model", type=str, default="lrcn", help="path to model checkpointr")

    labels = ["Sex","Oral","Outercourse","Noncourse"]
    inv_labels_dict = defaultdict(lambda : set())
    labels_dict = defaultdict(lambda : set())

    labels_name_dict = {}
    inv_labels_name_dict = {}

    for node in pos_hier.all_nodes():
        labels_name_dict[node.tag] = node.identifier
        inv_labels_name_dict[node.identifier] = node.tag

    for i,label in enumerate(labels):
        node_id = labels_name_dict[label]
        node = pos_hier.get_node(node_id)
        labels_dict[node.tag].add(i)
        inv_labels_dict[i].add(node.tag)
        for sub_node in pos_hier.subtree(node_id).all_nodes():
            labels_dict[sub_node.tag].add(i)
            inv_labels_dict[i].add(sub_node.tag)

    for name, id_set in labels_dict.items():
        if len(id_set)>1:
            most_specific = list(id_set)[np.argmax(np.array([pos_hier.level(id) for id in id_set]))]
            for id in id_set:
                if id != most_specific:
                    inv_labels_dict[id] = inv_labels_dict[id] - inv_labels_dict[most_specific]
        else:
            most_specific = list(id_set)[0]
        labels_dict[name] = most_specific


    args = parser.parse_args()

    # (X, Y), (X_test, Y_test) = cifar10.load_data(dirname=".")
    # print(X.shape,Y.shape,X_test.shape,Y_test.shape)
    print({l:i for i,l in enumerate(labels)})

    data_gen = PHImageSequenceDataGenerator()
    train_generator = data_gen.flow_from_directory(args.input,batch_size=64,train=True, labels_dict=labels_dict)
    test_generator = data_gen.flow_from_directory(args.input,batch_size=64,train=False, labels_dict=labels_dict)
    #train_generator = PHImageSequenceDataGenerator(args.input,batch_size=128)
    n_classes = train_generator.n_labels
    batch_size = train_generator.batch_size

    if args.model=="lrcn":
        clf = LRCN(n_classes=n_classes,batch_size=batch_size)
    elif args.model=="lrcnvgg16":
        clf = LRCNVGG16(n_classes=n_classes, batch_size=batch_size)
    else:
        raise("Model %s is not supported"%args.model)
    clf.fit_generator(train_generator=train_generator)
