from argparse import ArgumentParser
from os.path import join, isfile, isdir
from os import listdir
import numpy as np
import re
from collections import defaultdict
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os


def to_no_seq(seq_feats, seq_labels, test_size=0.1):
    X, Y, X_test, Y_test = [], [], [], []

    n_instances = len(seq_labels)
    test_i = np.random.choice(n_instances, int(test_size * n_instances), replace=False)

    for i, seq in enumerate(seq_feats):
        if i in test_i:
            X_test += seq
        else:
            X = X + seq
    for i, seq in enumerate(seq_labels):
        if i in test_i:
            Y_test += seq
        else:
            Y += seq

    return np.array(X), np.array(Y), np.array(X_test), np.array(Y_test)

def get_stats(seqs):
    lens = [len(x) for x in seqs]
    print("lengths in [%d,%d]"%(min(lens),max(lens)))
    plt.hist(lens)
    plt.show()


def load_sequences(input):
    seq_feats,seq_labels,labels,istances = load_sequences_with_paths(input)
    seq_feats = [[np.array(Image.open(path)) for path in seq] for seq in seq_feats]
    return seq_feats,seq_labels,labels,instances


def clean_faulty_images(seq_img_paths):
    for seq in seq_img_paths:
        for path in seq:
            img = Image.open(path)
            width, height = img.size
            if (width,height) != (160,90) or im.mode!="RGB":
                print("delete %s mode %s"%(path,img.mode))
                os.remove(path)



def load_sequences_with_paths(input):
    labels = [f for f in listdir(input) if isdir(join(input, f))]
    labels.remove("None")
    labels.remove("full")
    labels_dict = {l: i for i, l in enumerate(labels)}
    seq_labels = defaultdict(lambda: {})
    seq_feats = defaultdict(lambda: {})
    for label in labels:
        print("loading label "+label)
        count = 0
        for tn in listdir(join(input, label)):
            if isfile(join(input, label, tn)):
                pattern = "(.+)\-frame([0-9]+)(\.jpg)?"
                m = re.search(pattern, tn)
                if m and m.groups() and len(m.groups())>=2:
                    id = m.group(1)
                    frame = int(m.group(2))
                    img_path = join(input, label, tn)
                    #img = Image.open(img_path)
                    seq_labels[id][frame] = labels_dict[label]
                    seq_feats[id][frame] = img_path
                    count += 1
        print(str(count)+" images loaded")

    instances = list(seq_labels.keys())

    seq_labels = [list(v.values()) for k, v in seq_labels.items()]
    seq_feats = [list(v.values()) for k, v in seq_feats.items()]

    return seq_feats, seq_labels, instances, labels

def load_sequences(input):
    labels = [f for f in listdir(input) if isdir(join(input, f))]
    labels.remove("None")
    labels.remove("full")
    labels_dict = {l: i for i, l in enumerate(labels)}
    seq_labels = defaultdict(lambda: {})
    seq_feats = defaultdict(lambda: {})
    for label in labels:
        print("loading label "+label)
        count = 0
        for tn in listdir(join(input, label)):
            if isfile(join(input, label, tn)):
                pattern = "(.+)\-frame([0-9]+)(\.jpg)?"
                m = re.search(pattern, tn)
                if m and m.groups() and len(m.groups())>=2:
                    id = m.group(1)
                    frame = int(m.group(2))
                    img = Image.open(join(input, label, tn))
                    seq_labels[id][frame] = labels_dict[label]
                    seq_feats[id][frame] = np.array(img)
                    count += 1
        print(str(count)+" images loaded")

    instances = list(seq_labels.keys())

    seq_labels = [list(v.values()) for k, v in seq_labels.items()]
    seq_feats = [list(v.values()) for k, v in seq_feats.items()]

    return seq_feats, seq_labels, instances, labels


if __name__ == '__main__':
    parser = ArgumentParser("Transforms thumbnails folder into input sequences")
    parser.add_argument("input", type=str, default=None, help="path to folder containing labels folder")
    parser.add_argument("output", type=str, default=None, help="path to npy with the data")

    args = parser.parse_args()

    seq_feats, seq_labels, instances, labels = load_sequences_with_paths(args.input)
    np.save(args.output, (seq_feats, seq_labels))
    meta_file = args.output.replace(".npy", "-meta.npy")
    meta_data = {"labels": labels, "instances": instances}
    np.save(meta_file, meta_data)
