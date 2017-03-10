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
from treelib import Tree, Node

pos_hier = Tree()

pos_hier.create_node("Any",0)
pos_hier.create_node("Intercourse", 1,parent=0)
pos_hier.create_node("Outercourse", 2,parent=0)
pos_hier.create_node("Noncourse", 3, 0)

pos_hier.create_node("Sex",4,parent=1)
pos_hier.create_node("Oral",5,parent=1)
pos_hier.create_node("Fingering", 10, parent=1)
pos_hier.create_node("Fisting", 11, parent=1)

pos_hier.create_node("Missionary", 6, parent=4)
pos_hier.create_node("DP", 7, parent=4)
pos_hier.create_node("Anal", 8, parent=4)
pos_hier.create_node("Strap On", 9, parent=4)
pos_hier.create_node("Cowgirl", 12, parent=4)
pos_hier.create_node("Reverse Cowgirl", 13, parent=12)

pos_hier.create_node("Titty Fucking",14,parent=2)
pos_hier.create_node("Handjob",15,parent=2)
pos_hier.create_node("Footjob",16,parent=2)
pos_hier.create_node("Masturbating",17, parent=2)

pos_hier.create_node("Smoking",18,parent=3)
pos_hier.create_node("Pissing",19,parent=3)
pos_hier.create_node("Squirting",20,parent=3)
pos_hier.create_node("Oil",21,parent=3)
pos_hier.create_node("Cumshot",22,parent=3)
pos_hier.create_node("Creampie",23,parent=3)

pos_hier.create_node("Blowjob",24,parent=5)
pos_hier.create_node("Pussy Licking",25,parent=5)
pos_hier.create_node("Ass Licking",26,parent=5)
pos_hier.create_node("Ass to Mouth",27,parent=5)



def get_labels_dict(labels, pos_hier):
    inv_labels_dict = defaultdict(lambda: set())
    labels_dict = defaultdict(lambda: set())

    labels_name_dict = {}
    inv_labels_name_dict = {}

    for node in pos_hier.all_nodes():
        labels_name_dict[node.tag] = node.identifier
        inv_labels_name_dict[node.identifier] = node.tag

    for i, label in enumerate(labels):
        node_id = labels_name_dict[label]
        node = pos_hier.get_node(node_id)
        labels_dict[node.tag].add(i)
        inv_labels_dict[i].add(node.tag)
        for sub_node in pos_hier.subtree(node_id).all_nodes():
            labels_dict[sub_node.tag].add(i)
            inv_labels_dict[i].add(sub_node.tag)

    for name, id_set in labels_dict.items():
        if len(id_set) > 1:
            most_specific = list(id_set)[np.argmax(np.array([pos_hier.level(id) for id in id_set]))]
            for id in id_set:
                if id != most_specific:
                    inv_labels_dict[id] = inv_labels_dict[id] - inv_labels_dict[most_specific]
        else:
            most_specific = list(id_set)[0]
        labels_dict[name] = most_specific

    return labels_dict, inv_labels_dict

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


def load_sequences_with_paths(input, labels_dict=None):
    dirs = [f for f in listdir(input) if isdir(join(input, f))]
    dirs.remove("None")
    dirs.remove("full")
    if labels_dict is None:
        labels_dict = {label:i for i,label in enumerate(dirs)}
    else:
        labels_dict = {k:v for k,v in labels_dict.items() if k in dirs}
    seq_labels = defaultdict(lambda: {})
    seq_feats = defaultdict(lambda: {})
    for label in dirs:
        if label in labels_dict:
            print("loading directory %s with label %d"%(label,labels_dict[label]))
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

    return seq_feats, seq_labels, instances, dirs

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
