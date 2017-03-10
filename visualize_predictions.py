from argparse import ArgumentParser
from keras.models import load_model
from create_input import pos_hier, get_labels_dict
from cnn import PHImageDataGenerator
from lrcn import PHImageSequenceDataGenerator
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("input", type=str, help="path to data directory")
    parser.add_argument("saved_model", type=str, default=None, help="path to saved model")
    parser.add_argument("-nb","--n-batches", type=int, default=1, help="number of batches (64) to be displayed")

    args = parser.parse_args()

    print("loading model %s"%args.saved_model)
    m = load_model(args.saved_model)

    labels = ["Sex", "Oral", "Outercourse", "Noncourse"]
    labels_dict, _ = get_labels_dict(labels, pos_hier)

    print(labels)
    print(labels_dict)


    if args.saved_model.startswith("lrcn"):
        data_gen = PHImageSequenceDataGenerator()
    if args.saved_model.startswith("cnn"):
        data_gen = PHImageDataGenerator()

    test_generator = data_gen.flow_from_directory(args.input, batch_size=64, train=False, labels_dict=labels_dict)

    for i in range(args.n_batches):
        batch_x, batch_y = test_generator.next()
        preds = m.predict(batch_x)

        fig = plt.figure()
        for i in range(batch_x.shape[0]):
            instance = batch_x[i]
            pred = preds[i]
            y = batch_y[i]
            #if len(batch_x.shape) == 3:
                # img = Image.fromarray(instance,"RGB")

            pred = np.argmax(np.array(pred))
            y = np.argmax(np.array(y))
            a = fig.add_subplot(8, 8, i+1)
            plt.imshow(instance, interpolation="nearest")
            plt.axis("off")
            #plt.axes().get_xaxis().set_ticks([])
            #plt.axes().get_yaxis().set_ticks([])
            #a.tick_params(axis="both", which="both", bottom="off",top="off",labelbottom="off")

            a.set_title("%s(%s)"%(labels[pred] if pred<len(labels) else str(pred),labels[y] if y < len(labels) else str(y)), fontsize=8)


        plt.show()
