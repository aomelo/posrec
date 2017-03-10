from PIL import Image
import os
from argparse import ArgumentParser
from create_input import load_sequences_with_paths

def clean_faulty_images(seq_img_paths):
    for seq in seq_img_paths:
        for path in seq:
            img = Image.open(path)
            width, height = img.size
            if (width,height) != (160,90) or img.mode!="RGB":
                print("delete %s mode %s"%(path,img.mode))
                os.remove(path)

if __name__ == '__main__':
    parser = ArgumentParser("Cleans the data set to remove images which do not fit (160,90,3) shape")
    parser.add_argument("input",type=str,default=None,help="data directory path")
    args = parser.parse_args()

    seq_img_paths, _, _, _ = load_sequences_with_paths(args.input)
    clean_faulty_images(seq_img_paths)