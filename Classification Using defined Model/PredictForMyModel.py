import cv2
import os
import sys
import glob
import argparse
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.models import model_from_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    return args


def get_files(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*'))
    elif path.find('*') > 0:
        files = glob.glob(path)
    else:
        files = [path]

    files = [f for f in files if f.endswith('JPG') or f.endswith('jpg') or f.endswith('PNG') or f.endswith('png') or f.endswith('JPEG') or f.endswith('jpeg') or f.endswith('TIFF')]

    if not len(files):
        sys.exit('No images found by the given path!')

    return files


if __name__ == '__main__':
    args = parse_args()
    files = get_files(args.path)
    cls_list = ['Hatchback', 'SUV', 'Sedan']

    yaml_file = open('model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    loaded_model.load_weights("model.h5")
    loaded_model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])




    for f in files:
        image1 = cv2.imread(f)
        image1 = cv2.resize(image1,(256,256))
        cv2.imwrite('test.jpg', image1)
        img = image.load_img('test.jpg')
        
        if img is None:
            continue
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pred = loaded_model.predict(x)[0]
        top_inds = pred.argsort()[::-1][:5]
        print(f)
        for i in top_inds:
            print('    {:.2f}  {}'.format(pred[i], cls_list[i]))
