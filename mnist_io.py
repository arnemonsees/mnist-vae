import gzip
import matplotlib.pyplot as plt
import numpy as np
import os

img_size = 28 # MNIST image dimension: 28x28
nImgs_train = 60000
nImgs_test = 10000

#path_data = os.path.abspath('data/mnist')
path_data = 'data/mnist'
file_imgs_train = path_data + '/train-images-idx3-ubyte.gz'
file_labels_train = path_data + '/train-labels-idx1-ubyte.gz'
file_imgs_test = path_data + '/t10k-images-idx3-ubyte.gz'
file_labels_test = path_data + '/t10k-labels-idx3-ubyte.gz'

def load_images(imgs_file, nImgs):
    '''
    # imgs_file: string pointing to location of images
    # nImages: total number of images in imgs_file
    '''
    f = gzip.open(imgs_file, 'r')
    f.read(16) # skip first 16 entries
    buffer = f.read(img_size**2 * nImgs)
    f.close()
    imgs = np.frombuffer(buffer, dtype=np.uint8).reshape(nImgs, img_size, img_size)
    return imgs

def load_labels(labels_file, nLabels):
    '''
    # labels_file: string pointing to location of labels
    # nLabels: total number of labels in labels_file
    '''
    f = gzip.open(labels_file, 'r')
    f.read(8) # skip first 8 entries
    buf = f.read(nLabels)
    f.close()
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels
