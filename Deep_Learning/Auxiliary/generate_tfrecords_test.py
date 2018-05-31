from __future__ import absolute_import
from __future__ import division

import argparse
import os
import sys
import random

import re

import pickle

import cv2
import numpy as np

import tensorflow as tf



def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


IMAGE_SIZE = 100

depth = 3

pickle_filename = sys.argv[1]

jpg_pattern = re.compile('\w*\.jpg')

base_path = '.'
test_name = 'test'

test_filename = os.path.join(base_path, test_name + '.tfrecords')

test_writer = tf.python_io.TFRecordWriter(test_filename)

images_basefolder = '../Dataset/'

test_sp, test_classes = pickle.load( open( pickle_filename, "rb" ) )

full_test_filenames = [os.path.join(images_basefolder, i) for i in test_sp]



total_examples = len(full_test_filenames)

for fidx, filename in enumerate(full_test_filenames):

    img = cv2.imread(filename,cv2.IMREAD_COLOR)
    rows = img.shape[0]
    cols = img.shape[1]

    os.system('clear')
    print "Test: %f%%" % ( (float(fidx + 1)/ total_examples) * 100)
            
    img_example = cv2.resize(src=img, dsize= (IMAGE_SIZE, IMAGE_SIZE) , interpolation = cv2.INTER_LANCZOS4)

    image_raw = tf.compat.as_bytes(img_example.tostring())
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(IMAGE_SIZE),
        'width': _int64_feature(IMAGE_SIZE),
        'depth': _int64_feature(depth),          
        'label': _int64_feature(test_classes[fidx]),
        'image': _bytes_feature(image_raw)}))

    test_writer.write(example.SerializeToString())

test_writer.close()


