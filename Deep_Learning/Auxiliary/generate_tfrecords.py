from __future__ import absolute_import
from __future__ import division

import argparse
import os
import sys
import random

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

t_sp, t_cl, v_sp, v_cl = pickle.load( open( pickle_filename, "rb" ) )

pickle_filename, file_extension = os.path.splitext( os.path.basename( pickle_filename) )

base_path = '.'
train_name = pickle_filename + '_train'
full_train_name = 'full_train'
eval_name = pickle_filename + '_eval'


train_filename = os.path.join(base_path, train_name + '.tfrecords')
full_train_filename = os.path.join(base_path, full_train_name + '.tfrecords')
eval_filename = os.path.join(base_path, eval_name + '.tfrecords')

train_writer = tf.python_io.TFRecordWriter(train_filename)
eval_writer = tf.python_io.TFRecordWriter(eval_filename)
full_train_writer = tf.python_io.TFRecordWriter(full_train_filename)

images_basefolder = '../Dataset'


full_train_filenames = [os.path.join(images_basefolder,tr) for tr in t_sp]

full_eval_filenames = [os.path.join(images_basefolder,tr) for tr in v_sp]



total_examples = len(full_train_filenames) + len(full_eval_filenames)

for fidx, filename in enumerate(full_train_filenames):

    img = cv2.imread(filename,cv2.IMREAD_COLOR)
    rows = img.shape[0]
    cols = img.shape[1]

    os.system('clear')
    print "%s: %f%%" % (pickle_filename, (float(fidx + 1)/ total_examples) * 100)
            
    img_example = cv2.resize(src=img, dsize= (IMAGE_SIZE, IMAGE_SIZE) , interpolation = cv2.INTER_LANCZOS4)

    image_raw = tf.compat.as_bytes(img_example.tostring())
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(IMAGE_SIZE),
        'width': _int64_feature(IMAGE_SIZE),
        'depth': _int64_feature(depth),          
        'label': _int64_feature(t_cl[fidx]),
        'image': _bytes_feature(image_raw)}))

    train_writer.write(example.SerializeToString())
    full_train_writer.write(example.SerializeToString())

train_writer.close()

for fidx, filename in enumerate(full_eval_filenames):

    img = cv2.imread(filename,cv2.IMREAD_COLOR)
    rows = img.shape[0]
    cols = img.shape[1]

    os.system('clear')
    print "%s: %f%%" % (pickle_filename, (float(fidx + 1 + len(full_train_filenames) )/ total_examples) * 100)
            
    img_example = cv2.resize(src=img, dsize= (IMAGE_SIZE, IMAGE_SIZE) , interpolation = cv2.INTER_LANCZOS4)

    image_raw = tf.compat.as_bytes(img_example.tostring())
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(IMAGE_SIZE),
        'width': _int64_feature(IMAGE_SIZE),
        'depth': _int64_feature(depth),          
        'label': _int64_feature(v_cl[fidx]),
        'image': _bytes_feature(image_raw)}))

    eval_writer.write(example.SerializeToString())
    full_train_writer.write(example.SerializeToString())

eval_writer.close()
full_train_writer.close()
