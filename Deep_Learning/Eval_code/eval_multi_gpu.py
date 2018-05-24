# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime

import math
import time

import os
import sys

import random

import cv2

import importlib
import re

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf

from PIL import Image

FLAGS = tf.app.flags.FLAGS

out_filename = None
out_file_obj = None
check_dir = None
model = None

jpg_pattern = re.compile('\w*\.jpg')
caltech_basefolder = './Dataset/Train'
test_names = []
test_classes = []
all_images_dict = {}

IMAGE_SIZE = 100


def initiate_flags():
    global out_file_obj, IMAGE_SIZE

    all_dirs = [i for i in os.listdir(caltech_basefolder) if os.path.isdir(os.path.join(caltech_basefolder,i))]
    all_dirs.sort()
    for s_class,s_dir in enumerate(all_dirs):
        full_dir = os.path.join(caltech_basefolder,s_dir)
        all_files = [i for i in os.listdir(full_dir) if jpg_pattern.search(i) is not None]
        img_keys = [os.path.join(s_dir,i) for i in all_files]
        for img_key in img_keys:
            test_names.append(img_key)
            test_classes.append(s_class)
            all_images_dict[img_key] = {'class': s_class, 'data': cv2.imread(os.path.join(caltech_basefolder, img_key) ,cv2.IMREAD_COLOR)}    

    tf.app.flags.DEFINE_string('input_folder', './Dataset/Test',
                               """TFRecords training set filename""")

    tf.app.flags.DEFINE_string('eval_data', 'test',
                               """Either 'test' or 'train_eval'.""")

    tf.app.flags.DEFINE_string('checkpoint_dir', check_dir,
                               """Directory where to read model checkpoints.""")

    tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                                """How often to run the eval.""")
    tf.app.flags.DEFINE_integer('num_examples', len(test_classes),
                                """Number of examples to run.""")
    tf.app.flags.DEFINE_boolean('run_once', True,
                                """Whether to run eval only once.""")
    tf.app.flags.DEFINE_integer('num_gpus', 2,
                                """How many GPUs to use.""")
    tf.app.flags.DEFINE_boolean('log_device_placement', False,
                                """Whether to log device placement.""")

    out_file_obj = open(out_filename, 'a')

    IMAGE_SIZE = model.IMAGE_SIZE


def get_image_batch(index):


    full_sp = test_names
    full_cl = test_classes

    last_index = (index+FLAGS.batch_size) if (index+FLAGS.batch_size) < len(full_sp) else len(full_sp)

    extra_examples = []
    extra_classes = []
    if index+FLAGS.batch_size > len(full_sp):
        difference = index+FLAGS.batch_size - len(full_sp)
        sample = random.sample(xrange(len(full_sp)), difference)
        extra_examples = [full_sp[i] for i in sample]
        extra_classes = [full_cl[i] for i in sample]

    out_batch = full_sp[index:last_index] + extra_examples
    out_classes = full_cl[index:last_index] + extra_classes

    def fix_size(img):
        return cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE, 3) , interpolation = cv2.INTER_LANCZOS)

    return np.array([fix_size(all_images_dict[i]['data']) for i in out_batch]), np.array(out_classes)

def _normalize_input(raw_image_batch):
    def _normalize_single_image(image_decoded):
        final_image = tf.image.per_image_standardization(image_decoded)
        return final_image

    return tf.map_fn(_normalize_single_image, raw_image_batch)


def eval_once(saver, top_k_op, images, labels):
    """Run Eval once.

    Args:
      saver: Saver.
      top_k_op: Top K op.
    """
    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement)) as sess:

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint

            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split(
                '/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(
                math.ceil(FLAGS.num_examples / (FLAGS.batch_size * FLAGS.num_gpus)))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.num_gpus * FLAGS.batch_size
            step = 0
            eval_index = 0
            while step < num_iter and not coord.should_stop():

                in_dic = {}
                for x in xrange(FLAGS.num_gpus):
                    in_dic[images[x]], in_dic[labels[x]] = get_image_batch(
                        eval_index)
                    eval_index = (eval_index + FLAGS.batch_size) if (eval_index +
                                                                     FLAGS.batch_size) < num_examples else 0

                # evaluates some examples twice (very few), chose to ignore

                predictions = sess.run([top_k_op], in_dic)
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            # print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
            print('VALIDATING')
            print('%s' % (check_dir))
            print('%.4f' % (precision))
            out_file_obj.write('VALIDATING\n')
            out_file_obj.write('%s\n' % (check_dir))
            out_file_obj.write('%.4f\n' % (precision))
            out_file_obj.close()

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g, tf.device('/cpu:0'):
        # Get images and labels for CIFAR-10.
        eval_data = FLAGS.eval_data == 'test'

        images = []
        labels = []

        for i in xrange(FLAGS.num_gpus):
            images.append(_normalize_input(tf.placeholder(
                dtype=tf.float32, shape=[FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE])))
            labels.append(tf.placeholder(
                dtype=tf.int32, shape=[FLAGS.batch_size]))

        logits = [None] * FLAGS.num_gpus
        top_k_op_list = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (model.TOWER_NAME, i)) as scope:

                        logits[i] = model.inference(
                            images[i], should_summarize=False)
                        top_k_op_list.append(tf.nn.in_top_k(
                            logits[i], labels[i], 1))
                        tf.get_variable_scope().reuse_variables()

        top_k_op = top_k_op_list[0]
        for i in xrange(1, FLAGS.num_gpus):
            top_k_op = tf.concat([top_k_op, top_k_op_list[i]], 0)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            model.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            eval_once(saver, top_k_op, images, labels)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument

    global check_dir, out_filename, model

    check_dir = argv[1]

    out_filename = argv[2]

    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    model = importlib.import_module(argv[3])

    initiate_flags()

    evaluate()


if __name__ == '__main__':
    tf.app.run()
