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

import os,sys

import importlib

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf

from PIL import Image

FLAGS = tf.app.flags.FLAGS

out_filename = None
out_file_obj = None
tfrecords_filename = None
input_dir = None
model = None


INITIAL_IMAGE_SIZE = 32
IMAGE_SIZE = 32

def initiate_flags():
  global out_file_obj, INITIAL_IMAGE_SIZE, IMAGE_SIZE

  tf.app.flags.DEFINE_string('input_filename', tfrecords_filename,
                            """TFRecords training set filename""")  

  tf.app.flags.DEFINE_string('eval_data', 'test',
                            """Either 'test' or 'train_eval'.""")

  tf.app.flags.DEFINE_string('checkpoint_dir', input_dir,
                            """Directory where to read model checkpoints.""")

  tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                              """How often to run the eval.""")
  tf.app.flags.DEFINE_integer('num_examples', 10000,
                              """Number of examples to run.""")
  tf.app.flags.DEFINE_boolean('run_once', True,
                          """Whether to run eval only once.""")
  tf.app.flags.DEFINE_integer('num_gpus', 2,
                              """How many GPUs to use.""")
  tf.app.flags.DEFINE_boolean('log_device_placement', False,
                              """Whether to log device placement.""")

  out_file_obj  = open(out_filename, 'a')

  INITIAL_IMAGE_SIZE = model.IMAGE_SIZE
  IMAGE_SIZE = model.IMAGE_SIZE


def eval_once(saver, top_k_op, iterator):
  """Run Eval once.

  Args:
    saver: Saver.
    top_k_op: Top K op.
  """
  with tf.Session(config=tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=FLAGS.log_device_placement)) as sess:

    sess.run(iterator.initializer)

    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint

      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
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
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      # print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
      print('VALIDATING' )
      print('%s' % (input_dir))
      print('%s: %.4f' % (tfrecords_filename, precision))
      out_file_obj.write('VALIDATING\n')
      out_file_obj.write('%s\n' % (input_dir))
      out_file_obj.write('%s: %.4f\n' % (tfrecords_filename, precision))
      out_file_obj.close()

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


##############################################


def _parse_function(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, features)
    vector_decoded = tf.reshape(tf.decode_raw(parsed_features["image"],tf.uint8) , [INITIAL_IMAGE_SIZE*INITIAL_IMAGE_SIZE*3] )
    red_decoded = tf.slice(vector_decoded,[0],[INITIAL_IMAGE_SIZE*INITIAL_IMAGE_SIZE])
    green_decoded = tf.slice(vector_decoded,[INITIAL_IMAGE_SIZE*INITIAL_IMAGE_SIZE],[INITIAL_IMAGE_SIZE*INITIAL_IMAGE_SIZE])
    blue_decoded = tf.slice(vector_decoded,[2*INITIAL_IMAGE_SIZE*INITIAL_IMAGE_SIZE],[INITIAL_IMAGE_SIZE*INITIAL_IMAGE_SIZE])

    red_decoded = tf.reshape(red_decoded,[INITIAL_IMAGE_SIZE,INITIAL_IMAGE_SIZE,1] )
    green_decoded = tf.reshape(green_decoded,[INITIAL_IMAGE_SIZE,INITIAL_IMAGE_SIZE,1] )
    blue_decoded = tf.reshape(blue_decoded,[INITIAL_IMAGE_SIZE,INITIAL_IMAGE_SIZE,1] )

    image_decoded = tf.concat([red_decoded, green_decoded, blue_decoded],2)
    image_decoded = tf.cast(image_decoded, tf.float32)
    
    final_image = tf.image.per_image_standardization(image_decoded)
    return final_image, tf.cast(parsed_features["label"], tf.int32)


##############################################

def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g, tf.device('/cpu:0'):
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'

    dataset = tf.data.TFRecordDataset(FLAGS.input_filename)
    dataset = dataset.map(_parse_function)  # Parse the record into tensors.
    dataset = dataset.repeat()  # Repeat the input indefinitely. 
    dataset = dataset.batch(FLAGS.batch_size)               
    iterator = dataset.make_initializable_iterator()

    top_k_op_list = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in xrange(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (model.TOWER_NAME, i)) as scope:

                    images, labels = iterator.get_next()

                    logits = model.inference(
                        images, should_summarize=False)
                    top_k_op_list.append(tf.nn.in_top_k(
                        logits, labels, 1))

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
      eval_once(saver, top_k_op, iterator)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument

  global input_dir, tfrecords_filename, out_filename, model

  input_dir = argv[1]
  tfrecords_filename = argv[2]
  out_filename = argv[3]

  sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  model = importlib.import_module(argv[4])

  initiate_flags()

  evaluate()


if __name__ == '__main__':
  tf.app.run()
