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

"""A binary to train CIFAR-10 using multiple GPUs with synchronous updates.

Accuracy:
cifar10_multi_gpu_train.py achieves ~86% accuracy after 100K steps (256
epochs of data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
--------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
2 Tesla K20m  | 0.13-0.20              | ~84% at 30K steps  (2.5 hours)
3 Tesla K20m  | 0.13-0.18              | ~84% at 30K steps
4 Tesla K20m  | ~0.10                  | ~84% at 30K steps

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import re
import time
import importlib

import pickle

from distutils.dir_util import copy_tree

from PIL import Image

import math

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

NUM_EPOCHS_PER_DECAY = 400.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.002     # Initial learning rate.

calculate_rates = True
FLAGS = tf.app.flags.FLAGS

model = None
model_name = None

pickle_filename = None
t_sp = None
t_cl = None
v_sp = None
v_cl = None

split_index = None
split_size = None

IMAGE_SIZE = 100

def initiate_flags():
    global IMAGE_SIZE


    tf.app.flags.DEFINE_string('train_dir', './Current_training/current',
                            """Directory where to write event logs """
                            """and checkpoint.""")


    tf.app.flags.DEFINE_string('input_filename', './TFRecords_files/Splits_%s/split_%02d_train.tfrecords'%(split_size,split_index)  ,
                            """TFRecords training set filename""")

    tf.app.flags.DEFINE_string('eval_filename', './TFRecords_files/Splits_%s/split_%02d_eval.tfrecords'%(split_size,split_index) ,
                            """TFRecords eval set filename""")


    tf.app.flags.DEFINE_integer('num_examples', len(t_sp),
                                """Number of examples to run.""")

    tf.app.flags.DEFINE_integer('eval_num_examples', len(v_sp),
                                """Number of eval examples to run.""")

    tf.app.flags.DEFINE_integer('max_steps', 1000000,
                                """Number of batches to run.""")
    tf.app.flags.DEFINE_integer('num_gpus', 2,
                                """How many GPUs to use.""")
    tf.app.flags.DEFINE_boolean('log_device_placement', False,
                                """Whether to log device placement.""")

    tf.app.flags.DEFINE_string('best_dir', './Current_training/best',
                                """Best eval dir.""")                            
    
    IMAGE_SIZE = model.IMAGE_SIZE


def mix_indexes(a,b):
    if a + b == 0:
        return 0
    else:
        return a*b/(a+b)

def _parse_function_no_distortion(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, features)
    image_decoded = tf.reshape(tf.decode_raw(parsed_features["image"], tf.uint8), [
                                IMAGE_SIZE, IMAGE_SIZE, 3])


    image_decoded = tf.cast(image_decoded, tf.float32)

    final_image = tf.image.per_image_standardization(image_decoded)
    return final_image, tf.cast(parsed_features["label"], tf.int32)


def _parse_function(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, features)

    image_decoded = tf.reshape(tf.decode_raw(parsed_features["image"], tf.uint8), [
                                IMAGE_SIZE, IMAGE_SIZE, 3])


    image_decoded = tf.cast(image_decoded, tf.float32)

    brightness_percentage = tf.random_uniform(
        [], minval=0, maxval=1, dtype=tf.float32)
    contrast_percentage = tf.random_uniform(
        [], minval=0, maxval=1, dtype=tf.float32)
    hue_percentage = tf.random_uniform(
        [], minval=0, maxval=1, dtype=tf.float32)
    saturation_percentage = tf.random_uniform(
        [], minval=0, maxval=1, dtype=tf.float32)
    rotation_percentage = tf.random_uniform(
        [], minval=0, maxval=1, dtype=tf.float32)
    zoom_percentage = tf.random_uniform(
        [], minval=0, maxval=1, dtype=tf.float32)
    skew_x_percentage = tf.random_uniform(
        [], minval=0, maxval=1, dtype=tf.float32)
    skew_y_percentage = tf.random_uniform(
        [], minval=0, maxval=1, dtype=tf.float32)
    translate_percentage = tf.random_uniform(
        [], minval=0, maxval=1, dtype=tf.float32)

    image_decoded = tf.image.random_flip_left_right(image_decoded)

    angle = tf.random_uniform(
        [1], minval=(-1 * (math.pi / 4)), maxval=math.pi / 4, dtype=tf.float32)
    image_rotated = tf.contrib.image.rotate(
        image_decoded, angle, interpolation='BILINEAR')
    image_decoded = tf.cond(rotation_percentage < 0.05,
                            lambda: image_rotated, lambda: image_decoded)

    image_brightness = tf.image.random_brightness(image_decoded, max_delta=0.8)
    image_decoded = tf.cond(brightness_percentage < 0.05,
                            lambda: image_brightness, lambda: image_decoded)

    image_contrast = tf.image.random_contrast(
        image_decoded, lower=0.7, upper=1.5)
    image_decoded = tf.cond(contrast_percentage < 0.05,
                            lambda: image_contrast, lambda: image_decoded)

    image_hue = tf.image.random_hue(image_decoded, max_delta=0.5)
    image_decoded = tf.cond(hue_percentage < 0.05,
                            lambda: image_hue, lambda: image_decoded)

    image_saturation = tf.image.random_saturation(
        image_decoded, lower=0.5, upper=1.5)
    image_decoded = tf.cond(saturation_percentage < 0.05,
                            lambda: image_saturation, lambda: image_decoded)

    zoom_scale = tf.random_uniform(
        [], minval=0.9, maxval=1.1, dtype=tf.float32)
    new_size = tf.constant(
        [IMAGE_SIZE, IMAGE_SIZE], dtype=tf.float32) * zoom_scale
    new_size = tf.cast(new_size, tf.int32)
    image_zoom = tf.image.resize_images(image_decoded, new_size)
    image_zoom = tf.image.resize_image_with_crop_or_pad(
        image_zoom, IMAGE_SIZE, IMAGE_SIZE)
    image_decoded = tf.cond(zoom_percentage < 0.05,
                            lambda: image_zoom, lambda: image_decoded)

    skew_x_angle = tf.random_uniform(
        [1], minval=(-1 * (math.pi / 12)), maxval=math.pi / 12, dtype=tf.float32)
    skew_x_tan = tf.tan(skew_x_angle)
    skew_x_vector_1 = tf.constant([1], dtype=tf.float32)
    skew_x_vector_2 = tf.constant([0, 0, 1, 0, 0, 0], dtype=tf.float32)
    skew_x_vector = tf.concat([skew_x_vector_1,skew_x_tan, skew_x_vector_2],0)
    skewed_x_image = tf.contrib.image.transform(image_decoded, skew_x_vector, interpolation='BILINEAR')
    image_decoded = tf.cond(skew_x_percentage < 0.05,
                            lambda: skewed_x_image, lambda: image_decoded)

    skew_y_angle = tf.random_uniform(
        [1], minval=(-1 * (math.pi / 12)), maxval=math.pi / 6, dtype=tf.float32)
    skew_y_tan = tf.tan(skew_y_angle)
    skew_y_vector_1 = tf.constant([1, 0, 0], dtype=tf.float32)
    skew_y_vector_2 = tf.constant([1, 0, 0, 0], dtype=tf.float32)
    skew_y_vector = tf.concat([skew_y_vector_1,skew_y_tan, skew_y_vector_2],0)
    skewed_y_image = tf.contrib.image.transform(image_decoded, skew_y_vector, interpolation='BILINEAR')
    image_decoded = tf.cond(skew_y_percentage < 0.05,
                            lambda: skewed_y_image, lambda: image_decoded)

    translate_y = tf.random_uniform(
        [1], minval=(-1 * (IMAGE_SIZE / 5)), maxval=IMAGE_SIZE / 6, dtype=tf.float32)
    translate_x = tf.random_uniform(
        [1], minval=(-1 * (IMAGE_SIZE / 5)), maxval=IMAGE_SIZE / 6, dtype=tf.float32)
    translate_vector_1 = tf.constant([1, 0], dtype=tf.float32)
    translate_vector_2 = tf.constant([0, 1], dtype=tf.float32)
    translate_vector_3 = tf.constant([0, 0], dtype=tf.float32)
    translate_vector = tf.concat(
        [translate_vector_1, translate_x, translate_vector_2, translate_y, translate_vector_3], 0)
    translated_image = tf.contrib.image.transform(image_decoded, translate_vector, interpolation='BILINEAR')
    image_decoded = tf.cond(translate_percentage < 0.05,
                            lambda: translated_image, lambda: image_decoded)    

    final_image = tf.image.per_image_standardization(image_decoded)
    return final_image, tf.cast(parsed_features["label"], tf.int32)


def tower_loss(scope, images, labels):
    """Calculate the total loss on a single tower running the CIFAR model.

    Args:
      scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
      images: Images. 4D tensor of shape [batch_size, height, width, 3].
      labels: Labels. 1D tensor of shape [batch_size].

    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """

    # Build inference Graph.
    embeddings = model.generate_embeddings(images, is_train=True)


    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = model.loss(embeddings, labels)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % model.TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)

    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def eval_once(sess, summary_writer, comparisons, tp_count_op, same_class_op, fp_count_op, different_classes_op, summary_op, acc_iterator, global_step, eval, num_examples, current_lr, lr):
    """Run Eval once.

    Args:
        saver: Saver.
        summary_writer: Summary writer.
        summary_op: Summary op.
    """

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                             start=True))

        num_iter = int(math.ceil(num_examples / (FLAGS.num_gpus*FLAGS.batch_size) ))
        total_sample_count = num_iter *FLAGS.num_gpus* (FLAGS.batch_size**2)
        step = 0

        tp_count = 0.
        sc_count = 0.
        fp_count = 0.
        dc_count = 0.
        comp_count = 0.
        while step < num_iter and not coord.should_stop():
            # predictions = sess.run([comparisons], {lr: current_lr})
            comp, tp, sc, fp, dc = sess.run([comparisons, tp_count_op, same_class_op, fp_count_op, different_classes_op])

            tp_count += np.sum(tp)
            sc_count += np.sum(sc)
            fp_count += np.sum(fp)
            dc_count += np.sum(dc)

            comp_count += np.sum(comp)
            step += 1

        # Compute precision @ 1.
        comp_rate = comp_count / total_sample_count
        tp_rate = float(tp_count) / sc_count
        fp_rate = float(fp_count) / dc_count

        mixed_tp_fp = mix_indexes(tp_rate, 1-fp_rate)

        summary = tf.Summary()
        # summary.ParseFromString(
        #     sess.run(summary_op, {lr: current_lr}) )
        summary.ParseFromString(
            sess.run(summary_op) )        
        if eval:
            summary.value.add(tag='Eval Comp Rate @ 1',
                              simple_value=comp_rate)            
            summary.value.add(tag='Eval TP Rate @ 1',
                              simple_value=tp_rate)
            summary.value.add(tag='Eval FP Rate @ 1',
                              simple_value=fp_rate) 
            summary.value.add(tag='Eval MixTPFP Rate @ 1',
                              simple_value=mixed_tp_fp)                                                             
        else:
            summary.value.add(tag='Training Comp Rate @ 1',
                              simple_value=comp_rate)             
            summary.value.add(tag='Training TP Rate @ 1',
                              simple_value=tp_rate)
            summary.value.add(tag='Training FP Rate @ 1',
                              simple_value=fp_rate)   
            summary.value.add(tag='Training MixTPFP Rate @ 1',
                              simple_value=mixed_tp_fp)                                  
        summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    return tp_rate, fp_rate, comp_rate, mixed_tp_fp


def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        num_steps_per_epoch = (
            len(t_sp)) / (FLAGS.batch_size * FLAGS.num_gpus)
        decay_steps = int(num_steps_per_epoch * NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        # lr = tf.placeholder( dtype = tf.float32)
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)   

        # Create an optimizer that performs gradient descent.
        # opt = tf.contrib.opt.NadamOptimizer(lr)
        opt = tf.contrib.opt.NadamOptimizer(lr)

        # Get images and labels for CIFAR-10.

        dataset = tf.data.TFRecordDataset(FLAGS.input_filename)
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=len(t_sp)))
        dataset = dataset.map(_parse_function, num_parallel_calls= 8)
        dataset = dataset.batch(FLAGS.batch_size)                                     
        dataset = dataset.prefetch(FLAGS.batch_size)
        iterator = dataset.make_initializable_iterator()        

        # GET TRAINING ACCURACY
        acc_dataset = tf.data.TFRecordDataset(FLAGS.input_filename)
        # Parse the record into tensors.
        acc_dataset = acc_dataset.map(_parse_function_no_distortion, num_parallel_calls= 8)
        acc_dataset = acc_dataset.repeat()  # Repeat the input indefinitely.
        acc_dataset = acc_dataset.batch(FLAGS.batch_size)
        acc_dataset = acc_dataset.prefetch(FLAGS.batch_size)
        acc_iterator = acc_dataset.make_initializable_iterator()

        eval_acc_dataset = tf.data.TFRecordDataset(FLAGS.eval_filename)
        # Parse the record into tensors.
        eval_acc_dataset = eval_acc_dataset.map(_parse_function_no_distortion, num_parallel_calls= 8)
        # Repeat the input indefinitely.
        eval_acc_dataset = eval_acc_dataset.repeat()
        eval_acc_dataset = eval_acc_dataset.batch(FLAGS.batch_size)
        eval_acc_dataset = eval_acc_dataset.prefetch(FLAGS.batch_size)
        eval_acc_iterator = eval_acc_dataset.make_initializable_iterator()

        # Calculate the gradients for each model tower.
        tower_grads = []

        acc_comp_list = []
        acc_tp_count_list = []
        acc_same_class_list = []
        acc_fp_count_list = []
        acc_different_classes_list = []

        eval_acc_comp_list = []
        eval_acc_tp_count_list = []
        eval_acc_same_class_list = []
        eval_acc_fp_count_list = []
        eval_acc_different_classes_list = []        


        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (model.TOWER_NAME, i)) as scope:
                        # Dequeues one batch for the GPU

                        image_batch, label_batch = iterator.get_next()

                        # Calculate the loss for one tower of the CIFAR model. This function
                        # constructs the entire CIFAR model but shares the variables across
                        # all towers.
                        loss = tower_loss(scope, image_batch, label_batch)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        if i > 0:
                            op = tf.get_default_graph().get_operation_by_name("tower_%d/acc_iterate"%(i-1))
                            with tf.control_dependencies([op]):
                                acc_images, acc_labels = iterator.get_next(name="acc_iterate")
                        else:
                            acc_images, acc_labels = acc_iterator.get_next(name="acc_iterate")


                        acc_embeddings = model.generate_embeddings(
                            acc_images, should_summarize=False)

                        acc_tp_count, acc_same_class, acc_fp_count, acc_different_classes, acc_comp_t = model.get_comparisons_tensor(acc_embeddings, acc_labels)

                        acc_tp_count_list.append(acc_tp_count)
                        acc_same_class_list.append(acc_same_class)
                        acc_fp_count_list.append(acc_fp_count)
                        acc_different_classes_list.append(acc_different_classes)
                        acc_comp_list.append(acc_comp_t)

                        tf.get_variable_scope().reuse_variables()

                        if i > 0:
                            op = tf.get_default_graph().get_operation_by_name("tower_%d/eval_acc_iterate"%(i-1))
                            with tf.control_dependencies([op]):
                                eval_acc_images, eval_acc_labels = iterator.get_next(name="eval_acc_iterate")
                        else:
                            eval_acc_images, eval_acc_labels = eval_acc_iterator.get_next(name="eval_acc_iterate")


                        eval_acc_embeddings = model.generate_embeddings(
                            eval_acc_images, should_summarize=False)

                        eval_acc_tp_count, eval_acc_same_class, eval_acc_fp_count, eval_acc_different_classes, eval_acc_comp_t = model.get_comparisons_tensor(eval_acc_embeddings, eval_acc_labels)

                        eval_acc_tp_count_list.append(eval_acc_tp_count)
                        eval_acc_same_class_list.append(eval_acc_same_class)
                        eval_acc_fp_count_list.append(eval_acc_fp_count)
                        eval_acc_different_classes_list.append(eval_acc_different_classes)
                        eval_acc_comp_list.append(eval_acc_comp_t)

                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(
                            tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)


        acc_tp_count = acc_tp_count_list[0]
        acc_same_class = acc_same_class_list[0]
        acc_fp_count = acc_fp_count_list[0]
        acc_different_classes = acc_different_classes_list[0]
        acc_comp = acc_comp_list[0]

        eval_acc_tp_count = eval_acc_tp_count_list[0]
        eval_acc_same_class = eval_acc_same_class_list[0]
        eval_acc_fp_count = eval_acc_fp_count_list[0]
        eval_acc_different_classes = eval_acc_different_classes_list[0]
        eval_acc_comp = eval_acc_comp_list[0]

        for i in xrange(1,FLAGS.num_gpus):
            acc_tp_count = tf.concat([acc_tp_count,acc_tp_count_list[i]],0)
            eval_acc_tp_count = tf.concat([eval_acc_tp_count,eval_acc_tp_count_list[i]],0)

            acc_same_class = tf.concat([acc_same_class,acc_same_class_list[i]],0)
            eval_acc_same_class = tf.concat([eval_acc_same_class,eval_acc_same_class_list[i]],0)

            acc_fp_count = tf.concat([acc_fp_count,acc_fp_count_list[i]],0)
            eval_acc_fp_count = tf.concat([eval_acc_fp_count,eval_acc_fp_count_list[i]],0)

            acc_different_classes = tf.concat([acc_different_classes,acc_different_classes_list[i]],0)
            eval_acc_different_classes = tf.concat([eval_acc_different_classes,eval_acc_different_classes_list[i]],0)                        

            acc_comp = tf.concat([acc_comp,acc_comp_list[i]],0)
            eval_acc_comp = tf.concat([eval_acc_comp,eval_acc_comp_list[i]],0)            

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', lr))

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(
                    var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            apply_gradient_op = opt.apply_gradients(
                grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            model.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        coord = tf.train.Coordinator()

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess, coord=coord)

        sess.run(acc_iterator.initializer)
        sess.run(eval_acc_iterator.initializer)
        sess.run(iterator.initializer)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        acc_last_tpr = 0.0
        acc_last_fpr = 1.0
        acc_last_comp = 0.0
        acc_last_mixed_tp_fp = 0.0

        eval_acc_last_tpr = 0.0
        eval_acc_last_fpr = 1.0
        eval_acc_last_comp = 0.0
        eval_acc_last_mixed_tp_fp = 0.0

        # draw(sess, image_batch, label_batch)
        # exit()

        best_tpr = 0.0
        best_fpr = 1.0
        best_comp = 0.0
        best_mixed_tp_fp = 0.0

        early_stopping_difference = 0.0001
        exit_counter = 0
        exit_limit = 7


        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            current_epoch = float((step * FLAGS.batch_size * FLAGS.num_gpus) //
                                  (len(t_sp)))

            if current_epoch == 0:
                current_lr = INITIAL_LEARNING_RATE
            else:
                current_lr = (INITIAL_LEARNING_RATE) / \
                    math.sqrt(current_epoch)
                if current_lr < 1e-6:
                    current_lr = 1e-6


            # _, loss_value = sess.run(
            #     [train_op, loss], {lr: current_lr})
            _, loss_value = sess.run(
                [train_op, loss])                     
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus

                epoch = int(current_epoch)

                format_str = ('\nSplit %s-%d: %s: step: %d; epoch: %d; loss = %.2f,\n \
                                last_tpr = %.2f, last_fpr = %.2f, last_comp = %.2f, last_mixed_tp_fp = %.2f \n \
                                last_eval_tpr =  %.2f, last_eval_fpr =  %.2f, last_eval_comp = %.2f, last_eval_mixed_tp_fp = %.2f \n \
                                best_eval_tpr =  %.2f, best_eval_fpr =  %.2f, best_eval_comp = %.2f, best_mixed_tp_fp = %.2f \n \
                                (%.1f examples/sec; %.3f sec/batch)\n')
                print (format_str % (split_size, split_index, datetime.now(), step, epoch, loss_value, \
                                     acc_last_tpr, acc_last_fpr, acc_last_comp, acc_last_mixed_tp_fp, \
                                     eval_acc_last_tpr, eval_acc_last_fpr, eval_acc_last_comp, eval_acc_last_mixed_tp_fp, \
                                     best_tpr, best_fpr, best_comp, best_mixed_tp_fp, \
                                     examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                # summary_str = sess.run(summary_op, {lr: current_lr})
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 200 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            if step % 200 == 0 and calculate_rates and step > 0:

                acc_last_tpr, acc_last_fpr, acc_last_comp, acc_last_mixed_tp_fp = eval_once(
                    sess, summary_writer, acc_comp, acc_tp_count, acc_same_class, acc_fp_count, acc_different_classes, summary_op, acc_iterator, step, False, FLAGS.num_examples, current_lr, lr)
                eval_acc_last_tpr, eval_acc_last_fpr, eval_acc_last_comp, eval_acc_last_mixed_tp_fp = eval_once(
                    sess, summary_writer, eval_acc_comp, eval_acc_tp_count, eval_acc_same_class, eval_acc_fp_count, eval_acc_different_classes, summary_op, eval_acc_iterator, step, True, FLAGS.eval_num_examples, current_lr, lr)

                if best_mixed_tp_fp == 0.0:
                    tpfp_difference = 1.
                else:
                    tpfp_difference = (eval_acc_last_mixed_tp_fp - best_mixed_tp_fp)/best_mixed_tp_fp
                   

                if tpfp_difference > early_stopping_difference :
                    exit_counter = 0
                    best_tpr = eval_acc_last_tpr
                    best_fpr = eval_acc_last_fpr
                    best_comp = eval_acc_last_comp
                    best_mixed_tp_fp = eval_acc_last_mixed_tp_fp

                    if tf.gfile.Exists(FLAGS.best_dir):
                        tf.gfile.DeleteRecursively(FLAGS.best_dir)
                        tf.gfile.MakeDirs(FLAGS.best_dir) 
                        copy_tree(FLAGS.train_dir,FLAGS.best_dir)
                else:
                    exit_counter = exit_counter + 1
                    if exit_counter == exit_limit:
                        pickle.dump([step], open(n_steps_file, 'wb')) 
                        exit()
                        
        pickle.dump([FLAGS.max_steps], open(n_steps_file, 'wb')) 



def draw(sess, image_batch, label_batch):

    im, label = sess.run([image_batch, label_batch])
    shape = im.shape
    for i in xrange(shape[0]):
        imagem = im[i, :, :, :]

        im_min = np.amin(imagem)
        im_max = np.amax(imagem)

        imagem_f = (((imagem - im_min) / (im_max - im_min))
                    * 255).astype(np.uint8)

        pImg = Image.fromarray(imagem_f, "RGB")
        pImg = pImg.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        pImg.show()

        raw_input()


def main(argv=None):  # pylint: disable=unused-argument
    global model, model_name, split_index, split_size
    global pickle_filename, t_sp, t_cl, v_sp, v_cl
    global n_steps_file
    model_name = argv[1] 
    model = importlib.import_module(argv[1])

    split_size = argv[2]
    split_index = int(argv[3])

    pickle_filename = './Splits/Splits_%s/split_%02d.pkl'%(split_size, split_index)
    t_sp, t_cl, v_sp, v_cl = pickle.load( open( pickle_filename, "rb" ) )

    t_sp = list(t_sp)
    t_cl = list(t_cl)
    v_sp = list(v_sp)
    v_cl = list(v_cl)


    n_steps_file = argv[4]


    initiate_flags()

    if tf.gfile.Exists(FLAGS.train_dir):
      tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)

    if tf.gfile.Exists(FLAGS.best_dir):
      tf.gfile.DeleteRecursively(FLAGS.best_dir)
    tf.gfile.MakeDirs(FLAGS.best_dir)   
    train()


if __name__ == '__main__':
    tf.app.run()
