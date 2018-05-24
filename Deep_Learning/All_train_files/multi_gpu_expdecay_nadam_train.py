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
import random
import importlib
import pickle

from distutils.dir_util import copy_tree

from PIL import Image

import math

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import cv2


NUM_EPOCHS_PER_DECAY = 100.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001     # Initial learning rate.

calculate_rates = True
FLAGS = tf.app.flags.FLAGS

jpg_pattern = re.compile('\w*\.jpg')
caltech_basefolder = './Dataset/Train'
pickle_filename = None
t_sp = None
t_cl = None
v_sp = None
v_cl = None

all_images_dict = {}

model = None
model_name = None

IMAGE_SIZE = 100

def initiate_flags():
    global IMAGE_SIZE, all_images_dict

    all_dirs = [i for i in os.listdir(caltech_basefolder) if os.path.isdir(os.path.join(caltech_basefolder,i))]
    all_dirs.sort()
    for s_class,s_dir in enumerate(all_dirs):
        full_dir = os.path.join(caltech_basefolder,s_dir)
        all_files = [i for i in os.listdir(full_dir) if jpg_pattern.search(i) is not None]
        img_keys = [os.path.join(s_dir,i) for i in all_files]
        for img_key in img_keys:
            all_images_dict[img_key] = {'class': s_class, 'data': cv2.imread(os.path.join(caltech_basefolder, img_key) ,cv2.IMREAD_COLOR)}


    tf.app.flags.DEFINE_string('train_dir', './Current_training/current',
                            """Directory where to write event logs """
                            """and checkpoint.""")

    tf.app.flags.DEFINE_string('input_folder', './Dataset/Train',
                            """TFRecords training set filename""")

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


def get_image_batch(index, train = True):
    global t_sp, t_cl

    if train and index == 0:
        c = list(zip(t_sp, t_cl))
        random.shuffle(c)
        t_sp, t_cl = zip(*c)

    full_sp = t_sp if train else v_sp
    full_cl = t_cl if train else v_cl

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

def _augment_input(raw_image_batch):

    def _augment_single_image(image_decoded):

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
        image_decoded = tf.cond(rotation_percentage < 0.4,
                                lambda: image_rotated, lambda: image_decoded)

        image_brightness = tf.image.random_brightness(image_decoded, max_delta=0.8)
        image_decoded = tf.cond(brightness_percentage < 0.3,
                                lambda: image_brightness, lambda: image_decoded)

        image_contrast = tf.image.random_contrast(
            image_decoded, lower=0.7, upper=1.5)
        image_decoded = tf.cond(contrast_percentage < 0.3,
                                lambda: image_contrast, lambda: image_decoded)

        image_hue = tf.image.random_hue(image_decoded, max_delta=0.5)
        image_decoded = tf.cond(hue_percentage < 0.1,
                                lambda: image_hue, lambda: image_decoded)

        image_saturation = tf.image.random_saturation(
            image_decoded, lower=0.5, upper=1.5)
        image_decoded = tf.cond(saturation_percentage < 0.3,
                                lambda: image_saturation, lambda: image_decoded)

        zoom_scale = tf.random_uniform(
            [], minval=0.9, maxval=1.1, dtype=tf.float32)
        new_size = tf.constant(
            [IMAGE_SIZE, IMAGE_SIZE], dtype=tf.float32) * zoom_scale
        new_size = tf.cast(new_size, tf.int32)
        image_zoom = tf.image.resize_images(image_decoded, new_size)
        image_zoom = tf.image.resize_image_with_crop_or_pad(
            image_zoom, IMAGE_SIZE, IMAGE_SIZE)
        image_decoded = tf.cond(zoom_percentage < 0.4,
                                lambda: image_zoom, lambda: image_decoded)

        skew_x_angle = tf.random_uniform(
            [1], minval=(-1 * (math.pi / 12)), maxval=math.pi / 12, dtype=tf.float32)
        skew_x_tan = tf.tan(skew_x_angle)
        skew_x_vector_1 = tf.constant([1], dtype=tf.float32)
        skew_x_vector_2 = tf.constant([0, 0, 1, 0, 0, 0], dtype=tf.float32)
        skew_x_vector = tf.concat([skew_x_vector_1,skew_x_tan, skew_x_vector_2],0)
        skewed_x_image = tf.contrib.image.transform(image_decoded, skew_x_vector, interpolation='BILINEAR')
        image_decoded = tf.cond(skew_x_percentage < 0.1,
                                lambda: skewed_x_image, lambda: image_decoded)

        skew_y_angle = tf.random_uniform(
            [1], minval=(-1 * (math.pi / 12)), maxval=math.pi / 6, dtype=tf.float32)
        skew_y_tan = tf.tan(skew_y_angle)
        skew_y_vector_1 = tf.constant([1, 0, 0], dtype=tf.float32)
        skew_y_vector_2 = tf.constant([1, 0, 0, 0], dtype=tf.float32)
        skew_y_vector = tf.concat([skew_y_vector_1,skew_y_tan, skew_y_vector_2],0)
        skewed_y_image = tf.contrib.image.transform(image_decoded, skew_y_vector, interpolation='BILINEAR')
        image_decoded = tf.cond(skew_y_percentage < 0.1,
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
        image_decoded = tf.cond(translate_percentage < 0.1,
                                lambda: translated_image, lambda: image_decoded)    

        final_image = tf.image.per_image_standardization(image_decoded)
        return final_image

    return tf.map_fn(_augment_single_image, raw_image_batch)


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
    logits = model.inference(images, is_train=True)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = model.loss(logits, labels)

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


def eval_once(sess, summary_writer, top_k_op, summary_op, global_step, eval, num_examples, images, labels, current_lr, lr):
    """Run Eval once.

    Args:
        saver: Saver.
        summary_writer: Summary writer.
        top_k_op: Top K op.
        summary_op: Summary op.
    """

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                             start=True))

        num_iter = int(math.ceil(num_examples / (2*FLAGS.batch_size) ))
        true_count = 0  # Counts the number of correct predictions.
        total_sample_count = num_iter * 2 *FLAGS.batch_size
        step = 0
        eval_index = 0
        while step < num_iter and not coord.should_stop():
            # predictions = sess.run([top_k_op], {lr: current_lr})

            in_dic = {}
            for x in xrange(FLAGS.num_gpus):
                in_dic[images[x]], in_dic[labels[x]] = get_image_batch(eval_index, train = False)
                eval_index = (eval_index + FLAGS.batch_size) if (eval_index + FLAGS.batch_size) < num_examples else 0

            predictions = sess.run([top_k_op], in_dic)
            true_count += np.sum(predictions)
            step += 1

        # Compute precision @ 1.
        precision = true_count / total_sample_count

        summary = tf.Summary()
        # summary.ParseFromString(
        #     sess.run(summary_op, {lr: current_lr}) )
        summary.ParseFromString(
            sess.run(summary_op) )        
        if eval:
            summary.value.add(tag='Eval Set Precision @ 1',
                              simple_value=precision)
        else:
            summary.value.add(tag='Training Precision @ 1',
                              simple_value=precision)
        summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    return precision


def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        images = []
        labels = []
        acc_images = []
        acc_labels = []
        eval_acc_images = []
        eval_acc_labels = []                
        for i in xrange(FLAGS.num_gpus):
            images.append(_augment_input(tf.placeholder( dtype = tf.float32, shape=[FLAGS.batch_size,IMAGE_SIZE, IMAGE_SIZE]) ))
            labels.append(tf.placeholder( dtype = tf.int32, shape=[FLAGS.batch_size]) )

            acc_images.append(_normalize_input(tf.placeholder( dtype = tf.float32, shape=[FLAGS.batch_size,IMAGE_SIZE, IMAGE_SIZE]) ))
            acc_labels.append(tf.placeholder( dtype = tf.int32, shape=[FLAGS.batch_size]) )

            eval_acc_images.append(_normalize_input(tf.placeholder( dtype = tf.float32, shape=[FLAGS.batch_size,IMAGE_SIZE, IMAGE_SIZE]) ))
            eval_acc_labels.append(tf.placeholder( dtype = tf.int32, shape=[FLAGS.batch_size]) )                        

        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        num_steps_per_epoch = (
            model.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN) / (FLAGS.batch_size * FLAGS.num_gpus)
        decay_steps = int(num_steps_per_epoch * NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        # lr = tf.placeholder( dtype = tf.float32)
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)   

        # Create an optimizer that performs gradient descent.
        opt = tf.contrib.opt.NadamOptimizer(lr)


        # Calculate the gradients for each model tower.
        tower_grads = []
        acc_top_k_op_list = []
        eval_acc_top_k_op_list = []

        acc_logits = [None] * FLAGS.num_gpus
        eval_acc_logits = [None] * FLAGS.num_gpus
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (model.TOWER_NAME, i)) as scope:
                        # Dequeues one batch for the GPU

                        # Calculate the loss for one tower of the CIFAR model. This function
                        # constructs the entire CIFAR model but shares the variables across
                        # all towers.
                        loss = tower_loss(scope, images[i], labels[i])

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()


                        acc_logits[i] = model.inference(
                            acc_images[i], should_summarize=False)
                        acc_top_k_op_list.append(tf.nn.in_top_k(
                            acc_logits[i], acc_labels[i], 1))
                        tf.get_variable_scope().reuse_variables()


                        eval_acc_logits[i] = model.inference(
                            eval_acc_images[i], should_summarize=False)
                        eval_acc_top_k_op_list.append(tf.nn.in_top_k(
                            eval_acc_logits[i], eval_acc_labels[i], 1))
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(
                            tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        acc_top_k_op = acc_top_k_op_list[0]
        eval_acc_top_k_op = eval_acc_top_k_op_list[0]
        for i in xrange(1,FLAGS.num_gpus):
            acc_top_k_op = tf.concat([acc_top_k_op,acc_top_k_op_list[i]],0)
            eval_acc_top_k_op = tf.concat([eval_acc_top_k_op,eval_acc_top_k_op_list[i]],0)

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


        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        acc_last_precision = 0.0
        eval_acc_last_precision = 0.0

        # draw(sess, image_batch, label_batch)
        # exit()

        best_precision = 0.0
        early_stopping_difference = 0.0001
        exit_counter = 0
        exit_limit = 2


        train_index = 0
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            current_epoch = float((step * FLAGS.batch_size * FLAGS.num_gpus) //
                                  (model.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN))

            if current_epoch == 0:
                current_lr = INITIAL_LEARNING_RATE
            else:
                current_lr = (INITIAL_LEARNING_RATE) / \
                    math.sqrt(current_epoch)
                if current_lr < 1e-6:
                    current_lr = 1e-6

            in_dic = {}
            for x in xrange(FLAGS.num_gpus):
                in_dic[images[x]], in_dic[labels[x]] = get_image_batch(train_index, train = True)
                train_index = (train_index + FLAGS.batch_size) if (train_index + FLAGS.batch_size) < len(t_sp) else 0


            # _, loss_value = sess.run(
            #     [train_op, loss], {lr: current_lr})
            _, loss_value = sess.run(
                [train_op, loss], in_dic)                     
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus

                epoch = int(current_epoch)

                format_str = ('%s: %s: step: %d; epoch: %d; loss = %.2f, last_precision = %.2f, last_eval_precision =  %.2f, best_eval_precision =  %.2f  (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (model_name, datetime.now(), step, epoch, loss_value, acc_last_precision, eval_acc_last_precision, best_precision,
                                     examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                # summary_str = sess.run(summary_op, {lr: current_lr})
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            if step % 5000 == 0 and calculate_rates and step > 0:
                acc_last_precision = eval_once(
                    sess, summary_writer, acc_top_k_op, summary_op, step, False, FLAGS.num_examples, acc_images, acc_labels, current_lr, lr)
                eval_acc_last_precision = eval_once(
                    sess, summary_writer, eval_acc_top_k_op, summary_op, step, True, FLAGS.eval_num_examples, eval_acc_images, eval_acc_labels, current_lr, lr)

                if best_precision == 0.0:
                    precision_difference = 1.
                else:
                    precision_difference = (eval_acc_last_precision - best_precision)/best_precision

                if precision_difference > early_stopping_difference:
                    exit_counter = 0
                    best_precision = eval_acc_last_precision
                    if tf.gfile.Exists(FLAGS.best_dir):
                        tf.gfile.DeleteRecursively(FLAGS.best_dir)
                        tf.gfile.MakeDirs(FLAGS.best_dir) 
                        copy_tree(FLAGS.train_dir,FLAGS.best_dir)
                else:
                    exit_counter = exit_counter + 1
                    if exit_counter == exit_limit:
                        exit()




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
    global model, model_name
    global pickle_filename, t_sp, t_cl, v_sp, v_cl
    model_name = argv[1] 
    model = importlib.import_module(argv[1])

    pickle_filename = argv[2]
    t_sp, t_cl, v_sp, v_cl = pickle.load( open( pickle_filename, "rb" ) )

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
