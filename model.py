#!/usr/bin/env python
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

"""A binary to train CIFAR-10 using a single GPU.
Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.
Speed: With batch_size 128.
System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import os
import re
import sys

import tensorflow as tf
import file_gen,glob


FILES = 353
EXAMPLES_PER_FILE = 1000
TOTAL_NUMBER_OF_EXAMPLES = FILES * EXAMPLES_PER_FILE
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = TOTAL_NUMBER_OF_EXAMPLES * 0.8
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL  = TOTAL_NUMBER_OF_EXAMPLES * 0.2



# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 1.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
NUM_CLASSES = 3

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")


def main(argv=None):  # pylint: disable=unused-argument
   
   if tf.gfile.Exists(FLAGS.checkpoint_dir):
      tf.gfile.DeleteRecursively(FLAGS.checkpoint_dir)
   tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

   file_list = glob.glob('/Users/jchilders/workdir/ml/output/*.npz')
   fileGenerator = file_gen.FileGenerator(file_list,NUM_CLASSES,batch_size=FLAGS.batch_size)

   # Variables that affect learning rate.
   num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
   decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

   tf.logging.info('decay_steps = %s',decay_steps)

   """Train CIFAR-10 for a number of steps."""
   with tf.Graph().as_default():

      global_step = tf.train.get_or_create_global_step()

      # Get images and labels for CIFAR-10.
      # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
      # GPU and resulting in a slow down.
      with tf.device('/cpu:0'):
         images, labels = fileGenerator.next()

      # Build a Graph that computes the logits predictions from the
      # inference model.
      logits = inference(images)

      tf.logging.info(' %s %s',labels.shape,logits.shape)

      # Calculate loss.
      loss = losscalc(logits, labels)

      saver = tf.train.Saver(max_to_keep=0)

      # Build a Graph that trains the model with one batch of examples and
      # updates the model parameters.
      train_op = train(loss, global_step)

      class _LoggerHook(tf.train.SessionRunHook):
         """Logs loss and runtime."""

         def begin(self):
            self._step = -1
            self._start_time = time.time()

         def before_run(self, run_context):
            self._step += 1
            return tf.train.SessionRunArgs(loss)  # Asks for loss value.

         def after_run(self, run_context, run_values):
            if self._step % FLAGS.log_frequency == 0:
               current_time = time.time()
               duration = current_time - self._start_time
               self._start_time = current_time

               loss_value = run_values.results
               examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
               sec_per_batch = float(duration / FLAGS.log_frequency)

               format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                           'sec/batch)')
               print (format_str % (datetime.now(), self._step, loss_value,
                                  examples_per_sec, sec_per_batch))

      with tf.train.MonitoredTrainingSession(
         checkpoint_dir=FLAGS.checkpoint_dir,
         hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
         config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:

         while not mon_sess.should_stop():
            mon_sess.run(train_op)

def inference(images):
   """Build the CIFAR-10 model.
   Args:
    images: Images returned from distorted_inputs() or inputs().
   Returns:
    Logits.
   """
   # We instantiate all variables using tf.get_variable() instead of
   # tf.Variable() in order to share variables across multiple GPU training runs.
   # If we only ran this model on a single GPU, we could simplify this function
   # by replacing all instances of tf.get_variable() with tf.Variable().
   #
   # conv1
   with tf.variable_scope('conv1') as scope:
      kernel = _variable_with_weight_decay('weights',
                                         shape=[5,5,2,64],
                                         stddev=5e-2,
                                         wd=None)
      conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
      pre_activation = tf.nn.bias_add(conv, biases)
      conv1 = tf.nn.relu(pre_activation, name=scope.name)
      _activation_summary(conv1)

   # pool1
   pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
   # norm1
   norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

   # conv2
   with tf.variable_scope('conv2') as scope:
      kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=None)
      conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
      pre_activation = tf.nn.bias_add(conv, biases)
      conv2 = tf.nn.relu(pre_activation, name=scope.name)
      _activation_summary(conv2)

   # norm2
   norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
   # pool2
   pool2 = tf.nn.max_pool(norm2, 
                          ksize=[1, 3, 3, 1],
                          strides=[1, 2, 2, 1], 
                          padding='SAME', 
                          name='pool2')

   # local3
   with tf.variable_scope('local3') as scope:
      # Move everything into depth so we can perform a single matrix multiply.
      reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
      dim = reshape.get_shape()[1].value
      weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
      biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
      local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
      _activation_summary(local3)

   # local4
   with tf.variable_scope('local4') as scope:
      weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
      biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
      local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
      _activation_summary(local4)

   # linear layer(WX + b),
   # We don't apply softmax here because
   # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
   # and performs the softmax internally for efficiency.
   with tf.variable_scope('softmax_linear') as scope:
      weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=None)
      biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
      softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
      _activation_summary(softmax_linear)

   return softmax_linear

def losscalc(logits, labels):
   """Add L2Loss to all the trainable variables.
   Add summary for "Loss" and "Loss/avg".
   Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
   Returns:
    Loss tensor of type float.
   """
   # Calculate the average cross entropy loss across the batch.
   labels = tf.cast(labels, tf.int64)
   cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
   cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
   tf.add_to_collection('losses', cross_entropy_mean)

   # The total loss is defined as the cross entropy loss plus all of the weight
   # decay terms (L2 loss).
   return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _variable_on_cpu(name, shape, initializer):
   """Helper to create a Variable stored on CPU memory.
   Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
   Returns:
    Variable Tensor
   """
   with tf.device('/cpu:0'):
      dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
      var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
   return var


def _variable_with_weight_decay(name, shape, stddev, wd):
   """Helper to create an initialized Variable with weight decay.
   Note that the Variable is initialized with a truncated normal distribution.
   A weight decay is added only if one is specified.
   Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
   Returns:
    Variable Tensor
   """
   dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
   var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
   if wd is not None:
      weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
      tf.add_to_collection('losses', weight_decay)
   return var

def _activation_summary(x):
   """Helper to create summaries for activations.
   Creates a summary that provides a histogram of activations.
   Creates a summary that measures the sparsity of activations.
   Args:
    x: Tensor
   Returns:
    nothing
   """
   # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
   # session. This helps the clarity of presentation on tensorboard.
   tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
   tf.summary.histogram(tensor_name + '/activations', x)
   tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))

def _add_loss_summaries(total_loss):
   """Add summaries for losses in CIFAR-10 model.
   Generates moving average for all losses and associated summaries for
   visualizing the performance of the network.
   Args:
    total_loss: Total loss from loss().
   Returns:
    loss_averages_op: op for generating moving averages of losses.
   """
   # Compute the moving average of all individual losses and the total loss.
   loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
   losses = tf.get_collection('losses')
   loss_averages_op = loss_averages.apply(losses + [total_loss])

   # Attach a scalar summary to all individual losses and the total loss; do the
   # same for the averaged version of the losses.
   for l in losses + [total_loss]:
      # Name each loss as '(raw)' and name the moving average version of the loss
      # as the original loss name.
      tf.summary.scalar(l.op.name + '_raw', l)
      tf.summary.scalar(l.op.name, loss_averages.average(l))

   return loss_averages_op

def train(total_loss, global_step):
   """Train CIFAR-10 model.
   Create an optimizer and apply to all trainable variables. Add moving
   average for all trainable variables.
   Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
   Returns:
    train_op: op for training.
   """
   # Variables that affect learning rate.
   num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
   decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

   # Decay the learning rate exponentially based on the number of steps.
   lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=False)
   tf.summary.scalar('learning_rate', lr)

   # Generate moving averages of all losses and associated summaries.
   loss_averages_op = _add_loss_summaries(total_loss)

   # Compute gradients.
   with tf.control_dependencies([loss_averages_op]):
      opt = tf.train.GradientDescentOptimizer(lr)
      grads = opt.compute_gradients(total_loss)

   # Apply gradients.
   apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

   # Add histograms for trainable variables.
   for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

   # Add histograms for gradients.
   for grad, var in grads:
      if grad is not None:
         tf.summary.histogram(var.op.name + '/gradients', grad)

   # Track the moving averages of all trainable variables.
   variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
   variables_averages_op = variable_averages.apply(tf.trainable_variables())

   with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      train_op = tf.no_op(name='train')

   return train_op


if __name__ == '__main__':
   tf.app.run()