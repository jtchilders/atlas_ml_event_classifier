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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import os
import re
import sys

import tensorflow as tf
import file_gen,glob,numpy


FILES = 353
EXAMPLES_PER_FILE = 1000
TOTAL_NUMBER_OF_EXAMPLES = FILES * EXAMPLES_PER_FILE
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = TOTAL_NUMBER_OF_EXAMPLES * 0.8
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL  = TOTAL_NUMBER_OF_EXAMPLES * 0.2
EVAL_NUM_EXAMPLES = 1000


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


tf.app.flags.DEFINE_string('eval_data_path', '/Users/jchilders/workdir/ml/eval_data/',
                           """Directory where to read eval data.""")
tf.app.flags.DEFINE_string('eval_dir', '.',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")



def main(argv=None):  # pylint: disable=unused-argument
   
   
   eval_file_list = glob.glob(FLAGS.eval_data_path + '/*.npz')
   evalFileGenerator = file_gen.FileGenerator(eval_file_list,NUM_CLASSES,batch_size=FLAGS.batch_size)
   


   """Train CIFAR-10 for a number of steps."""
   with tf.Graph().as_default() as graph:
      
      # Get images and labels for CIFAR-10.
      # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
      # GPU and resulting in a slow down.
      with tf.device('/cpu:0'):
         images = tf.placeholder(tf.float32,(FLAGS.batch_size,60,64,2))
         labels = tf.placeholder(tf.float32,(FLAGS.batch_size))


      # Build a Graph that computes the logits predictions from the
      # inference model.
      logits = inference(images)
      #eval_logits = inference(eval_images,True)

      # For evaluation
      #top_k_op      = tf.nn.in_top_k (logits,tf.cast(labels, tf.int32),1)

      # get predicted labels:
      argmax        = tf.argmax(logits,1)

      # create confusion matrix
      cf = tf.confusion_matrix(labels,argmax)

      saver = tf.train.Saver()

      with tf.Session() as sess:
         
         ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
         if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
         else:
            tf.logging.error('No checkpoint file found')
            return

         num_steps = len(evalFileGenerator)
         step = 0
         summed_confusion_matrix = None
         tf.logging.info('running %s steps',num_steps)
         while step < num_steps:
            
            eval_images,eval_labels = evalFileGenerator.next()
            output = sess.run([cf],feed_dict={images:eval_images,labels:eval_labels})
            confusion_matrix = output[0]
            if summed_confusion_matrix is None:
               summed_confusion_matrix = confusion_matrix
            else:
               summed_confusion_matrix = tf.add(summed_confusion_matrix,confusion_matrix)
            step += 1

            if step % 25 == 0:
               tf.logging.info(' on step %s',step)
               tf.logging.info('confusion_matrix: %s',summed_confusion_matrix.eval())
               tf.logging.info('sum = %s',tf.reduce_sum(summed_confusion_matrix).eval())
         
         all_entries = tf.reduce_sum(summed_confusion_matrix).eval()
         summed_confusion_matrix = summed_confusion_matrix.eval()
         tf.logging.info('confusion_matrix (row = truth, column = prediction): %s',summed_confusion_matrix)
         tf.logging.info('sum = %s',all_entries)

         # normalize by row (truth)
         cf_norm = numpy.zeros((3,3))
         for i in range(3):
            row_sum = numpy.sum(summed_confusion_matrix[i])
            for j in range(3):
               cf_norm[i][j] = float(summed_confusion_matrix[i][j])/float(row_sum)

         tf.logging.info('normalized by row confusion matrix (row=truth,column=prediction): %s',cf_norm)




      #while True:
      #eval_once(saver, top_k_op,evalFileGenerator,images,labels)
      #   if FLAGS.run_once:
      #     break
      #   time.sleep(FLAGS.eval_interval_secs)


def eval_once(saver, summary_writer, top_k_op, summary_op,gen,images,labels):
   """Run Eval once.
   Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
   """
   with tf.Session() as sess:
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

         num_iter = int(numpy.ceil(FLAGS.num_examples / FLAGS.batch_size))
         true_count = 0  # Counts the number of correct predictions.
         total_sample_count = num_iter * FLAGS.batch_size
         step = 0
         while step < num_iter and not coord.should_stop():
            eval_images,eval_labels = gen.next()
            predictions = sess.run([top_k_op],feed_dict={images:eval_images,labels:eval_labels})
            true_count += numpy.sum(predictions)
            step += 1

         # Compute precision @ 1.
         precision = true_count / total_sample_count
         print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

         summary = tf.Summary()
         summary.ParseFromString(sess.run(summary_op))
         summary.value.add(tag='Precision @ 1', simple_value=precision)
         summary_writer.add_summary(summary, global_step)
      except Exception as e:  # pylint: disable=broad-except
         coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)

def inference(images, reuse = False):
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
   with tf.variable_scope('conv1',reuse=reuse) as scope:
      kernel = _variable_with_weight_decay('weights',
                                         shape=[5,5,2,64],
                                         stddev=5e-2,
                                         wd=None)
      conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
      pre_activation = tf.nn.bias_add(conv, biases)
      conv1 = tf.nn.relu(pre_activation, name=scope.name)

   # pool1
   pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
   # norm1
   norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

   # conv2
   with tf.variable_scope('conv2',reuse=reuse) as scope:
      kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=None)
      conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
      pre_activation = tf.nn.bias_add(conv, biases)
      conv2 = tf.nn.relu(pre_activation, name=scope.name)

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
   with tf.variable_scope('local3',reuse=reuse) as scope:
      # Move everything into depth so we can perform a single matrix multiply.
      reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
      dim = reshape.get_shape()[1].value
      weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
      biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
      local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

   # local4
   with tf.variable_scope('local4',reuse=reuse) as scope:
      weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
      biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
      local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

   # linear layer(WX + b),
   # We don't apply softmax here because
   # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
   # and performs the softmax internally for efficiency.
   with tf.variable_scope('softmax_linear',reuse=reuse) as scope:
      weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=None)
      biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
      softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

   return softmax_linear



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


if __name__ == '__main__':
   tf.app.run()