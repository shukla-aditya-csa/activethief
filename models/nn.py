"""
MIT License

Copyright (c) 2019 Soham Pal, Yash Gupta, Aditya Shukla, Aditya Kanade,
Shirish Shevade, Vinod Ganapathy. Indian Institute of Science.

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import division
import tensorflow as tf

class NN(object):
    
    def __init__(self, height, width, channels, num_classes, multilabel=False, batch_size = None, is_training=True, optimizer=None, loss_fn=None):
        self.X                 = tf.placeholder(tf.float32, shape=(None, height, width, channels), name='X')
        self.labels            = tf.placeholder(tf.float32, shape=(None, num_classes), name='labels')
        self.dropout_keep_prob = tf.placeholder_with_default(tf.constant(1.0, dtype=tf.float32), tuple(),name='dropout_keep_prob')
        
        self.num_classes = num_classes
        self.is_training = is_training
        self.height      = height
        self.width       = width
        self.channels    = channels
        self.multilabel  = multilabel
        self.batch_size  = batch_size
        self.optimizer   = optimizer
        self.loss_fn     = loss_fn
        self.build_arch()
        self.normalize_scores()
        self.measure_accuracy()
        self.setup_loss()
        if is_training:
            self.setup_training()
            
        tf.logging.info('Seting up the main structure')

    def get_batch_size(self):
        return self.batch_size
        
    def measure_accuracy(self):
        with tf.name_scope("accuracy"):
            if self.multilabel:
                ground_truth              = tf.round(self.labels)
            else: 
                ground_truth              = tf.argmax(self.labels, axis=-1)
            
            correct_prediction            = tf.equal(ground_truth, self.predictions)
            self.sum_correct_prediction   = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
            self.accuracy                 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    def is_multilabel(self):
        return self.multilabel
    
    def get_num_classes(self):
        return self.num_classes
    
    def build_arch(self):
        """Architecture for the desired neural network
        """
        raise NotImplementedError("subclasses must override build_arch")
        
    def setup_loss(self):
 
        with tf.name_scope('loss'):
            if self.loss_fn is None:
                if self.multilabel:
                    self.loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
                else:
                    self.loss_fn = tf.nn.softmax_cross_entropy_with_logits

                self.cross_entropy = self.loss_fn(labels=tf.stop_gradient(self.labels), logits=self.scores)
            else:
                self.cross_entropy, _ = self.loss_fn(self.labels, self.prob)
            
            self.sum_loss      = tf.reduce_sum(self.cross_entropy)
            self.mean_loss     = tf.reduce_mean(self.cross_entropy)
    
    def setup_training(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if self.optimizer is None:
            self.optimizer = tf.train.AdamOptimizer()
        self.train_op    = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
        self._summary()
    
    def normalize_scores(self):
        """ Normalizes the raw scores using softmax/sigmoid depending on model type """
        assert self.scores is not None
        
        with tf.name_scope('output'):
            if self.multilabel:
                self.prob        = tf.nn.sigmoid(self.scores, name="prob")
                self.predictions = tf.round( self.prob )
                self.predictions_one_hot = self.predictions
            else:
                self.prob                = tf.nn.softmax(self.scores, name="prob")
                self.predictions         = tf.argmax(self.prob, axis = 1)
                self.predictions_one_hot = tf.one_hot(self.predictions, self.num_classes) 

    def print_trainable_parameters( self ):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print "Total trainable parameters ", total_parameters        
    
    def _summary(self):
        """Summary information (loss, accuracy) for tensboard"""
        train_summary = []
        train_summary.append(tf.summary.scalar('train/mean_loss', self.mean_loss))
        train_summary.append(tf.summary.scalar('train/accuracy', self.accuracy))
        self.train_summary = tf.summary.merge(train_summary)


    def print_arch( self ):
        "Prints the architecture"
        raise NotImplementedError("subclasses must override print_arch")

    def get_graph(self):
        return tf.get_default_graph()