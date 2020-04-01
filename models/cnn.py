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
from nn import NN

class CNN(NN):
    
    def __init__(self, height, width, channels, num_classes, multilabel=False, batch_size = None, is_training=True, conv_kernel_size=(3, 3), pool_kernel_size=(2, 2), num_filters=[32, 64, 128], fc_layers=[], l2_regularizer=0.001, convs_in_block=2, optimizer=None, loss_fn=None):
        
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.num_filters      = num_filters
        self.fc_layers        = fc_layers 
        self.l2_regularizer   = l2_regularizer
        self.convs_in_block   = convs_in_block
        
        assert len(conv_kernel_size) == 2
        assert len(pool_kernel_size) == 2
        
        super(CNN, self).__init__(
                height=height,
                width=width,
                channels=channels,
                num_classes=num_classes,
                multilabel=multilabel,
                batch_size=batch_size,
                is_training=is_training,
                optimizer=optimizer,
                loss_fn=loss_fn
            )
    
    def build_arch(self):
        """deepnn builds the graph for a deep net for classifying digits.
        Expects:
        self.X: an input tensor with the dimensions (N_examples, height, width, channels)
        self.model_type
        
        Creates:
        self.scores
        """
        
        self.convs = [self.X]
        
        if self.is_training:
            dropout = lambda x: tf.nn.dropout(x, self.dropout_keep_prob)
        else:
            dropout = lambda x: x
        
        for i, filters_i in enumerate(self.num_filters, start=1):

            with tf.variable_scope('conv%d' % i):
                for _ in range(self.convs_in_block):
                    self.convs.append(
                        tf.layers.batch_normalization(
                            tf.layers.conv2d(
                                inputs=self.convs[-1],
                                filters=filters_i,
                                kernel_size=self.conv_kernel_size,
                                padding="same",
                                activation=tf.nn.relu,
                                strides=1,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularizer)
                            )
                        )
                    )

            # Pooling layer - downsamples by 2X.
            with tf.variable_scope('pool%d' % i):                    
                self.convs.append(
                    dropout(
                        tf.layers.max_pooling2d(
                            inputs=self.convs[-1],
                            pool_size=self.pool_kernel_size,
                            strides=2
                        )
                    )
                )
        
        # Remove input
        self.convs = self.convs[1:]
        
        with tf.name_scope('flat'):
            dims = self.convs[-1].get_shape().as_list()
            dim  = reduce(lambda x, y: x*y, dims[1:])
            
            self.flat = tf.reshape(self.convs[-1], [-1, dim])
          
        self.fcs = [self.flat]
       
        for i, num_neurons in enumerate(self.fc_layers, start=1):
            with tf.variable_scope('fc%d' %i ): 
                    self.fcs.append(dropout(tf.layers.dense(self.fcs[-1], num_neurons, activation=tf.nn.relu)))
            
        with tf.variable_scope('scores'):
            self.scores = tf.layers.dense(self.fcs[-1], self.num_classes)


    def print_arch( self ):
        print self.X
        print self.labels
        
        for conv in self.convs:
            print conv
          
        for fc in self.fcs:
            print fc
        
        print self.scores
        print self.prob
        print self.predictions
