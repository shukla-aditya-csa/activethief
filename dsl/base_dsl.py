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

import numpy as np
import random, cv2, os
import logging

def one_hot_labels(Y, dim):
    b = np.zeros((len(Y), dim))
    b[np.arange(len(Y)), Y] = 1

    return b

class BaseDSL(object):
    def __init__(self, batch_size, shuffle_each_epoch=False, seed=1337, normalize=True, mode='train', val_frac=None, normalize_channels=False, resize=None):
        # Set the random seed
        if seed:
            random.seed(seed)

        self.random_state = random.getstate()
        
        print "Loading {} data" .format(mode)
        # Initializes self.data and self.labels
        self.load_data(mode, val_frac)
        self.shuffle_data()
        self.batch_index = 0

        # Resize dataset if it is not already of the specified size
        if resize is not None and not self.data.shape[1:3] == resize:
            print 'Resizing...'
            data = np.empty((self.data.shape[0], resize[0], resize[1], self.data.shape[3]))

            for i, image in enumerate(self.data):
                if self.data.shape[3] == 1:
                    data[i,:,:,0] = cv2.resize(image.squeeze(), resize)
                else:
                    data[i,:,:,:] = cv2.resize(image, resize)

            self.data = data

        # Normalize color channels if requested
        if normalize_channels:
            self.data  = np.mean(self.data, axis=-1)
            self.data  = np.expand_dims(self.data, -1)

            assert len(self.data.shape) == 4
        
        # Normalize data to lie in [0,1]
        if normalize:
            self.data = self.data/float(np.max(self.data))
            
            assert np.abs(np.min(self.data) - 0.0) < 1e-1
            assert np.abs(np.max(self.data) - 1.0) < 1e-1

        # Copy attributes from method call
        self.shuffle_each_epoch = shuffle_each_epoch
        self.batch_size         = batch_size

        last_batch_size = self.data.shape[0] % self.batch_size
        
        logging.info("Number of batches: %d" % self.get_num_batches())
        logging.info("Number of samples: %d" % self.get_num_samples())

        if last_batch_size != 0:
            logging.warn('Your last batch only has %d samples!' % last_batch_size)
    
    def is_multilabel(self):
        raise NotImplementedError
        
    def convert_Y(self):
        raise NotImplementedError
        
    def load_data(self):
        raise NotImplementedError
    
    def shuffle_data(self):
        random.setstate(self.random_state)

        perm = np.arange(self.data.shape[0]) 
        random.shuffle(perm)
        
        self.data   = self.data[perm]
        
        if self.labels is not None:
            self.labels = self.labels[perm]

        self.random_state = random.getstate()

    def partition_validation_set(self, mode, val_frac):
        self.shuffle_data()                            # Unconditional shuffle to prevent classes in train/val split contiguous
        train_end = int(len(self.data)*(1-val_frac))
        
        if mode == 'train':
            self.data = self.data[:train_end]
            
            if self.labels is not None:
                self.labels = self.labels[:train_end]
        elif mode == 'val':
            self.data = self.data[train_end:]
            
            if self.labels is not None:
                self.labels = self.labels[train_end:]

    def get_num_batches(self):
        return int(np.ceil(self.data.shape[0]/float(self.batch_size)))

    def get_batch_size(self):
        return self.batch_size

    def get_num_samples(self):
        return self.data.shape[0]

    def get_sample_shape(self):
        return self.data.shape[1:]
    
    def get_num_classes(self):
        return self.convert_Y(self.labels[:1]).shape[1]
    
    def reset_batch_counter(self):
        self.batch_index = 0

    def load_next_batch(self, batch_index=None):
        if batch_index is not None:
            self.batch_index = batch_index % self.get_num_batches()
        
        if self.batch_index == 0 and self.shuffle_each_epoch:
            self.shuffle_data()
        
        start = self.batch_size * self.batch_index
        end   = start + self.batch_size 
                
        X = self.data[start:end]

        if self.labels is not None:
            Y = self.labels[start:end]
            Y = self.convert_Y(Y)
        else:
            Y = None
            
        self.batch_index = (self.batch_index + 1) % self.get_num_batches()

        return X, Y
