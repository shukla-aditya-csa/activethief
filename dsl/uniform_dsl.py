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
import random
from base_dsl import BaseDSL

class UniformDSL(BaseDSL):
    def __init__(self, batch_size, shuffle_each_epoch=False, seed=1337, normalize=False, mode='train', val_frac=None, normalize_channels=False, resize=None, shape=None, sample_limit=None):
        assert resize is None, 'Does not support resizing.'
        assert shape is not None, 'Shape must be specified.'
        assert normalize is False, 'Normalization is not supported.'
        assert normalize_channels is False, 'Normalization is not supported.'
        
        self.shuffle_each_epoch = False
        
        self.seed        = seed
        self.batch_size  = batch_size
        
        if mode == 'val':
            self.seed *= 2
        elif mode == 'test':
            self.seed *= 3
        
        if sample_limit is None:
            assert shuffle_each_epoch is False
            self.num_batches = np.infty
            self.num_samples = np.infty
            
            self.get_num_batches = lambda : np.infty
            self.get_num_samples = lambda : np.infty
        else:
            self.num_batches = int(np.ceil(sample_limit/float(batch_size)))
            self.num_samples = sample_limit
        
        self.shape = shape
        self.load_data(mode, val_frac)
        self.random_state = random.getstate()
    
    def load_data(self, mode, val_frac):
        if np.isinf(self.num_samples):
            self.load_next_batch = self._load_next_batch
        else:
            self.data = np.empty((self.num_samples, self.shape[0], self.shape[1], self.shape[2]), dtype=np.float32)
            self.labels = None
            self.rs = np.random.RandomState(self.seed)
            
            for i in range(self.num_batches):
                self.data[i*self.batch_size:(i+1)*self.batch_size] = self.rs.uniform(size=(self.batch_size, self.shape[0], self.shape[1], self.shape[2]))

    def get_sample_shape(self):
        return self.shape

    def _load_next_batch(self, batch_index):
        if batch_index == 0:
            self.rs = np.random.RandomState(self.seed)
        
        X = self.rs.uniform(size=(self.batch_size, self.shape[0], self.shape[1], self.shape[2]))
        
        return X, None
