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

from cfg import cfg
import numpy as np, os, struct
from base_dsl import BaseDSL, one_hot_labels
from os.path import expanduser, join

class MNISTDSL(BaseDSL):
    def __init__(self, batch_size, shuffle_each_epoch=False, seed=1337, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, path=None, resize=None):
        if mode == 'val':
            assert val_frac is not None

        if path is None:
            home = expanduser("~")
            self.path = os.path.join(cfg.dataset_dir, 'mnist')
        else:
            self.path = path
        
        super(MNISTDSL, self).__init__(
            batch_size,
            shuffle_each_epoch=shuffle_each_epoch,
            seed=seed,
            normalize=normalize,
            mode=mode,
            val_frac=val_frac,
            normalize_channels=normalize_channels,
            resize=resize
        )
        
    def is_multilabel(self):
        return False

    def load_data(self, mode, val_frac):
        if mode == 'test':
            fname_img = os.path.join(self.path, 't10k-images-idx3-ubyte')
            fname_lbl = os.path.join(self.path, 't10k-labels-idx1-ubyte')
        else:
            assert mode == 'train' or mode == 'val'
            fname_img = os.path.join(self.path, 'train-images-idx3-ubyte')
            fname_lbl = os.path.join(self.path, 'train-labels-idx1-ubyte')

        with open(fname_lbl, 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            self.labels = np.fromfile(flbl, dtype=np.int8)
        
        with open(fname_img, 'rb') as fimg:
            print fname_img
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            self.data = np.fromfile(fimg, dtype=np.uint8).reshape(len(self.labels), rows, cols, 1)

        # Perform splitting
        if val_frac is not None:
            self.partition_validation_set(mode, val_frac)
            
        self.labels = np.squeeze(self.labels)

    def convert_Y(self, Y):
        return one_hot_labels(Y, 10)
