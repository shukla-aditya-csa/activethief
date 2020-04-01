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
import numpy as np, os
import json
from base_dsl import BaseDSL, one_hot_labels
from os.path import expanduser, join

class ImagenetDSL(BaseDSL):
    def __init__(self, batch_size, shuffle_each_epoch=False, seed=1337, normalize=True, mode='train', val_frac=None, normalize_channels=False, path=None, resize=None, start_batch=1, end_batch=1, num_to_keep=None):
        assert val_frac is None, 'This dataset has pre-specified splits.'
        assert start_batch >= 1
        assert end_batch <= 10
        
        self.start_batch = start_batch
        self.end_batch = end_batch

        if path is None:
            self.path = os.path.join(cfg.dataset_dir, 'Imagenet64')
        else:
            self.path = path
        
        self.num_to_keep = num_to_keep
        
        super(ImagenetDSL, self).__init__(
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
        xs = []
        ys = []

        if mode == 'train':
            data_files = [os.path.join(self.path, 'train_data_batch_%d.json' % idx) for idx in range(self.start_batch, self.end_batch+1)]
        else:
            assert mode == 'val', 'Mode not supported.'
            data_files = [os.path.join(self.path, 'val_data.json')]

        for data_file in data_files:
            print('Loading', data_file)
        
            with open(data_file, 'rb') as data_file_handle:
                d = json.load(data_file_handle)
        
            x = np.array(d['data'], dtype=np.float32)
            y = np.array(d['labels'])

            # Labels are indexed from 1, shift it so that indexes start at 0
            y = [i-1 for i in y]

            img_size  = 64
            img_size2 = img_size * img_size

            x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
            x = x.reshape((x.shape[0], img_size, img_size, 3))

            xs.append(x)
            ys.append(np.array(y))

        if len(xs) == 1:
            self.data   = xs[0]
            self.labels = ys[0]
        else:
            self.data   = np.concatenate(xs, axis=0)
            self.labels = np.concatenate(ys, axis=0)

        if self.num_to_keep is not None:
            self.shuffle_data()
            self.data = self.data[:self.num_to_keep]
            self.labels = self.labels[:self.num_to_keep]

    def convert_Y(self, Y):
        return one_hot_labels(Y, 1000)
