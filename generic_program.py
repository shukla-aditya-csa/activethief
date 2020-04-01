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

from __future__ import absolute_import
from __future__ import division
import sys, time, os, logging
import numpy as np
import shutil
from utils.model import *
from utils.class_loader import *
from dsl.uniform_dsl import UniformDSL
from dsl.imagenet_dsl import ImagenetDSL
import tensorflow as tf
from cfg import cfg

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

for key, value in tf.flags.FLAGS.__flags.items():
    try:
        logging.info("{} {}" .format(key, value.value) )
        print "{} {}" .format(key, value.value) 
    except AttributeError:
        logging.info("{} {}" .format(key, value) )
        print "{} {}" .format(key, value)
        
assert cfg.source_model is not None
assert cfg.copy_model is not None
assert cfg.true_dataset is not None
assert cfg.noise_dataset is not None

logdir_true = os.path.join('logdir', cfg.source_model, cfg.true_dataset, 'true')

noise_dataset = cfg.noise_dataset

if cfg.num_to_keep is not None:
    noise_dataset = noise_dataset + '-' + str(cfg.num_to_keep)

if cfg.iterative:
    assert cfg.initial_seed is not None 
    assert cfg.val_size is not None
    assert cfg.num_iter is not None
    assert cfg.k is not None
    noise_dataset = "{}-{}-{}+{}+{}-{}" .format( noise_dataset, cfg.sampling_method, cfg.initial_seed, cfg.val_size , cfg.num_iter * cfg.k, cfg.optimizer )
    
if cfg.copy_one_hot:
    api_retval = 'onehot'
else:
    api_retval = 'softmax'

logdir_copy = os.path.join('logdir' , cfg.source_model, cfg.true_dataset, api_retval, cfg.copy_model, noise_dataset)

logdir_papernot_copy = os.path.join('logdir', cfg.source_model, cfg.true_dataset, api_retval, cfg.copy_model, 'papernot')


print "seed set is ", cfg.seed

print logdir_true

print logdir_copy

print logdir_papernot_copy

true_dataset_dsl = load_dataset(cfg.true_dataset) 

train_dsl = true_dataset_dsl(batch_size = cfg.batch_size, mode='train', shuffle_each_epoch=True, seed=cfg.seed)
val_dsl   = true_dataset_dsl(batch_size = cfg.batch_size, mode='val', shuffle_each_epoch=False, seed=cfg.seed)
test_dsl  = true_dataset_dsl(batch_size = cfg.batch_size, mode='test', shuffle_each_epoch=False)

sample_shape = train_dsl.get_sample_shape()
width, height, channels = sample_shape
is_multilabel = train_dsl.is_multilabel()
resize = (width, height)
num_classes = train_dsl.get_num_classes()

noise_dataset_dsl = load_dataset(cfg.noise_dataset) 

count = 1

while True:
    try:
        print "Loading noise data. Attempt {}" .format(count)
        if noise_dataset_dsl == UniformDSL:
            noise_val_dsl = noise_dataset_dsl(batch_size = cfg.batch_size, mode='val', shape=sample_shape, sample_limit=20100, seed=cfg.seed)
        elif noise_dataset_dsl == ImagenetDSL and cfg.num_to_keep is not None:
            noise_val_dsl = noise_dataset_dsl(batch_size = cfg.batch_size, mode='val', num_to_keep=cfg.num_to_keep, start_batch=cfg.subsampling_start_batch, end_batch=cfg.subsampling_end_batch, seed=cfg.seed)
        else:
            noise_val_dsl = noise_dataset_dsl(batch_size = cfg.batch_size, mode='val', seed=cfg.seed)
        break
    except MemoryError as e:
        if count==5:
            raise Exception("Memory error could not be resolved using time delay")
        else:
            print "Loading data failed. Waiting for 5 min.."
            time.sleep(300)        
        count = count + 1

print "Data loaded"        

_, _, noise_channels = noise_val_dsl.get_sample_shape()

normalize_channels = (channels == 1 and noise_channels != 1)

count = 1

while True:
    try:
        print "Loading data. Attempt {}" .format(count)
        if noise_dataset_dsl == UniformDSL:
            noise_train_dsl = noise_dataset_dsl(batch_size = cfg.batch_size, mode='train', shape=sample_shape, sample_limit= 80100, seed=cfg.seed)
            noise_val_dsl = noise_dataset_dsl(batch_size = cfg.batch_size, mode='val', shape=sample_shape, sample_limit=20100, seed=cfg.seed)
        elif noise_dataset_dsl == ImagenetDSL and cfg.num_to_keep is not None:
            noise_train_dsl = noise_dataset_dsl(batch_size = cfg.batch_size, mode='train', resize=resize, normalize_channels=normalize_channels, num_to_keep=cfg.num_to_keep, start_batch=cfg.subsampling_start_batch, end_batch=cfg.subsampling_end_batch, seed=cfg.seed)
            noise_val_dsl = noise_dataset_dsl(batch_size = cfg.batch_size, mode='val', resize=resize, normalize_channels=normalize_channels, num_to_keep=cfg.num_to_keep, start_batch=cfg.subsampling_start_batch, end_batch=cfg.subsampling_end_batch, seed=cfg.seed)
        else:
            noise_train_dsl = noise_dataset_dsl(batch_size = cfg.batch_size, mode='train', resize=resize, normalize_channels=normalize_channels)
            noise_val_dsl = noise_dataset_dsl(batch_size = cfg.batch_size, mode='val', resize=resize, normalize_channels=normalize_channels, seed=cfg.seed)
        break
    except MemoryError as e:
        if count==5:
            raise Exception("Memory error could not be resolved using time delay")
        else:
            print "Loading data failed. Waiting for 5 min.."
            time.sleep(300)        
        count = count + 1

print "Training data loaded"        

source_model_type = load_model(cfg.source_model)
copy_model_type = load_model(cfg.copy_model)

if cfg.train_source_model:
    tf.reset_default_graph()

    true_model = source_model_type(batch_size=cfg.batch_size, height=height, width=width, channels=channels, num_classes=num_classes, multilabel=is_multilabel)
    true_model.print_trainable_parameters()
    true_model.print_arch()

    # Train our ground truth model.
    logging.info("Training source model...")
    t = time.time()
    shutil.rmtree(logdir_true, ignore_errors=True, onerror=None)
    train_model(model=true_model, train_dsl=train_dsl, val_dsl=val_dsl, logdir=logdir_true)
    logging.info("Training source model completed {} min" .format( round((time.time() - t)/60, 2)  ) )

# Set `true_model` to safe mode.
tf.reset_default_graph()

tf.set_random_seed(cfg.seed)
true_model = source_model_type(batch_size=cfg.batch_size, height=height, width=width, channels=channels, num_classes=num_classes, multilabel=is_multilabel, is_training=False)

true_model.print_trainable_parameters()
true_model.print_arch()

print "True Model",
evaluate(model=true_model, dsl=test_dsl, logdir=logdir_true)


with tf.variable_scope("copy_model"):
    test_var = tf.get_variable('foo', shape=(1,5))
    
    if cfg.optimizer == 'adagrad':
        optimizer  = tf.train.AdagradOptimizer(cfg.learning_rate) # None
    elif cfg.optimizer == 'adam':
        optimizer  = tf.train.AdamOptimizer() # None
    else:
        assert cfg.optimizer is None
        optimizer = None
    
    copy_model = copy_model_type(batch_size=cfg.batch_size, height=height, width=width, channels=channels, num_classes=num_classes, multilabel=is_multilabel, fc_layers=[], optimizer=optimizer)
    copy_model.test_var = test_var
    copy_model.print_trainable_parameters()
    copy_model.print_arch()


if cfg.copy_source_model:
    print "deleting the dir {}" .format( logdir_copy )

    shutil.rmtree(logdir_copy, ignore_errors=True, onerror=None)
    
    if cfg.iterative:
        logging.info("Copying source model using iterative approach")
        train_copynet_iter(true_model, copy_model, noise_train_dsl, noise_val_dsl, test_dsl, logdir_true, logdir_copy)
    else:
        raise Exception("not implemented")

