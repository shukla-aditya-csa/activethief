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

import tensorflow as tf

flags = tf.app.flags

# Parameters
# ==================================================

# Data loading params

# Model Hyperparameters
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")
tf.flags.DEFINE_float("l2_reg_lambda", 0.001, "L2 regularization lambda")
tf.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.flags.DEFINE_string("optimizer", "adam", "Custom optimizer")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 50)")
tf.flags.DEFINE_integer("num_epochs", 1000, "Number of training epochs (default: 100)")
tf.flags.DEFINE_integer("copy_num_epochs", 1000, "Number of training epochs (default: 1000)")
tf.flags.DEFINE_integer("evaluate_every", 1, "Evaluate model on dev set after this many steps (default: 1)")
tf.flags.DEFINE_integer("early_stop_tolerance", 20, "Early stop (default: 10 evaluations)")
tf.flags.DEFINE_integer("copy_evaluate_every", 1, "Evaluate copy model after this many epochs")
tf.flags.DEFINE_boolean("copy_one_hot", False, "Copy using one hot")
tf.flags.DEFINE_boolean("copy_source_model", False, "copy source model")
tf.flags.DEFINE_integer("grad0_prate", 10, "grad0_prate")

# GPU Parameters
tf.flags.DEFINE_boolean("allow_gpu_growth", True, "Allow gpu growth")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

tf.flags.DEFINE_string("source_model", None,  "Source model type(eg DeepCNN/DNN) to copy")
tf.flags.DEFINE_string("copy_model", None,  "Copy model type(eg DeepCNN/DNN)")
tf.flags.DEFINE_string("true_dataset", None,  "Source model will be trained on this")
tf.flags.DEFINE_string("noise_dataset", None, "Source model will be copied using this")
tf.flags.DEFINE_string("sampling_method", "random", "sampling method")

tf.flags.DEFINE_integer("phase1_fac", 5, "Multiple of samples to use in Phase 1")
tf.flags.DEFINE_integer("phase2_fac", 10, "Multiple of samples to use in Phase 2")

tf.flags.DEFINE_integer("phase1_size", 20000, "Multiple of samples to use in Phase 1")
tf.flags.DEFINE_integer("phase2_size", 10000, "Multiple of samples to use in Phase 2")

tf.flags.DEFINE_boolean("iterative", True, "Use the iterative method to copy the source model")

tf.flags.DEFINE_integer("subsampling_start_batch", 1, "Start Batch of imagenet to use for subsampling experiments")
tf.flags.DEFINE_integer("subsampling_end_batch", 1, "End Batch of imagenet to use for subsampling experiments")

tf.flags.DEFINE_integer("num_to_keep", None, "Number of samples to make use of for imagenet")

tf.flags.DEFINE_integer("initial_seed", None, "intial seed")
tf.flags.DEFINE_integer("num_iter", None, "num of iterations")
tf.flags.DEFINE_integer("k", None, "add queries")

tf.flags.DEFINE_integer("seed", 1337, "seed for RNGs")

tf.flags.DEFINE_integer("val_size", 1000, "validation size")

# Hack for dealing with Jupyter Notebook
tf.flags.DEFINE_string("f", "f", "f")

# Parameter used by `generic_runner.py`
tf.flags.DEFINE_boolean("train_source_model", False, "Train the source model")

# Directories
tf.flags.DEFINE_string("dataset_dir", "dataset_dir", "Directory for datasets")

cfg = tf.app.flags.FLAGS

config                          = tf.ConfigProto()
config.gpu_options.allow_growth = cfg.allow_gpu_growth
config.log_device_placement     = cfg.log_device_placement   
config.allow_soft_placement     = cfg.allow_soft_placement   
