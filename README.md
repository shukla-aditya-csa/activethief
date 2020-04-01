# Getting started

ActiveThief runs on Python 2.7 using TensorFlow 1.x. The recommended Python 2 packages to install can be found in `requirements.txt`.

To download the required datasets, first run:

    python2 download_datasets.py

Then, to preprocess the ImageNet data, you must first run the following script in Python 3:

    python3 preprocess_imagenet.py

Following this, the repository is ready for use. Here is a sample command:

    python2 generic_program.py --source model cnn_3_2 --copy_model cnn_3_2 --true_dataset mnist --noise_dataset imagenet --initial_seed 200 --k 100 --num_iter 10 --train_source_model --copy_source_model --sampling_method kcenter

Where the architecture of the model must be specified as `cnn_x_y` where `x` is the number of convolution blocks, and `y` is the number of convolutions in each block. The choices of datasets available are `mnist`, `cifar` and `gtsr` for the true dataset (aka the **secret dataset**) and `uniform` and `imagenet` for the noise dataset (aka the **thief dataset**). The query budget is specified as `initial_seed + k * num_iter`, which makes the algorithm run in `num_iter+1` iterations. The first iteration picks `initial_seed` samples, whereas in each one of the subsequent iterations, `k` samples are chosen in accordance with the `sampling_method` (aka the **subset selection strategy**). Choices of subset selection strategy available are `random`, `uncertainty`, `adversarial`, `kcenter` and `adversarial-kcenter`. By default, `random` is used.

# License

ActiveThief is available under an MIT License. Parts of the codebase is based on code from other repositories, also under an MIT license.  
Please see the LICENSE file, and the inline license included in each code file for more details.

# Reference

**ActiveThief: Model Extraction Using Active Learning and Unannotated Public Data**  
Soham Pal, Yash Gupta, Aditya Shukla, Aditya Kanade, Shirish Shevade, Vinod Ganapathy.  
_AAAI Conference on Artificial Intelligence (AAAI), 2020_



