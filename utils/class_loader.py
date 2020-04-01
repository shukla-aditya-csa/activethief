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

def load_dataset(dataset):
    dsl = None
    
    if dataset == 'mnist':
        from dsl.mnist_dsl import MNISTDSL
        dsl = MNISTDSL

    elif dataset == 'gtsr':
        from dsl.gtsr_dsl import GTSRDSL
        dsl = GTSRDSL

    elif dataset == 'cifar':
        from dsl.cifar_dsl import CifarDSL
        dsl = CifarDSL

    elif dataset == 'imagenet':
        from dsl.imagenet_dsl import ImagenetDSL
        dsl = ImagenetDSL
    
    elif dataset == 'uniform':
        from dsl.uniform_dsl import UniformDSL
        dsl = UniformDSL
    
    else:
        raise Exception("Dataset {} could not be loaded" .format( dataset ) ) 
    
    return dsl


def load_model(model_class):
    model = None
    
    if model_class.startswith('cnn_'):
        from models.cnn import CNN
        blocks, convs_in_block = model_class.strip().split('_')[1:]
        blocks, convs_in_block = int(blocks), int(convs_in_block)
    
        class CNNWrapper(CNN):
            def __init__(self, *args, **kwargs):
                super(CNNWrapper, self).__init__(*args, convs_in_block=convs_in_block, num_filters=[32, 64, 128, 256][:blocks], **kwargs)

        return CNNWrapper
    
    else:
        raise Exception("Model {} could not be loaded" .format( model_class ) ) 
        
    return model
