"""
Create various models.
"""

from neon.util.argparser import NeonArgparser
from neon.initializers import Constant, Gaussian
from neon.layers import Conv, Dropout, Pooling, GeneralizedCost, Affine
from neon.optimizers import GradientDescentMomentum, MultiOptimizer, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, TopKMisclassification
from neon.models import Model
from neon.data import ImageLoader
from neon.callbacks.callbacks import Callbacks

n_class = 555

def create_model(model_type, freeze):
    if model_type == 'alexnet':
        layers = [Conv((11, 11, 64), init=Gaussian(scale=0.01), bias=Constant(0), activation=Rectlin(),
                       padding=3, strides=4),
                  Pooling(3, strides=2),
                  Conv((5, 5, 192), init=Gaussian(scale=0.01), bias=Constant(1), activation=Rectlin(),
                       padding=2),
                  Pooling(3, strides=2),
                  Conv((3, 3, 384), init=Gaussian(scale=0.03), bias=Constant(0), activation=Rectlin(),
                       padding=1),
                  Conv((3, 3, 256), init=Gaussian(scale=0.03), bias=Constant(1), activation=Rectlin(),
                       padding=1),
                  Conv((3, 3, 256), init=Gaussian(scale=0.03), bias=Constant(1), activation=Rectlin(),
                       padding=1),
                  Pooling(3, strides=2),
                  Affine(nout=4096, init=Gaussian(scale=0.01), bias=Constant(1), activation=Rectlin()),
                  Dropout(keep=0.5),
                  Affine(nout=4096, init=Gaussian(scale=0.01), bias=Constant(1), activation=Rectlin()),
                  Dropout(keep=0.5),
                  Affine(nout=n_class, init=Gaussian(scale=0.01), bias=Constant(-7), activation=Softmax())]
    else:
        raise NotImplementedError(model_type + " has not been implemented")

    model = Model(layers=layers)
    if freeze > 0:
        pass

    return model