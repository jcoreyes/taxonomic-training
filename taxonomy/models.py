#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Runs one epoch of Alexnet on imagenet data.
For running complete alexnet
alexnet.py -e 90 -eval 1 -s <save-path> -w <path-to-saved-batches>
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