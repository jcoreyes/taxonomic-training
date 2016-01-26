"""
Create various models.
"""
import os

from neon.util.argparser import NeonArgparser
from neon.initializers import Constant, Gaussian
from neon.layers import Conv, Dropout, Pooling, GeneralizedCost, Affine
from neon.optimizers import GradientDescentMomentum, MultiOptimizer, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, TopKMisclassification
from neon.models import Model
from neon.data import ImageLoader
from neon.callbacks.callbacks import Callbacks

from layer import TaxonomicBranch, TaxonomicAffine, FreezeSequential
from class_taxonomy import ClassTaxonomy
from model_branch import TaxonomicBranchModel

def create_alexnet_layers(nclass):
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
                      Affine(nout=nclass, init=Gaussian(scale=0.01), bias=Constant(-7), activation=Softmax())]
    return layers

def create_branched_alexnet(ctree, img_loader):
    # Replace last layer with Branch Layer
    layers = create_alexnet_layers(img_loader.nclass)[:-1]
    assert isinstance(layers[-1], Dropout)

    layer_container = {k: TaxonomicAffine(nout=len(v), init=Gaussian(scale=0.01), bias=Constant(-.7),
                          activation=Softmax(), linear_name='branch', bias_name='branch')
                          for k, v in ctree.internalid_to_childrenid.items()}

    cost_container = {k: GeneralizedCost(costfunc=CrossEntropyMulti())
                         for k in ctree.internalid_to_childrenid.keys()}

    branch = TaxonomicBranch(layer_container, cost_container, ctree, img_loader)
    layers.append(branch)
    return layers

def create_model(model_type, freeze, dataset_dir, model_file, img_loader):
    if model_type == 'alexnet':
        cost = GeneralizedCost(costfunc=CrossEntropyMulti())
        layers = create_alexnet_layers(img_loader.nclass)
        model = Model(layers=layers)
    elif model_type == 'branched':
        ctree = ClassTaxonomy('Aves', 'taxonomy_dict.p', dataset_dir)
        layers = create_branched_alexnet(ctree, img_loader)
        cost = GeneralizedCost(costfunc=CrossEntropyMulti())
        model = TaxonomicBranchModel(layers=layers)
    else:
        raise NotImplementedError(model_type + " has not been implemented")

    if freeze > 0:
        saved_model = Model(layers=create_alexnet_layers(1000))
        saved_model.load_params(model_file)
        model.initialize(img_loader)
        model.initialized = False
        saved_lto = saved_model.layers.layers_to_optimize
        model_lto = model.layers.layers_to_optimize
        keep_length = len(saved_lto) - freeze * 2

        for i in range(len(saved_lto))[:keep_length]:
            model_lto[i].W[:] = saved_lto[i].W
            model_lto[i].optimize = False
        for i in range(len(model_lto))[keep_length:]:
            model_lto[i].optimize = True

        model.layers = FreezeSequential(layers)
        model.layers_to_optimize = model.layers.layers_to_optimize

    return model, cost