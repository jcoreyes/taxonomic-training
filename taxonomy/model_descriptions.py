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

from layer import TaxonomicBranch
from class_taxonomy import ClassTaxonomy
from model_branch import TaxonomicBranchModel
n_class = 555

def create_alexnet():
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
    return layers

def create_branched_alexnet(ctree, img_loader):
    # Replace last 3 layers (Linear, bias, and activation) with Branch Layer
    layers = create_alexnet()[:-4]

    layer_container = {k: Affine(nout=len(v), init=Gaussian(scale=0.01), bias=Constant(-.7),
                          activation=Softmax(), linear_name='branch', bias_name='branch')
                          for k, v in ctree.internalid_to_childrenid.items()}

    cost_container = {k: GeneralizedCost(costfunc=CrossEntropyMulti())
                         for k in ctree.internalid_to_childrenid.keys()}

    branch = TaxonomicBranch(layer_container, cost_container, ctree, img_loader)
    layers.append(branch)
    return layers

def create_model(model_type, freeze, dataset_dir, img_loader):
    if model_type == 'alexnet':
        layers = create_alexnet()
        cost = GeneralizedCost(costfunc=CrossEntropyMulti())
        model = Model(layers=layers)
    elif model_type == 'branched':
        ctree = ClassTaxonomy('Aves', 'taxonomy_dict.p', dataset_dir)
        layers = create_branched_alexnet(ctree, img_loader)
        cost = None
        model = TaxonomicBranchModel(layers=layers)
    else:
        raise NotImplementedError(model_type + " has not been implemented")

    if freeze > 0:
        pass


    return model, cost