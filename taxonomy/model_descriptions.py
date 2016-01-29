"""
Create various models.
"""
import os

from neon.util.argparser import NeonArgparser
from neon.initializers import Constant, Gaussian, GlorotUniform
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

def create_vgg_layers(nclass):
    cost_scale = 1.
    VGG = 'B'
    use_batch_norm = True
    biases = None if use_batch_norm else Constant(0)
    init1 = GlorotUniform()
    relu = Rectlin()
    common_params = dict(init=init1, activation=Rectlin(), batch_norm=use_batch_norm, bias=biases)
    conv_params = dict(padding=1, **common_params)

    # Set up the model layers, using 3x3 conv stacks with different feature map sizes
    layers = []

    for nofm in [64, 128, 256, 512, 512]:
        layers.append(Conv((3, 3, nofm), **conv_params))
        layers.append(Conv((3, 3, nofm), **conv_params))
        if nofm > 128:
            if VGG in ('D', 'E'):
                layers.append(Conv((3, 3, nofm), **conv_params))
            if VGG == 'E':
                layers.append(Conv((3, 3, nofm), **conv_params))
        layers.append(Pooling(3, strides=2))

    layers.append(Affine(nout=4096, **common_params))
    layers.append(Dropout(keep=0.5))
    layers.append(Affine(nout=4096, **common_params))
    layers.append(Dropout(keep=0.5))
    layers.append(Affine(nout=nclass, init=init1, bias=Constant(0), activation=Softmax()))
    return layers

def create_branched(layer_func, ctree, img_loader):
    # Replace last layer with Branch Layer
    layers = layer_func(img_loader.nclass)[:-1]
    #assert isinstance(layers[-1], Dropout)

    layer_container = {k: TaxonomicAffine(nout=len(v), init=Gaussian(scale=0.01), bias=Constant(-7),
                          activation=Softmax(), linear_name='branch', bias_name='branch')
                          for k, v in ctree.internalid_to_childrenid.items()}

    cost_container = {k: GeneralizedCost(costfunc=CrossEntropyMulti())
                         for k in ctree.internalid_to_childrenid.keys()}

    branch = TaxonomicBranch(layer_container, cost_container, ctree, img_loader)
    layers.append(branch)
    return layers

def create_alexnet_opt():
    # drop weights LR by 1/250**(1/3) at epochs (23, 45, 66), drop bias LR by 1/10 at epoch 45
    weight_sched = Schedule([22, 44, 65], (1/250.)**(1/3.))
    opt_gdm = GradientDescentMomentum(0.01, 0.9, wdecay=0.0005, schedule=weight_sched,
                                      stochastic_round=args.rounding)
    opt_biases = GradientDescentMomentum(0.02, 0.9, schedule=Schedule([44], 0.1),
                                         stochastic_round=args.rounding)
    opt = MultiOptimizer({'default': opt_gdm, 'Bias': opt_biases})
    return opt

def create_vgg_opt():
    weight_sched = Schedule(range(14, 75, 15), 0.1)
    opt_gdm = GradientDescentMomentum(0.01, 0.9, wdecay=0.0005, schedule=weight_sched)
    opt_biases = GradientDescentMomentum(0.02, 0.9, schedule=weight_sched)
    opt = MultiOptimizer({'default': opt_gdm, 'Bias': opt_biases})
    return opt

def create_model(model_type, model_tree, freeze, dataset_dir, model_file, img_loader):
    cost = GeneralizedCost(costfunc=CrossEntropyMulti())

    if model_type == 'alexnet':
        opt = create_alexnet_opt()
        layer_func = create_alexnet_layers
    elif model_type == 'vgg':
        opt = create_vgg_opt()
        layer_func = create_vgg_layers
    else:
        raise NotImplementedError(model_type + " has not been implemented")

    if model_tree:
        ctree = ClassTaxonomy('Aves', 'taxonomy_dict.p', dataset_dir)
        layers = created_branched(layer_func, ctree, img_loader)
        model = TaxonomicBranchModel(layers=layers)
    else:
        layers = layer_func(img_loader.nclass)
        model = Model(layers=layers)

    if freeze > 0:
        saved_model = Model(layers=layer_func(1000))
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

    return model, cost, opt