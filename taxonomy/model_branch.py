from collections import OrderedDict
import logging

from neon import NervanaObject
from neon.transforms import CrossEntropyBinary, Logistic
from neon.util.persist import load_obj, save_obj
from neon.layers import Sequential, Activation, Tree
from neon.models.model import Model
import numpy as np

logger = logging.getLogger(__name__)

class TaxonomicBranchModel(Model):

    """
    Taxonomic branch model has cost in last layer so does not use traditional cost.

    Arguments:
        layers: layer container, or a list of layers (that will be containerized)
        name (str): Model name.  Defaults to "model"
        optimizer (Optimizer): Optimizer object which defines the learning rule
                               for updating model parameters (ie DescentMomentum, AdaDelta)
    """

    def __init__(self, layers=[], name="model", optimizer=None):
        super(TaxonomicBranchModel, self).__init__(layers, name, optimizer)

    def _epoch_fit(self, dataset, callbacks):
        """
        Helper function for fit which performs training on a dataset for one epoch.

        Arguments:
            dataset (iterable): Dataset iterator to perform fit on
        """
        #self.cost = CostContainer(None)
        epoch = self.epoch_index
        self.total_cost[:] = 0
        # iterate through minibatches of the dataset
        for mb_idx, (x, t) in enumerate(dataset):

            callbacks.on_minibatch_begin(epoch, mb_idx)

            cost = self.fprop(x)
            # To allow for cost call backs to work
            self.cost.cost = cost
            self.total_cost[:] = self.total_cost + cost

            self.bprop(None)

            self.optimizer.optimize(self.layers_to_optimize, epoch=epoch)

            callbacks.on_minibatch_end(epoch, mb_idx)

        # now we divide total cost by the number of batches,
        # so it was never total cost, but sum of averages
        # across all the minibatches we trained on
        self.total_cost[:] = self.total_cost / dataset.nbatches

    def eval(self, dataset, metric):
        """
        Evaluates a model on a dataset according to an input metric.

        Arguments:
            datasets (iterable): dataset to evaluate on.
            metric (Cost): what function to evaluate dataset on.
        """
        self.initialize(dataset)
        running_error = np.zeros((len(metric.metric_names)), dtype=np.float32)
        nprocessed = 0
        dataset.reset()
        for x, t in dataset:
            if metric.name == 'root_misclass':
                for l in self.layers.layers[:-1]:
                    x = l.fprop(x, inference=True)
                x, t = self.layers.layers[-1].get_root_preds(x, t)
            else:
                x = self.fprop(x, inference=True)

            # This logic is for handling partial batch sizes at the end of the dataset
            bsz = min(dataset.ndata - nprocessed, self.be.bsz)
            running_error += metric(x, t, calcrange=slice(0, bsz)) * bsz
            nprocessed += bsz
        running_error /= nprocessed
        return running_error

""" Cost container to work with minibatch end call back """
class CostContainer():
    def __init__(self, cost):
        self.cost = cost
        self.costfunc = "taxonomic branch cost"

    def get_cost(self, x, t):
        pass