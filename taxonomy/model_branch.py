# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
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
        self.cost = CostContainer(None)
        epoch = self.epoch_index
        self.total_cost[:] = 0
        # iterate through minibatches of the dataset
        for mb_idx, (x, t) in enumerate(dataset):

            callbacks.on_minibatch_begin(epoch, mb_idx)

            cost = self.fprop(x)
            self.cost.cost = cost
            self.total_cost[:] = self.total_cost + cost

            self.bprop(None)

            self.optimizer.optimize(self.layers_to_optimize, epoch=epoch)

            callbacks.on_minibatch_end(epoch, mb_idx)

        # now we divide total cost by the number of batches,
        # so it was never total cost, but sum of averages
        # across all the minibatches we trained on
        self.total_cost[:] = self.total_cost / dataset.nbatches

""" Cost container to work with minibatch end call back """
class CostContainer():
    def __init__(self, cost):
        self.cost = cost
        self.costfunc = "taxonomic branch cost"