from neon.layers.container import LayerContainer
from neon.layers.layer import interpret_in_shape, Layer

class TaxonomicBranch(LayerContainer):

    """
    A layer which creates a classifier for each internal node in the class taxonomy.
    Training: For a given leaf node, only the internal nodes that leaf node falls under will be
    propagated.

    Arguments:
        nout (int, tuple): Desired size or shape of layer output
        init (Initializer, optional): Initializer object to use for
            initializing layer weights
        name (str, optional): Layer name. Defaults to "LinearLayer"
    """

    def __init__(self, layer_container, cost_container, ctree, img_loader, name="LinearLayer"):
        super(TaxonomicBranch, self).__init__(name)
        self.nout = len(layer_container)
        self.inputs = None
        self.layers = layer_container
        self.costs = cost_container
        self.ctree = ctree
        self.has_params = False
        self.img_loader = img_loader

    @property
    def layers_to_optimize(self):
        lto = []
        for l in self.layers.values():
            if isinstance(l, list):
                for li in l:
                    if li.has_params:
                        lto.append(li)
            elif l.has_params:
                lto.append(l)
        return lto

    def _do_fprop(self, obj, inputs, inference=False):
        assert isinstance(obj, list)
        tmp_output = inputs
        for layer in obj:
            tmp_output = layer.fprop(tmp_output, inference)
        return tmp_output

    def _do_bprop(self, obj, error):
        assert isinstance(obj, list)
        for layer in reversed(obj):

            error = layer.bprop(error)
        return obj[0].deltas

    def _do_configure(self, obj, prev_input):
        for l in obj:
            prev_input = l.configure(prev_input)
        return prev_input

    def _do_allocate(self, obj):
        for l in obj:
            l.allocate()

    def _do_set_deltas(self, obj, delta_buffer):
        for l in obj:
            l.set_deltas(delta_buffer)

    def set_deltas(self, delta_buffer):
        old_bsz = self.be.bsz
        self.be.bsz = 1
        for v in self.layers.values():
            self._do_set_deltas(v, delta_buffer)
        self.be.bsz = old_bsz

    def allocate(self, shared_outputs=None, shared_deltas=None):
        self.deltas = self.be.iobuf(self.in_shape) if shared_deltas is None else shared_deltas
        self.total_deltas = self.be.zeros(self.deltas.shape)
        self.inputs = [self.be.zeros((self.nin, 1)) for _ in range(self.be.bsz)]
        old_bsz = self.be.bsz
        self.be.bsz = 1
        self.targets = {}
        for k, v in self.layers.items():
            self._do_allocate(v)
            self.targets[k] = self.be.zeros(v[-1].out_shape)
        self.be.bsz = old_bsz

    def configure(self, in_obj):
        assert isinstance(in_obj, Layer)
        self.prev_layer = in_obj
        self.in_shape = in_obj.out_shape
        (self.nin, self.nsteps) = interpret_in_shape(self.in_shape)
        self.out_shape = self.in_shape

        old_bsz = self.be.bsz
        self.be.bsz = 1
        for k in self.ctree.internalid_to_childrenid.keys():
            prev_input = self._do_configure(self.layers[k], self.prev_layer)
            self.costs[k].initialize(prev_input)

        self.be.bsz = old_bsz

        return self

    def fprop(self, inputs, inference=False):
        if inference:
            return self._fprop_inference(inputs)
        else:
            return self._fprop(inputs)

    def get_outputs(self, inputs):
        preds = []
        all_probs = []
        for i in range(self.be.bsz):
            # Copy column of inputs which is one data point
            self.inputs[i][:] = inputs[:, i]
            pred = [] # list of (internal_id, prob)
            curr_id = self.ctree.root
            # Continue predicting till we get to leaf node
            prev_prob = 1.0
            single_probs = []
            while True:
                x = self._do_fprop(self.layers[curr_id], self.inputs[i]).get()
                single_probs.append((curr_id, x))
                curr_idx = x.argmax()
                prob = prev_prob * x[curr_idx, 0]
                curr_id = self.ctree.internalid_to_childrenid[curr_id][curr_idx]
                pred.append((curr_id, prob))
                prev_prob = prob
                if curr_id in self.ctree.leafid_to_internallabels:
                    break
            preds.append(pred)
            all_probs.append(single_probs)
        return all_probs

    def _fprop_inference(self, inputs):
        preds = []
        for i in range(self.be.bsz):
            # Copy column of inputs which is one data point
            self.inputs[i][:] = inputs[:, i]
            pred = [] # list of (internal_id, prob)
            curr_id = self.ctree.root
            # Continue predicting till we get to leaf node
            prev_prob = 1.0
            while True:
                x = self._do_fprop(self.layers[curr_id], self.inputs[i]).get()
                curr_idx = x.argmax()
                prob = prev_prob * x[curr_idx, 0]
                curr_id = self.ctree.internalid_to_childrenid[curr_id][curr_idx]
                pred.append((curr_id, prob))
                prev_prob = prob
                if curr_id in self.ctree.leafid_to_internallabels:
                    break
            preds.append(pred)
        return preds

    def _fprop(self, inputs):
        # Get lead node label idxs from img loader
        temp_lbl = self.img_loader.labels[self.img_loader.idx].get()[0]
        self.total_deltas[:] = 0
        self.total_cost = self.be.zeros((1, 1))
        self.cost = self.be.zeros((1, 1))
        # Iterate over each data point in batch
        for i in range(self.be.bsz):
            label_idx = temp_lbl[i]
            label_id = self.ctree.labelidx_to_leafid[label_idx]
            self.inputs[i][:] = inputs[:, i]
            # Fprop all internal nodes label idx falls under
            self.cost[:] = 0
            self.deltas[:] = 0
            for internalid, internallbl in self.ctree.leafid_to_internallabels[label_id]:
                # Convert label idx to 1 hot encoding
                targets = self.targets[internalid]
                targets[:] = 0
                targets[internallbl] = 1
                x = self._do_fprop(self.layers[internalid],  self.inputs[i])

                cost = self.costs[internalid].get_cost(x, targets)
                self.cost[:] = self.cost + cost
                delta = self.costs[internalid].get_errors(x, targets)
                # Accumulate gradients
                self.deltas[:, i] = self.deltas[:, i] + self._do_bprop(self.layers[internalid], delta)

            self.total_deltas[:] = self.total_deltas + self.deltas #/ len(self.ctree.leafid_to_parentsid[label_id])

            self.total_cost[:] = self.total_cost + self.cost #/ len(self.ctree.leafid_to_parentsid[label_id])
        return self.total_cost

    def bprop(self, error):
        return self.total_deltas
