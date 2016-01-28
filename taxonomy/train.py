"""
Trains various models.
Example command:
python train.py -eval 1 --model_type alexnet --freeze 2 -w ~/nervana/data/NABirds_batchs
 -b gpu -i 0 -e 40 --dataset_dir ~/NABirds --model_file alexnet.p -vvvv
"""

from neon.util.argparser import NeonArgparser
from neon.initializers import Constant, Gaussian
from neon.layers import Conv, Dropout, Pooling, GeneralizedCost, Affine
from neon.optimizers import GradientDescentMomentum, MultiOptimizer, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, TopKMisclassification
from neon.models import Model
from neon.data import ImageLoader
from neon.callbacks.callbacks import Callbacks

import model_descriptions

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
parser.add_argument('--model_type', help='Name of model', required=True, choices=['alexnet', 'branched'])
parser.add_argument('--freeze', type=int, help='Layers to freeze starting from end', default=0)
parser.add_argument('--dataset_dir', help='Directory containing images folder and label text files')
args = parser.parse_args()

# setup data provider
train_set_options = dict(repo_dir=args.data_dir,
                       inner_size=224,
                       dtype=args.datatype,
                       subset_pct=100)
test_set_options = dict(repo_dir=args.data_dir,
                       inner_size=224,
                       dtype=args.datatype,
                       subset_pct=20)

train = ImageLoader(set_name='train', **train_set_options)
test = ImageLoader(set_name='train', do_transforms=False, **test_set_options)

model, cost = model_descriptions.create_model(args.model_type, args.freeze, args.dataset_dir,
                                              args.model_file, train)

# drop weights LR by 1/250**(1/3) at epochs (23, 45, 66), drop bias LR by 1/10 at epoch 45
weight_sched = Schedule([22, 44, 65], (1/250.)**(1/3.))
opt_gdm = GradientDescentMomentum(0.01, 0.9, wdecay=0.0005, schedule=weight_sched,
                                  stochastic_round=args.rounding)
opt_biases = GradientDescentMomentum(0.02, 0.9, schedule=Schedule([44], 0.1),
                                     stochastic_round=args.rounding)
opt = MultiOptimizer({'default': opt_gdm, 'Bias': opt_biases})

# configure callbacks
valmetric = TopKMisclassification(k=5)
valmetric.name = 'root_misclass'
# If freezing layers, load model in create_model
if args.freeze > 0:
    args.callback_args['model_file'] = None
callbacks = Callbacks(model, train, eval_set=test, metric=valmetric, **args.callback_args)
model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
