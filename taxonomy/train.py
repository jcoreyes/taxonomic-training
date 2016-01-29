"""
Trains various models.
Example command:
python train.py -eval 1 --model_type alexnet --freeze 2 -w ~/nervana/data/NABirds_batchs
 -b gpu -i 0 -e 40 --dataset_dir ~/NABirds --model_file alexnet.p -vvvv
"""

from neon.util.argparser import NeonArgparser
from neon.initializers import Constant, Gaussian
from neon.layers import Conv, Dropout, Pooling, GeneralizedCost, Affine
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, TopKMisclassification
from neon.models import Model
from neon.data import ImageLoader
from neon.callbacks.callbacks import Callbacks

from model_descriptions import create_model

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
parser.add_argument('--model_type', help='Name of model', required=True, choices=['alexnet', 'vgg'])
parser.add_argument('--model_tree', help='Whether or not to train tree of classifiers',
                    default=False, type=bool)
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

model, cost, opt = create_model(args.model_type, args.model_tree, args.freeze, args.dataset_dir,
                                args.model_file, train)

# configure callbacks
valmetric = TopKMisclassification(k=5)
valmetric.name = 'root_misclass'
# If freezing layers, load model in create_model
if args.freeze > 0:
    args.callback_args['model_file'] = None
callbacks = Callbacks(model, train, eval_set=test, metric=valmetric, **args.callback_args)
model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
