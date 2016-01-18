"""
Process macro batches of data in a pipelined fashion.
"""

import logging

from glob import glob
import functools
import gzip
from multiprocessing import Pool
import numpy as np
import os
import tarfile
import struct
from PIL import Image as PILImage
from neon.util.compat import range, StringIO
from neon.util.persist import load_obj, save_obj
from neon.util.argparser import NeonArgparser

"""
Example command:
python batch_writer.py --data_dir ~/nervana/data/NABirds_batchs --dataset_dir ~/NABirds
"""

# NOTE: We have to leave this helper function out of the class to use multiprocess pool.map
def proc_img(target_size, squarecrop, is_string=False, imgfile=None):
    imgfile = StringIO(imgfile) if is_string else imgfile
    im = PILImage.open(imgfile)

    scale_factor = target_size / np.float32(min(im.size))
    if scale_factor == 1 and im.size[0] == im.size[1] and is_string is False:
        return np.fromfile(imgfile, dtype=np.uint8)

    (wnew, hnew) = map(lambda x: int(round(scale_factor * x)), im.size)
    if scale_factor != 1:
        filt = PILImage.BICUBIC if scale_factor > 1 else PILImage.ANTIALIAS
        im = im.resize((wnew, hnew), filt)

    if squarecrop is True:
        (cx, cy) = map(lambda x: (x - target_size) // 2, (wnew, hnew))
        im = im.crop((cx, cy, cx+target_size, cy+target_size))

    buf = StringIO()
    im.save(buf, format='JPEG', subsampling=0, quality=95)
    return buf.getvalue()


class BirdsBatchWriter(object):

    def __init__(self, out_dir, dataset_dir, target_size=256, squarecrop=True,
                 class_samples_max=None, file_pattern='*.jpg', macro_size=3072):
        np.random.seed(0)
        self.out_dir = os.path.expanduser(out_dir)
        self.dataset_dir = os.path.expanduser(dataset_dir)
        self.macro_size = macro_size
        self.num_workers = 8
        self.target_size = target_size
        self.squarecrop = squarecrop
        self.file_pattern = file_pattern
        self.class_samples_max = class_samples_max
        self.train_file = os.path.join(self.out_dir, 'train_file.csv.gz')
        self.val_file = os.path.join(self.out_dir, 'val_file.csv.gz')
        self.test_file = os.path.join(self.out_dir, 'test_file.csv.gz')
        self.meta_file = os.path.join(self.out_dir, 'dataset_cache.pkl')
        self.global_mean = np.array([0, 0, 0]).reshape((3, 1))
        self.batch_prefix = 'data_batch_'

    def write_csv_files(self):
        # image_idx : split
        split_file = np.loadtxt(os.path.join(self.dataset_dir, 'train_test_val_split.txt'), delimiter=' ')
        # image_idx : file_name
        images_key = np.loadtxt(os.path.join(self.dataset_dir, 'images.txt'), delimiter=' ', dtype='str')
        # image_idx : class_idx
        image_class_labels = np.loadtxt(os.path.join(self.dataset_dir, 'image_class_labels.txt'), delimiter=' ') - 1
        # image_idx : class_name
        self.label_names = [x.strip()[x.index(' ')+1:] for x in open(os.path.join(self.dataset_dir, 'classes.txt'), 'r').readlines()]

        self.nclass = len(self.label_names)
        self.label_dict = dict(zip(self.label_names, range(self.nclass)))

        # Get the labels as the subdirs
        tlines = []
        tslines = []
        vlines = []
        splits = {0:tlines, 1:tslines, 2:vlines}
        for i in xrange(split_file.shape[0]):
            if self.class_samples_max and i > self.class_samples_max:
                break
            full_filename = os.path.join(os.path.join(self.dataset_dir, 'images'), images_key[i, 1])
            splits[split_file[i, 1]].append((full_filename, image_class_labels[i, 1]))

        np.random.shuffle(tlines)

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        for ff, ll in zip([self.train_file, self.test_file, self.val_file], [tlines, tslines, vlines]):
            with gzip.open(ff, 'wb') as f:
                f.write('filename,l_id\n')
                for tup in ll:
                    f.write('{},{}\n'.format(*tup))

        self.train_nrec = len(tlines)
        self.ntrain = -(-self.train_nrec // self.macro_size)
        self.train_start = 0

        self.test_nrec = len(tslines)
        self.ntest = -(-self.test_nrec // self.macro_size)
        self.test_start = self.train_start + self.ntrain + 1

        self.val_nrec = len(vlines)
        self.nval = -(-self.val_nrec // self.macro_size)
        self.val_start = self.test_start + self.ntest + 1

    def parse_file_list(self, infile):
        lines = np.loadtxt(infile, delimiter=',', skiprows=1, dtype={'names': ('fname', 'l_id'),
                                                                     'formats': (object, 'i4')})
        imfiles = [l[0] for l in lines]
        labels = {'l_id': [l[1] for l in lines]}
        self.nclass = {'l_id': (max(labels['l_id']) + 1)}
        return imfiles, labels

    def write_batches(self, name, offset, labels, imfiles):
        pool = Pool(processes=self.num_workers)
        npts = -(-len(imfiles) // self.macro_size)
        starts = [i * self.macro_size for i in range(npts)]
        is_tar = isinstance(imfiles[0], tarfile.ExFileObject)
        proc_img_func = functools.partial(proc_img, self.target_size, self.squarecrop, is_tar)
        imfiles = [imfiles[s:s + self.macro_size] for s in starts]
        labels = [{k: v[s:s + self.macro_size] for k, v in labels.iteritems()} for s in starts]

        print("Writing %s batches..." % (name))
        for i, jpeg_file_batch in enumerate(imfiles):
            if is_tar:
                jpeg_file_batch = [j.read() for j in jpeg_file_batch]
            jpeg_strings = pool.map(proc_img_func, jpeg_file_batch)
            bfile = os.path.join(self.out_dir, '%s%d' % (self.batch_prefix, offset + i))
            self.write_binary(jpeg_strings, labels[i], bfile)
            print("Writing batch %d" % (i))
        pool.close()

    def write_binary(self, jpegs, labels, ofname):
        num_imgs = len(jpegs)
        keylist = ['l_id']
        with open(ofname, 'wb') as f:
            f.write(struct.pack('I', num_imgs))
            f.write(struct.pack('I', len(keylist)))

            for key in keylist:
                ksz = len(key)
                f.write(struct.pack('L' + 'B' * ksz, ksz, *bytearray(key)))
                f.write(struct.pack('I' * num_imgs, *labels[key]))

            for i in range(num_imgs):
                jsz = len(jpegs[i])
                bin = struct.pack('I' + 'B' * jsz, jsz, *bytearray(jpegs[i]))
                f.write(bin)

    def save_meta(self):
        save_obj({'ntrain': self.ntrain,
                  'nval': self.nval,
                  'train_start': self.train_start,
                  'test_start': self.test_start,
                  'val_start': self.val_start,
                  'macro_size': self.macro_size,
                  'batch_prefix': self.batch_prefix,
                  'global_mean': self.global_mean,
                  'label_dict': self.label_dict,
                  'label_names': self.label_names,
                  'val_nrec': self.val_nrec,
                  'train_nrec': self.train_nrec,
                  'img_size': self.target_size,
                  'nclass': self.nclass}, self.meta_file)

    def run(self):
        self.write_csv_files()
        namelist = ['train', 'test', 'validation']
        filelist = [self.train_file, self.test_file, self.val_file]
        startlist = [self.train_start, self.test_start, self.val_start]
        for sname, fname, start in zip(namelist, filelist, startlist):
            print("%s %s %s" % (sname, fname, start))
            if fname is not None and os.path.exists(fname):
                imgs, labels = self.parse_file_list(fname)
                self.write_batches(sname, start, labels, imgs)
            else:
                print("Skipping %s, file missing" % (sname))
        self.save_meta()

if __name__ == "__main__":
    parser = NeonArgparser(__doc__)
    parser.add_argument('--dataset_dir', help='Directory containing images folder and label text files', required=True)
    parser.add_argument('--target_size', type=int, default=256,
                        help='Size in pixels to scale images (Must be 256 for i1k dataset)')
    parser.add_argument('--macro_size', type=int, default=2000, help='Images per processed batch')
    parser.add_argument('--class_samples_max', help='Only process smaller amount of images', type=int, default=None)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    # Supply dataset location
    # out_dir defaults to ~/nervana/data
    bw = BirdsBatchWriter(out_dir=args.data_dir, dataset_dir=args.dataset_dir,
                     target_size=args.target_size, macro_size=args.macro_size,
                     class_samples_max=args.class_samples_max)

    bw.run()
