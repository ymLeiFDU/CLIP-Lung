
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init

from shutil import copyfile



def save_args(ckpt_dir, args):
    # -----------------------------------
    # save running files (.py, .sh, etc.)
    # -----------------------------------
    if not os.path.exists('ckpt_dir'):
        os.mkdir(ckpt_dir)
    if not os.path.exists(ckpt_dir + '/trained_models'):
        os.mkdir(ckpt_dir + '/trained_models')

    os.mkdir(ckpt_dir + '/trained_models/models')
    os.mkdir(ckpt_dir + '/trained_models/utils')
    os.mkdir(ckpt_dir + '/results')
    os.mkdir(ckpt_dir + '/configs')
    os.mkdir(ckpt_dir + '/data')


    print('Saving files ...')
    for f in args:
        print(f, ckpt_dir + '/' + f)
        if 'utils' in f or 'models' in f:
            copyfile(f, ckpt_dir + '/trained_models/' + f )
        else:
            copyfile(f, ckpt_dir + '/' + f)


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __def__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        if not '...' in data:
            self.file.write(data)

        self.stdout.write(data)
    
    def flush(self):
        self.file.flush()



































