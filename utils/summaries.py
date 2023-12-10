import os
import torch
import numpy as np
import cv2
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer