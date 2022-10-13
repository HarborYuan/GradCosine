import torch.nn as nn
from mmcv.cnn.bricks.registry import NORM_LAYERS


class NoNorm(nn.Identity):
    pass


NORM_LAYERS.register_module('ID', module=NoNorm)
