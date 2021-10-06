import torch.nn as nn

from .binary_crossentropy import *
from .dice_loss import *


class HybridLoss(nn.Module):

    def __init__(self):
        super(HybridLoss, self).__init__()
        self.dice = BinaryDiceLoss()
        self.bce = BalancedBinaryCrossEntropy()

    def forward(self, pred, target):
        return self.dice(pred, target) + self.bce(pred, target)
