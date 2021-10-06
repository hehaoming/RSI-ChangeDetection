import torch
import torch.nn as nn


class BinaryDiceLoss(nn.Module):
    """二分类版本的Dice Loss"""
    def __init__(self,
                 smooth: int = 1,
                 exponent: int = 1,
                 reduction: str = "mean",
                 loss_weight: float = 1.0,
                 balance_weight: float = 1.0,
                 activation: bool = False) -> None:
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activation = activation
        self.balanced_weight = balance_weight  # TODO 这是这篇paper自定义的权重论文并没有言明！

    def forward(self, pred, target):
        assert self.reduction in ['sum', 'mean'], "reduction is in ['sum', 'mean']"
        if self.activation:
            pred = torch.sigmoid(pred)
        assert pred.size(0) == target.size(0)
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1).to(pred.dtype)
        num = torch.sum(torch.mul(pred, target), dim=1) * 2 + self.smooth
        den = self.balanced_weight * torch.sum(pred.pow(self.exponent), dim=1) + torch.sum(target.pow(self.exponent), dim=1) + self.smooth
        loss = 1 - num / den
        if self.reduction == 'mean':
            loss = loss.mean(dim=0)
        else:
            loss = loss.sum(dim=0)
        return self.loss_weight * loss
