import torch
import torch.nn as nn


class MetricRe(nn.Module):
    def __init__(self, device):
        super(MetricRe, self).__init__()
        self.device = device
        self.tp = torch.tensor([0], dtype=torch.float, device=self.device)
        self.fp = torch.tensor([0], dtype=torch.float, device=self.device)
        self.fn = torch.tensor([0], dtype=torch.float, device=self.device)
        self.eps = torch.tensor([1e-10], dtype=torch.float, device=self.device)

    def add(self, tp, fp, fn):
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def value(self):
        pres = (self.tp + self.eps) / (self.tp + self.fp + self.eps)
        recall = (self.tp + self.eps) / (self.tp + self.fn + self.eps)
        f1 = (2 * pres * recall) / (pres + recall)
        return pres.item(), recall.item(), f1.item()

    def reset(self):
        self.tp = torch.tensor([0], dtype=torch.float, device=self.device)
        self.fn = torch.tensor([0], dtype=torch.float, device=self.device)
        self.fp = torch.tensor([0], dtype=torch.float, device=self.device)


class LossRe(nn.Module):
    def __init__(self, device):
        super(LossRe, self).__init__()
        self.device = device
        self.loss = torch.tensor([0], dtype=torch.float, device=self.device)
        self.cnt = torch.tensor([0], dtype=torch.float, device=self.device)

    def add(self, loss, cnt):
        self.loss += loss
        self.cnt += cnt

    def value(self):

        return (self.loss / self.cnt).item()

    def reset(self):
        self.loss = torch.tensor([0], dtype=torch.float, device=self.device)
        self.cnt = torch.tensor([0], dtype=torch.float, device=self.device)


def metric(pred, target):
    bin_pred = torch.zeros_like(pred)
    bin_pred[pred > 0.5] = 1
    bin_pred.view(-1)
    target.view(-1)
    tp = torch.sum(bin_pred * target)
    fp = torch.sum(bin_pred * (1 - target))
    fn = torch.sum((1 - bin_pred) * target)
    return tp, fp, fn


