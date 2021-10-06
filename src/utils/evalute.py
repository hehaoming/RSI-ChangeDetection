import torch

class val:
    def __init__(self, device):
        self.device = device
        self.tp = torch.tensor(0, dtype=torch.float, device=self.device)
        self.fp = torch.tensor(0, dtype=torch.float, device=self.device)
        self.fn = torch.tensor(0, dtype=torch.float, device=self.device)
        self.eps = torch.tensor(1e-10, dtype=torch.float, device=self.device)

    def add(self, tp, fp, fn):
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def value(self):
        pres = (self.tp + self.eps) / (self.tp + self.fp + self.eps)
        recall = (self.tp + self.eps) / (self.tp + self.fn + self.eps)
        f1 = (2 * pres * recall) / (pres + recall)
        return pres, recall, f1

    def reset(self):
        self.tp = torch.tensor(0, dtype=torch.float, device=self.device)
        self.fn = torch.tensor(0, dtype=torch.float, device=self.device)
        self.fp = torch.tensor(0, dtype=torch.float, device=self.device)



