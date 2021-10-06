
class val:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.eps = 1e-10

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
        self.tp = 0
        self.fn = 0
        self.fp = 0



