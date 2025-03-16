import numpy as np


class F1Score:
    def __init__(self):
        self.init()

    def init(self):
        self.tp = self.fp = self.fn = 0

    def add(self, reference, prediction):
        self.tp += sum([1 for x in prediction if x in reference])
        self.fp += sum([1 for x in prediction if x not in reference])
        self.fn += sum([1 for x in reference if x not in prediction])

    def add_batch(self, references, predictions):
        for ref, pred in zip(references, predictions):
            self.add(ref, pred)

    def compute(self):
        r = F1Score.f1_score(self.tp, self.fp, self.fn)
        self.init()
        return r

    @staticmethod
    def f1_score(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return {"f1": f1, "p": p, "r": r}


class MRR:
    def __init__(self):
        self.init()

    def init(self):
        self.scores = []
        self.n_samples = 0

    def add(self, reference, prediction):
        rank = -1
        for i, p in enumerate(prediction):
            if p in reference:
                rank = i + 1
                break

        if rank != -1:
            self.scores.append(1 / rank)
        self.n_samples += 1

    def add_batch(self, references, predictions):
        for ref, pred in zip(references, predictions):
            self.add(ref, pred)

    def compute(self):
        r = np.sum(self.scores) / self.n_samples
        self.init()
        return {"mrr": r}
