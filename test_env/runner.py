import torch


class Runner:
    def __init__(self, learner, data):
        self.iters = None
        self.stop = None
        self.loss = None
        self.pred = None
        self.yb = None
        self.xb = None
        self.learner = learner
        self.data = data

    @property
    def optimizer(self):
        return self.learner.opt

    @property
    def model(self):
        return self.learner.model

    @property
    def loss_fn(self):
        return self.learner.loss_fn

    @property
    def data(self):
        return self.learner.data

    def batch(self, xb, yb):
        self.xb, self.yb = xb, yb
        self.pred = self.model(xb)
        self.loss = self.loss_fn(self.pred, yb)
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def all_batches(self, dl):
        self.iters = len(dl)
        for xb, yb in dl:
            if self.stop:
                break
            self.batch(xb, yb)

    def fit(self, epochs, learn):
        self.epochs = epochs
        self.learn = learn



