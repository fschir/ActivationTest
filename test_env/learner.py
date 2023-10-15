from functools import partial
from operator import attrgetter

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import preprocessing

from torch import optim

from .callbacks import TestCallback, run_cbs
from .dataloader import DataLoader


class Learner:
    def __init__(self, model, dls: DataLoader = (0,), loss_func=F.mse_loss, lr=0.1, cbs=None, opt_func=optim.SGD):
        """
        :param model: model
        :param dls: dataloader
        :param loss_func: torch loss fn
        :param lr: learning rate
        :param cbs: [optional] callbacks
        :param opt_func: [default SGD] optimizer
        """
        self.opt = opt_func
        self.epochs = None
        self.dl = None
        self.n_epochs = None
        self.cbs = cbs
        self.dls = dls
        self.model = model
        self.loss_fn = loss_func
        self.lr = lr

    def _one_batch(self):
        self.predict()
        self.callback('after_predict')
        self.get_loss()
        self.callback('after_loss')
        if self.model.training:
            self.backward()
            self.callback('after_backward')
            self.step()
            self.callback('after_step')
            self.zero_grad()

    def _one_epoch(self):
        for self.iter, self.batch in enumerate(self.dl):
            self._one_batch()

    def one_epoch(self, training):
        self.model.train(training)
        self.dl = self.dls.train if training else self.dls.valid
        self._one_epoch()

    def _fit(self, train, valid):
        for self.epoch in self.epochs:
            if train:
                self.one_epoch(True)
            if valid:
                torch.no_grad()(self.one_epoch)(False)

    def fit(self, n_epochs=1, train=True, valid=True, cbs=None, lr=None):
        # `add_cb` and `rm_cb` were added in lesson 18
        for cb in cbs:
            self.cbs.append(cb)
        try:
            self.n_epochs = n_epochs
            self.epochs = range(n_epochs)
            if lr is None: lr = self.lr
            if self.opt_func:
                self.opt = self.opt_func(self.model.parameters(), lr)
            self._fit(train, valid)
        finally:
            for cb in cbs:
                self.cbs.remove(cb)

    def predict(self):
        self.preds = self.model()
    def get_loss(self):
        pass

    def __getattr__(self, name):
        if name in ('predict', 'backward', 'step', 'zero_grad'):
            return partial(self.callback, name)
        raise AttributeError(name)

    def callback(self, method_nm):
        run_cbs(self.cbs, method_nm, self)

    @property
    def training(self):
        return self.model.training


class LearnerSimple:
    def __init__(self, model, dls, loss_func, lr, opt_func=optim.SGD):

        self.dls = dls
        self.batch = None
        self.loss_func = loss_func
        self.model = model
        self.loss = None
        self.preds = None
        self.xb, self.yb = None, None
        self.opt_func = opt_func
        self.lr = lr

        self.activations = pd.DataFrame(columns=['Layer', 'Epoch', 'Values'])

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu:0')

    def one_batch(self):
        self.xb = self.batch[0].to(self.device)
        self.yb = self.batch[1].to(self.device)
        self.preds = self.model(self.xb)
        self.loss = self.loss_func(self.preds, self.yb)
        if self.model.training:
            self.loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        with torch.no_grad():
            self.calc_stats()

    def calc_stats(self):
        acc = (self.preds.argmax(dim=1) == self.yb).float().sum()
        self.accs.append(acc)
        n = len(self.xb)
        self.losses.append(self.loss * n)
        self.ns.append(n)

    def one_epoch(self, train):
        self.model.training = train
        dl = self.dls.train if train else self.dls.val
        for self.num, self.batch in enumerate(dl):
            self.one_batch()
        n = sum(self.ns)
        print(self.epoch, self.model.training, sum(self.losses).item() / n, sum(self.accs).item() / n)

    def log_activations(self, epoch):
        keys = self.model.activation.keys()
        for k in keys:
            values = {
                'Layer': k,
                'Epoch': epoch,
                'Values': self.model.activation[k].reshape(-1)
            }
            self.activations = pd.concat([self.activations, pd.DataFrame.from_dict(values)], ignore_index=True)

    def get_normalized_activations(self):
        self.activations.loc[:, 'normalized'] = self.activations.iloc[:, 2:].apply(lambda x: (x - x.mean()) / x.std())
        return self.activations

    def fit(self, n_epochs):
        curr_epoch = 1
        self.accs, self.losses, self.ns = [], [], []
        self.model.to(self.device)
        self.opt = self.opt_func(self.model.parameters(), self.lr)
        self.n_epochs = n_epochs
        for self.epoch in range(n_epochs):
            self.one_epoch(True)
            curr_epoch += 1
            self.log_activations(curr_epoch)
            with torch.no_grad():
                self.one_epoch(False)

