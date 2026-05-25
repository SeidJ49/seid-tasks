# This file is adapted from RadarDistill/OpenPCDet's fastai-style optimization helpers.

from functools import partial

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import numpy as np
import torch.nn as nn
import torch.optim as optim


bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def split_bn_bias(layer_groups):
    split_groups = []
    for layer_group in layer_groups:
        non_bn_layers, bn_layers = [], []
        for child in layer_group.children():
            if isinstance(child, bn_types):
                bn_layers.append(child)
            else:
                non_bn_layers.append(child)
        split_groups += [nn.Sequential(*non_bn_layers), nn.Sequential(*bn_layers)]
    return split_groups


def listify(value=None, q=None):
    if value is None:
        value = []
    elif isinstance(value, str):
        value = [value]
    elif not isinstance(value, Iterable):
        value = [value]
    n_items = q if type(q) == int else len(value) if q is None else len(q)
    if len(value) == 1:
        value = value * n_items
    assert len(value) == n_items, f'List len mismatch ({len(value)} vs {n_items})'
    return list(value)


def trainable_params(module):
    return filter(lambda param: param.requires_grad, module.parameters())


def is_tuple(value):
    return isinstance(value, tuple)


class OptimWrapper:
    def __init__(self, opt, wd, true_wd=False, bn_wd=True):
        self.opt = opt
        self.true_wd = true_wd
        self.bn_wd = bn_wd
        self.opt_keys = list(self.opt.param_groups[0].keys())
        self.opt_keys.remove('params')
        self.read_defaults()
        self.wd = wd

    @classmethod
    def create(cls, opt_func, lr, layer_groups, **kwargs):
        split_groups = split_bn_bias(layer_groups)
        opt = opt_func([{'params': trainable_params(group), 'lr': 0} for group in split_groups])
        opt = cls(opt, **kwargs)
        opt.lr = listify(lr, layer_groups)
        opt.opt_func = opt_func
        return opt

    def step(self):
        if self.true_wd:
            for lr, wd, group_non_bn, group_bn in zip(self._lr, self._wd, self.opt.param_groups[::2], self.opt.param_groups[1::2]):
                for param in group_non_bn['params']:
                    if not param.requires_grad:
                        continue
                    param.data.mul_(1 - wd * lr)
                if self.bn_wd:
                    for param in group_bn['params']:
                        if not param.requires_grad:
                            continue
                        param.data.mul_(1 - wd * lr)
            self.set_val('weight_decay', listify(0, self._wd))
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad()

    def __getattr__(self, key):
        return getattr(self.opt, key, None)

    @property
    def lr(self):
        return self._lr[-1]

    @lr.setter
    def lr(self, value):
        self._lr = self.set_val('lr', listify(value, self._lr))

    @property
    def mom(self):
        return self._mom[-1]

    @mom.setter
    def mom(self, value):
        if 'momentum' in self.opt_keys:
            self.set_val('momentum', listify(value, self._mom))
        elif 'betas' in self.opt_keys:
            self.set_val('betas', (listify(value, self._mom), self._beta))
        self._mom = listify(value, self._mom)

    @property
    def beta(self):
        return None if self._beta is None else self._beta[-1]

    @beta.setter
    def beta(self, value):
        if value is None:
            return
        if 'betas' in self.opt_keys:
            self.set_val('betas', (self._mom, listify(value, self._beta)))
        elif 'alpha' in self.opt_keys:
            self.set_val('alpha', listify(value, self._beta))
        self._beta = listify(value, self._beta)

    @property
    def wd(self):
        return self._wd[-1]

    @wd.setter
    def wd(self, value):
        if not self.true_wd:
            self.set_val('weight_decay', listify(value, self._wd), bn_groups=self.bn_wd)
        self._wd = listify(value, self._wd)

    def read_defaults(self):
        self._beta = None
        if 'lr' in self.opt_keys:
            self._lr = self.read_val('lr')
        if 'momentum' in self.opt_keys:
            self._mom = self.read_val('momentum')
        if 'alpha' in self.opt_keys:
            self._beta = self.read_val('alpha')
        if 'betas' in self.opt_keys:
            self._mom, self._beta = self.read_val('betas')
        if 'weight_decay' in self.opt_keys:
            self._wd = self.read_val('weight_decay')

    def set_val(self, key, value, bn_groups=True):
        if is_tuple(value):
            value = [(v1, v2) for v1, v2 in zip(*value)]
        for v, group_non_bn, group_bn in zip(value, self.opt.param_groups[::2], self.opt.param_groups[1::2]):
            group_non_bn[key] = v
            if bn_groups:
                group_bn[key] = v
        return value

    def read_val(self, key):
        value = [param_group[key] for param_group in self.opt.param_groups[::2]]
        if is_tuple(value[0]):
            value = [v[0] for v in value], [v[1] for v in value]
        return value


def annealing_cos(start, end, pct):
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start - end) / 2 * cos_out


class OneCycle:
    def __init__(self, optimizer, total_step, lr_max, moms, div_factor, pct_start, last_step=0):
        self.optimizer = optimizer
        self.total_step = total_step
        self.step_num = max(int(last_step), 0)
        self.lr_phases = []
        self.mom_phases = []
        low_lr = lr_max / div_factor
        lr_phases = (
            (0, partial(annealing_cos, low_lr, lr_max)),
            (pct_start, partial(annealing_cos, lr_max, low_lr / 1e4)),
        )
        mom_phases = (
            (0, partial(annealing_cos, *moms)),
            (pct_start, partial(annealing_cos, *moms[::-1])),
        )
        optimizer.lr = low_lr
        optimizer.mom = moms[0]

        for index, (start, func) in enumerate(lr_phases):
            end = lr_phases[index + 1][0] if index < len(lr_phases) - 1 else 1.0
            self.lr_phases.append((int(start * total_step), int(end * total_step), func))
        for index, (start, func) in enumerate(mom_phases):
            end = mom_phases[index + 1][0] if index < len(mom_phases) - 1 else 1.0
            self.mom_phases.append((int(start * total_step), int(end * total_step), func))

        self.step_per_batch = True

    def step(self):
        step = self.step_num
        for start, end, func in self.lr_phases:
            if step >= start:
                self.optimizer.lr = func((step - start) / max(end - start, 1))
        for start, end, func in self.mom_phases:
            if step >= start:
                self.optimizer.mom = func((step - start) / max(end - start, 1))
        self.step_num += 1