# -*- coding: utf-8 -*-
"""Framework-agnostic self-check: is any BatchNorm updated twice per iteration?

The criterion needs no knowledge of where gradient checkpointing is used.
Record ``num_batches_tracked`` for every BatchNorm before and after ONE
training iteration (forward + backward). A layer whose counter advances by
more than 1 sits inside a checkpointed region and is accumulating the same
batch twice.

Usage::

    from detect_contamination import snapshot, report

    before = snapshot(model)
    loss = model(batch).sum()       # your normal training step
    loss.backward()
    print(report(model, before))

Exit status of ``python detect_contamination.py`` is 1 if the built-in
self-test finds contamination, which makes it usable in CI.
"""
import sys

from torch.nn.modules.batchnorm import _BatchNorm


def snapshot(model):
    """Return {layer_name: num_batches_tracked} for every BatchNorm."""
    return {name: int(sub.num_batches_tracked)
            for name, sub in model.named_modules()
            if isinstance(sub, _BatchNorm) and sub.num_batches_tracked is not None}


def diff(model, before):
    """Return {layer_name: delta} for layers whose counter advanced != 1."""
    after = snapshot(model)
    return {name: after[name] - before[name]
            for name in after
            if name in before and after[name] - before[name] != 1}


def report(model, before):
    bad = diff(model, before)
    total = len(snapshot(model))
    if not bad:
        return 'OK: all %d BatchNorm layers advanced exactly 1 per iteration.' % total
    lines = ['CONTAMINATED: %d of %d BatchNorm layers advanced != 1 per iteration.'
             % (len(bad), total)]
    for name, delta in sorted(bad.items()):
        lines.append('  %-60s +%d' % (name, delta))
    lines.append('A delta of 2 means the layer sits inside a checkpointed region '
                 'and accumulates the same batch twice (effective momentum '
                 'm -> 2m - m**2).')
    return '\n'.join(lines)


def _self_test():
    """Build a tiny checkpointed block and confirm the criterion fires."""
    import torch
    import torch.nn as nn
    import torch.utils.checkpoint as cp

    class Block(nn.Module):
        def __init__(self, use_cp):
            super().__init__()
            self.conv = nn.Conv2d(4, 4, 3, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(4)
            self.use_cp = use_cp

        def forward(self, x):
            def _inner(x):
                return torch.relu(self.bn(self.conv(x)))
            if not self.use_cp:
                return _inner(x)
            try:
                return cp.checkpoint(_inner, x, use_reentrant=True)
            except TypeError:          # torch < 1.11 has no use_reentrant
                return cp.checkpoint(_inner, x)

    contaminated = False
    for use_cp in (False, True):
        m = Block(use_cp)
        m.train()
        before = snapshot(m)
        x = torch.randn(2, 4, 8, 8, requires_grad=True)
        m(x).sum().backward()
        text = report(m, before)
        print('checkpointing=%-5s -> %s' % (use_cp, text.split('\n')[0]))
        if use_cp and text.startswith('CONTAMINATED'):
            contaminated = True
    return contaminated


if __name__ == '__main__':
    sys.exit(1 if _self_test() else 0)
