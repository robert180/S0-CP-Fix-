# -*- coding: utf-8 -*-
"""对比 FairScale patch_batchnorm 与本文修复在两种检查点实现下的行为。

FairScale (fairscale/nn/checkpoint/checkpoint_utils.py, 2021) 已经用
forward hook + track_running_stats 抑制了重算轮的 BN 更新，判别条件是
torch.is_grad_enabled()。本文的修复改用调用序计数。

两者在 reentrant 检查点下等价；关键分歧在 use_reentrant=False。
本脚本给出可复现的判据：每次训练迭代后 num_batches_tracked 的增量。
"""
import os
import sys

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from bn_safe_checkpoint import checkpoint as bn_safe_checkpoint  # noqa: E402

try:
    from fairscale.nn.checkpoint.checkpoint_utils import patch_batchnorm
except ImportError:
    sys.exit('This comparison needs FairScale:  pip install fairscale==0.4.13')


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(4)          # momentum 默认 0.1

    def forward(self, x):
        def _inner(x):
            return torch.relu(self.bn(self.conv(x)))
        return self.ckpt(_inner, x)


def make(mode, reentrant):
    m = Block()
    if mode == 'stock':
        m.ckpt = lambda f, x: cp.checkpoint(f, x, use_reentrant=reentrant)
    elif mode == 'fairscale':
        patch_batchnorm(m)
        m.ckpt = lambda f, x: cp.checkpoint(f, x, use_reentrant=reentrant)
    elif mode == 'ours':
        m.ckpt = lambda f, x: bn_safe_checkpoint(f, x, module=m,
                                                 use_reentrant=reentrant)
    elif mode == 'none':                      # 参照：完全不用检查点
        m.ckpt = lambda f, x: f(x)
    return m


def run(mode, reentrant, iters=5):
    torch.manual_seed(0)
    m = make(mode, reentrant)
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=0.0)   # lr=0 隔离权重变化
    for _ in range(iters):
        x = torch.randn(2, 4, 8, 8, requires_grad=True)
        m(x).sum().backward()
        opt.step()
        opt.zero_grad()
    bn = [s for s in m.modules() if isinstance(s, _BatchNorm)][0]
    return int(bn.num_batches_tracked), float(bn.running_mean.abs().sum())


ITERS = 5
print('每次迭代 BN 应当只累计 1 次；%d 次迭代后 num_batches_tracked 应为 %d\n'
      % (ITERS, ITERS))
print('%-12s %-24s %-24s' % ('实现', 'use_reentrant=True', 'use_reentrant=False'))
print('-' * 62)
ref = None
for mode, label in [('none', '无检查点(参照)'), ('stock', '原始 cp.checkpoint'),
                    ('fairscale', 'FairScale patch'), ('ours', '本文修复')]:
    cells = []
    for reentrant in (True, False):
        if mode == 'none' and not reentrant:
            cells.append('—')
            continue
        n, s = run(mode, reentrant, ITERS)
        if mode == 'none':
            ref = (n, s)
        ok = '✓' if n == ITERS else '✗'
        cells.append('n=%d %s  |mean|=%.5f' % (n, ok, s))
    print('%-12s %-24s %-24s' % (label, cells[0], cells[1]))

print('\n参照（无检查点）: n=%d  |running_mean|=%.5f' % ref)
print('\n判读：')
print('  FairScale 用 torch.is_grad_enabled() 判别，在其自带的 CheckpointFunction')
print('  （内部显式 torch.no_grad()）下成立；直接配合 PyTorch 的非重入检查点时，')
print('  两轮都是 grad-enabled，两个 hook 都提前 return，抑制失效。')
