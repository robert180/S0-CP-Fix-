# -*- coding: utf-8 -*-
"""用真正的默认后端（inductor）复核 torch.compile 下的行为。"""
import sys, warnings, torch, torch.nn as nn, torch.utils.checkpoint as cp
warnings.filterwarnings('ignore')
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from bn_safe_checkpoint import checkpoint as bn_safe_checkpoint

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.mode, self.reentrant = 'none', True
    def forward(self, x):
        def _inner(x):
            return torch.relu(self.bn(self.conv(x)))
        if self.mode == 'none':
            return _inner(x)
        if self.mode == 'stock':
            return cp.checkpoint(_inner, x, use_reentrant=self.reentrant)
        return bn_safe_checkpoint(_inner, x, module=self, use_reentrant=self.reentrant)

def run(mode, reentrant, compiled, iters=3):
    import torch._dynamo as dynamo
    dynamo.reset()
    torch.manual_seed(0)
    m = Block(); m.mode, m.reentrant = mode, reentrant; m.train()
    net = torch.compile(m) if compiled else m      # 默认后端 = inductor
    for i in range(iters):
        torch.manual_seed(100 + i)
        net(torch.randn(2, 4, 8, 8, requires_grad=True)).sum().backward()
    return int(m.bn.num_batches_tracked), m.bn.running_mean.clone()

print('torch', torch.__version__, '| 后端: 默认 (inductor)')
for compiled in (False, True):
    lab = 'torch.compile(inductor)' if compiled else 'eager'
    try:
        ref_n, ref_mean = run('none', True, compiled)
    except Exception as e:
        print('%s 参照就失败: %s' % (lab, type(e).__name__)); continue
    print('\n=== %s ===  参照(无检查点) n=%d' % (lab, ref_n))
    for mode in ('stock', 'fixed'):
        for reentrant in (True, False):
            nm = '原始检查点' if mode == 'stock' else '本文修复  '
            try:
                n, mean = run(mode, reentrant, compiled)
                d = (mean - ref_mean).abs().max().item()
                print('  %s reentrant=%-5s  n=%d (%d/迭代)  与参照差 %.3e  %s'
                      % (nm, reentrant, n, n // 3, d,
                         'BUG' if n == 2 * ref_n else 'OK'))
            except Exception as e:
                print('  %s reentrant=%-5s  抛出 %s: %s'
                      % (nm, reentrant, type(e).__name__,
                         str(e).split('\n')[0][:88]))
