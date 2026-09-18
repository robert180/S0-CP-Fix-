# -*- coding: utf-8 -*-
"""S11 GPU part: SyncBatchNorm and CUDA Graph, on the paper's own env (torch 1.10.2)."""
import os, sys, warnings, torch, torch.nn as nn
import torch.distributed as dist, torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm
warnings.filterwarnings('ignore')

TRACKED = ('running_mean', 'running_var', 'num_batches_tracked')

def protect(fn, mod):
    st = {'done': False}
    def w(*a, **k):
        if not st['done']:
            st['done'] = True
            return fn(*a, **k)
        saved = []
        try:
            for s in mod.modules():
                if isinstance(s, _BatchNorm):
                    for n in TRACKED:
                        v = getattr(s, n, None)
                        if v is not None:
                            saved.append((s, n, v)); setattr(s, n, v.detach().clone())
            return fn(*a, **k)
        finally:
            for s, n, v in reversed(saved):
                setattr(s, n, v)
    return w

class Block(nn.Module):
    def __init__(self, sync=False):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        bn = nn.BatchNorm2d(4)
        self.bn = nn.SyncBatchNorm.convert_sync_batchnorm(bn) if sync else bn
        self.mode = 'none'
    def forward(self, x):
        def _inner(x):
            return torch.relu(self.bn(self.conv(x)))
        if self.mode == 'none':
            return _inner(x)
        if self.mode == 'stock':
            return cp.checkpoint(_inner, x)
        return cp.checkpoint(protect(_inner, self), x)

def run(mode, sync, iters=3, graph=False):
    torch.manual_seed(0)
    m = Block(sync=sync).cuda(); m.mode = mode; m.train()
    for i in range(iters):
        torch.manual_seed(100 + i)
        x = torch.randn(2, 4, 8, 8, device='cuda', requires_grad=True)
        m(x).sum().backward()
    bn = [s for s in m.modules() if isinstance(s, _BatchNorm)][0]
    return int(bn.num_batches_tracked), bn.running_mean.detach().clone()

os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', '29613')
dist.init_process_group('nccl', rank=0, world_size=1)
print('torch', torch.__version__, '| gpu', torch.cuda.get_device_name(0))
print('judge: BN updates per training iteration; correct = 1\n')
for sync in (False, True):
    lab = 'SyncBatchNorm' if sync else 'BatchNorm2d'
    try:
        rn, rmean = run('none', sync)
    except Exception as e:
        print('%-14s reference FAILED: %s' % (lab, type(e).__name__)); continue
    print('=== %s ===  reference(no checkpoint) n=%d' % (lab, rn))
    for mode, nm in (('stock', 'stock cp   '), ('fixed', 'this fix   ')):
        try:
            n, mean = run(mode, sync)
            d = (mean - rmean).abs().max().item()
            print('  %s n=%d (%d/iter)  max|diff vs ref| %.3e  %s'
                  % (nm, n, n // 3, d, 'BUG' if n == 2 * rn else 'OK'))
        except Exception as e:
            print('  %s raised %s: %s' % (nm, type(e).__name__, str(e).split('\n')[0][:80]))
    print()

print('=== CUDA Graph ===')
try:
    m = Block().cuda(); m.mode = 'stock'; m.train()
    static = torch.randn(2, 4, 8, 8, device='cuda', requires_grad=True)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            m(static).sum().backward()
    torch.cuda.current_stream().wait_stream(s)
    before = int(m.bn.num_batches_tracked)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = m(static).sum()
    g.replay(); torch.cuda.synchronize()
    print('  captured OK; num_batches_tracked %d -> %d after one replay'
          % (before, int(m.bn.num_batches_tracked)))
    print('  NOTE: num_batches_tracked is a CPU-side .add_ on a captured graph;')
    print('        replays do not re-run it, so counts go stale under CUDA Graph.')
except Exception as e:
    print('  CUDA Graph capture raised %s: %s'
          % (type(e).__name__, str(e).split('\n')[0][:120]))
