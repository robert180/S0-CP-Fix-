# -*- coding: utf-8 -*-
"""验证 bn_safe_checkpoint：BN 只更新一次，且输出/梯度/统计与不使用检查点时完全一致。"""
import copy
import os
import sys
import time

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision.models.resnet import BasicBlock, Bottleneck

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from bn_safe_checkpoint import checkpoint as safe_checkpoint  # noqa: E402


def make(kind):
    torch.manual_seed(0)
    return (BasicBlock(16, 16) if kind == 'basic' else Bottleneck(16, 4)).train()


def bn_state(m):
    return {n: (s.running_mean.clone(), s.running_var.clone(), int(s.num_batches_tracked))
            for n, s in m.named_modules() if isinstance(s, _BatchNorm)}


def one_iteration(mod, x, mode):
    before = {n: int(s.num_batches_tracked)
              for n, s in mod.named_modules() if isinstance(s, _BatchNorm)}
    if mode == 'none':
        out = mod(x)
    elif mode == 'stock':
        out = cp.checkpoint(mod, x, use_reentrant=True)
    elif mode == 'stock_nr':
        out = cp.checkpoint(mod, x, use_reentrant=False)
    elif mode == 'fixed':
        out = safe_checkpoint(mod, x, module=mod, use_reentrant=True)
    elif mode == 'fixed_nr':
        out = safe_checkpoint(mod, x, module=mod, use_reentrant=False)
    loss = out.square().mean()
    loss.backward()
    delta = {n: int(s.num_batches_tracked) - before[n]
             for n, s in mod.named_modules() if isinstance(s, _BatchNorm)}
    grads = {n: p.grad.clone() for n, p in mod.named_parameters() if p.grad is not None}
    return out.detach(), grads, delta, bn_state(mod)


def run(kind):
    print('\n' + '=' * 76)
    print('模型: torchvision %s' % ('ResNet BasicBlock' if kind == 'basic' else 'ResNet Bottleneck'))
    print('=' * 76)
    torch.manual_seed(1)
    x0 = torch.randn(4, 16, 8, 8)

    ref_mod = make(kind)
    ref_out, ref_grads, ref_delta, ref_bn = one_iteration(ref_mod, x0.clone().requires_grad_(), 'none')
    print('%-26s BN更新=%s' % ('不使用检查点（基准）', sorted(set(ref_delta.values()))))

    rows = []
    for mode, label in [('stock', '原 checkpoint (reentrant)'),
                        ('stock_nr', '原 checkpoint (non-reentrant)'),
                        ('fixed', '修复后 (reentrant)'),
                        ('fixed_nr', '修复后 (non-reentrant)')]:
        mod = make(kind)
        out, grads, delta, bn = one_iteration(mod, x0.clone().requires_grad_(), mode)
        d_out = (out - ref_out).abs().max().item()
        d_grad = max((grads[k] - ref_grads[k]).abs().max().item() for k in ref_grads)
        d_mean = max((bn[k][0] - ref_bn[k][0]).abs().max().item() for k in ref_bn)
        d_var = max((bn[k][1] - ref_bn[k][1]).abs().max().item() for k in ref_bn)
        rows.append((label, sorted(set(delta.values())), d_out, d_grad, d_mean, d_var))

    print('\n%-30s %-8s %-11s %-11s %-11s %-11s' %
          ('设置', 'BN更新', '输出差', '梯度差', 'run_mean差', 'run_var差'))
    for label, delta, do, dg, dm, dv in rows:
        print('%-30s %-8s %-11.3e %-11.3e %-11.3e %-11.3e' % (label, delta, do, dg, dm, dv))

    ok = all(r[1] == [1] and r[4] < 1e-6 and r[5] < 1e-6 for r in rows if '修复后' in r[0])
    bad = all(r[1] == [2] for r in rows if '原 ' in r[0])
    print('\n判定: 修复后BN更新1次且统计与基准一致 =', ok, '| 未修复时更新2次 =', bad)
    return ok and bad


def check_no_module_arg():
    """未传 module 时应退化为原行为（drop-in 安全性）。"""
    mod = make('basic')
    before = {n: int(s.num_batches_tracked)
              for n, s in mod.named_modules() if isinstance(s, _BatchNorm)}
    torch.manual_seed(1)
    out = safe_checkpoint(mod, torch.randn(4, 16, 8, 8, requires_grad=True), use_reentrant=True)
    out.sum().backward()
    delta = {n: int(s.num_batches_tracked) - before[n]
             for n, s in mod.named_modules() if isinstance(s, _BatchNorm)}
    print('\n未传 module 时退化为原行为（应为 [2]）:', sorted(set(delta.values())))
    return sorted(set(delta.values())) == [2]


def check_eval_mode():
    """eval 模式下不应做任何额外操作。"""
    mod = make('basic').eval()
    x = torch.randn(4, 16, 8, 8, requires_grad=True)
    a = mod(x)
    b = safe_checkpoint(mod, x, module=mod, use_reentrant=True)
    same = torch.allclose(a, b, atol=1e-6)
    print('eval 模式下输出与直接前向一致:', same)
    return same


def bench(kind='basic', n=40, rounds=5):
    """交替测量多轮取中位数，避免单轮的缓存/调度噪声。"""
    import statistics
    torch.manual_seed(1)
    x = torch.randn(8, 16, 32, 32)
    fns = {'stock': lambda m, t: cp.checkpoint(m, t, use_reentrant=True),
           'fixed': lambda m, t: safe_checkpoint(m, t, module=m, use_reentrant=True)}
    samples = {'stock': [], 'fixed': []}
    mods = {k: make(kind) for k in fns}
    for k, fn in fns.items():                      # 预热
        for _ in range(10):
            fn(mods[k], x.clone().requires_grad_()).sum().backward()
    for _ in range(rounds):
        for k in ('stock', 'fixed'):               # 交替，抵消漂移
            t0 = time.perf_counter()
            for _ in range(n):
                fns[k](mods[k], x.clone().requires_grad_()).sum().backward()
            samples[k].append((time.perf_counter() - t0) / n * 1000)
    a, b = statistics.median(samples['stock']), statistics.median(samples['fixed'])
    print('\n开销（%d 轮 x %d 次取中位数）: 原 %.3f ms/iter | 修复后 %.3f ms/iter | 相对 %+.1f%%'
          % (rounds, n, a, b, (b / a - 1) * 100))


if __name__ == '__main__':
    ok1 = run('basic')
    ok2 = run('bottleneck')
    ok3 = check_no_module_arg()
    ok4 = check_eval_mode()
    bench()
    print('\n' + '=' * 76)
    print('ALL_TESTS_PASS' if all((ok1, ok2, ok3, ok4)) else 'SOME_TESTS_FAILED')
    print('torch', torch.__version__)
