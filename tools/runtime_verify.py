# -*- coding: utf-8 -*-
"""运行时验证：梯度检查点是否使 BatchNorm 滑动统计被同一批次更新两次。

使用 PyTorch 官方 torch.utils.checkpoint 与 torchvision 官方模型，
不依赖 mmcv / BEVFusion，以验证该现象与具体框架无关。
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.densenet import _DenseLayer

torch.manual_seed(0)


def bns(m):
    return [(n, mod) for n, mod in m.named_modules()
            if isinstance(mod, nn.modules.batchnorm._BatchNorm)]


def probe(build, forward, label):
    """返回 (BN 更新次数字典, running_mean 快照)"""
    torch.manual_seed(0)
    mod = build()
    mod.train()
    torch.manual_seed(1)
    x = torch.randn(4, 16, 8, 8, requires_grad=True)
    before = {n: int(b.num_batches_tracked) for n, b in bns(mod)}
    out = forward(mod, x)
    (out.sum() if torch.is_tensor(out) else out[0].sum()).backward()
    delta = {n: int(b.num_batches_tracked) - before[n] for n, b in bns(mod)}
    rmean = {n: b.running_mean.detach().clone() for n, b in bns(mod)}
    return delta, rmean, label


def report(title, cases):
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)
    base_delta, base_mean, _ = cases[0]
    for delta, rmean, label in cases:
        d = sorted(set(delta.values()))
        diff = max((rmean[n] - base_mean[n]).abs().max().item() for n in rmean)
        print('  %-34s BN更新次数=%-8s  running_mean与基线最大差=%.3e'
              % (label, d, diff))


# ─── 1. torchvision ResNet BasicBlock（mmdet/mmseg ResNet 的等价结构）───
mk = lambda: BasicBlock(16, 16)
report('1. torchvision ResNet BasicBlock（含 2 个 BatchNorm2d）', [
    probe(mk, lambda m, x: m(x), '不使用检查点（基线）'),
    probe(mk, lambda m, x: checkpoint(m, x, use_reentrant=True), 'checkpoint use_reentrant=True'),
    probe(mk, lambda m, x: checkpoint(m, x, use_reentrant=False), 'checkpoint use_reentrant=False'),
])

# ─── 2. torchvision Bottleneck（ResNet-50 的基本块，3 个 BN）───
mk2 = lambda: Bottleneck(16, 4)
report('2. torchvision ResNet Bottleneck（含 3 个 BatchNorm2d）', [
    probe(mk2, lambda m, x: m(x), '不使用检查点（基线）'),
    probe(mk2, lambda m, x: checkpoint(m, x, use_reentrant=True), 'checkpoint use_reentrant=True'),
    probe(mk2, lambda m, x: checkpoint(m, x, use_reentrant=False), 'checkpoint use_reentrant=False'),
])

# ─── 3. torchvision DenseNet _DenseLayer 的官方 memory_efficient 开关 ───
print('\n' + '=' * 72)
print('3. torchvision DenseNet _DenseLayer：官方 memory_efficient 开关')
print('=' * 72)
res = {}
for me in (False, True):
    torch.manual_seed(0)
    layer = _DenseLayer(16, 8, 4, 0.0, memory_efficient=me)
    layer.train()
    torch.manual_seed(1)
    x = torch.randn(4, 16, 8, 8, requires_grad=True)
    before = {n: int(b.num_batches_tracked) for n, b in bns(layer)}
    out = layer(x)
    out.sum().backward()
    delta = {n: int(b.num_batches_tracked) - before[n] for n, b in bns(layer)}
    res[me] = (delta, {n: b.running_mean.detach().clone() for n, b in bns(layer)})
    print('  memory_efficient=%-6s BN更新次数=%s' % (me, sorted(set(delta.values()))))
d0, m0 = res[False]
d1, m1 = res[True]
mx = max((m1[n] - m0[n]).abs().max().item() for n in m0)
print('  两种设置下 running_mean 最大差 = %.3e' % mx)

# ─── 4. 等效动量验证：更新两次 == 一次 m_eff = 2m - m^2 ───
print('\n' + '=' * 72)
print('4. 等效动量：同一批次更新两次 vs 一次 m_eff = 2m - m²')
print('=' * 72)
torch.manual_seed(2)
x = torch.randn(12, 8, 32, 88)
init_m, init_v = torch.randn(8), torch.rand(8) + 0.5


def ema(m, times):
    bn = nn.BatchNorm2d(8, momentum=m)
    bn.train()
    with torch.no_grad():
        bn.running_mean.copy_(init_m)
        bn.running_var.copy_(init_v)
    for _ in range(times):
        bn(x)
    return bn.running_mean.clone(), bn.running_var.clone()


for m in (0.1, 0.05, 0.01, 0.3):
    meff = 2 * m - m * m
    a = ema(m, 1)
    b = ema(m, 2)
    c = ema(meff, 1)
    print('  m=%-5.2f  m_eff=%-6.4f | 缺陷vs等效: mean %.3e var %.3e | 缺陷vs正常: mean %.3e var %.3e'
          % (m, meff,
             (b[0] - c[0]).abs().max(), (b[1] - c[1]).abs().max(),
             (b[0] - a[0]).abs().max(), (b[1] - a[1]).abs().max()))

print('\ntorch %s | torchvision %s' % (torch.__version__, __import__('torchvision').__version__))
print('RUNTIME_VERIFY_DONE')
