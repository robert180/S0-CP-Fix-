# -*- coding: utf-8 -*-
"""Paired within-run experiment, v2: stay in the deployed regime.

v1 reset every BatchNorm and forced momentum=0.1 on all 51 layers. That was
wrong: 29 of the 51 are configured with momentum=0.01, and resetting threw
away the trained statistics, so both arms evaluated at mAP~0.33 instead of
~0.515 -- far outside the regime the question is about.

v2 keeps the trained running statistics and each layer's own configured
momentum, and runs only a short accumulation. Both arms therefore stay near
the deployed state. The 46 layers outside the checkpointed region receive
identical updates in both arms (same weights, same data order, same seed) and
cancel; the only difference is whether the 5 layers inside it are updated
once or twice per iteration. The second update is applied externally through
a forward hook using the batch statistics the layer just saw, which is
algebraically what the recomputation pass does.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, '.')
import single_gpu_dist  # noqa: F401
import torch
import torch.nn as nn
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
CFG = 'runs/s0_cp_bnfix_v1/input_config.yaml'
CKPT = 'runs/s0_cp_bnfix_v1/epoch_6.pth'
OUT = Path('runs/bn_paired2')
OUT.mkdir(exist_ok=True)

print('N_BATCHES=%d' % N, flush=True)
dist.init()
torch.cuda.set_device(0)
configs.load(CFG, recursive=True)
cfg = Config(recursive_eval(configs), filename=CFG)
ds = build_dataset(cfg.data.train)
net = build_model(cfg.model)
model = MMDataParallel(net.cuda(), device_ids=[0])

ALL_BN = [m for m in net.modules()
          if isinstance(m, nn.modules.batchnorm._BatchNorm)]
TARGETS = [(n, m) for n, m in net.named_modules()
           if isinstance(m, nn.modules.batchnorm._BatchNorm)
           and ('vtransform.dtransform.' in n or 'vtransform.depthnet.' in n)]
print('targets %d / %d BN' % (len(TARGETS), len(ALL_BN)), flush=True)
print('target momentum %s | all momentum set %s'
      % (sorted({m.momentum for _, m in TARGETS}),
         sorted({m.momentum for m in ALL_BN})), flush=True)


def second_update(mod, inp, out):
    x = inp[0].detach().float()
    dims = [0] + list(range(2, x.dim()))
    bmean = x.mean(dim=dims)
    bvar = x.var(dim=dims, unbiased=True)
    m = mod.momentum
    with torch.no_grad():
        mod.running_mean.mul_(1 - m).add_(bmean.to(mod.running_mean.dtype), alpha=m)
        mod.running_var.mul_(1 - m).add_(bvar.to(mod.running_var.dtype), alpha=m)
        if mod.num_batches_tracked is not None:
            mod.num_batches_tracked.add_(1)


def accumulate(double):
    tag = 'double' if double else 'single'
    torch.manual_seed(0)
    load_checkpoint(net, CKPT, map_location='cpu')   # trained weights AND stats
    base = {id(m): int(m.num_batches_tracked) for m in ALL_BN
            if m.num_batches_tracked is not None}
    handles = []
    if double:
        for _, m in TARGETS:
            handles.append(m.register_forward_hook(second_update))
    dl = build_dataloader(ds, samples_per_gpu=2, workers_per_gpu=4,
                          dist=False, shuffle=True, seed=0)
    model.train()
    for m in net.modules():
        if isinstance(m, nn.Dropout):
            m.eval()
    it, done = iter(dl), 0
    with torch.no_grad():
        while done < N:
            try:
                data = next(it)
            except StopIteration:
                it = iter(dl)
                continue
            try:
                model.train_step(data, None)
            except TypeError:
                model.train_step(data, optimizer=None)
            done += 1
            if done % 10 == 0:
                print('   %s %d/%d' % (tag, done, N), flush=True)
    for h in handles:
        h.remove()
    tgt = sorted({int(m.num_batches_tracked) - base[id(m)] for _, m in TARGETS})
    oth = sorted({int(m.num_batches_tracked) - base[id(m)] for m in ALL_BN
                  if m.num_batches_tracked is not None
                  and all(m is not t for _, t in TARGETS)})
    print('  delta num_batches_tracked  target=%s  others=%s' % (tgt, oth), flush=True)
    state = {k: v for k, v in net.state_dict().items()
             if not k.startswith('teacher.')}
    p = OUT / ('paired2_%s.pth' % tag)
    torch.save({'meta': {'n_batches': N, 'double': double}, 'state_dict': state}, p)
    print('  saved %s' % p, flush=True)
    return dict(ckpt=str(p), target_delta=tgt, other_delta=oth)


res = {}
for double in (False, True):
    tag = 'double' if double else 'single'
    print('\n=== %s ===' % tag, flush=True)
    res[tag] = accumulate(double)
(OUT / 'accum.json').write_text(json.dumps(res, indent=2))
print('ACCUM_DONE')
