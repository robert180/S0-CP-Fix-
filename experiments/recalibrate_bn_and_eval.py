# -*- coding: utf-8 -*-
"""闭合因果链的关键实验（审稿意见 S4 / 质疑一）。

论证前提：该缺陷只影响推理期滑动统计，不改变前向输出与梯度（论文 3.1、3.2 节）。
若此说成立，S0-CP 与 S0-CP-Fix 学到的权重应当近乎相同，评估期的全部差异都应来自
BatchNorm 滑动统计。于是有一个直接可证伪的检验：

    固定各自权重不变，用累积平均在训练集上重新估计滑动统计，写回后重新评估验证集。

预期：两组重估后 mAOE 收敛到同一水平（0.72~0.75），且都接近未开启检查点的 S0 基线。
    → 因果链闭合，且把"两次独立训练的差分"变成"同一权重下的受控对比"，
      绕开单次运行的统计学困境。
若不收敛：说明 0.066 3 的 mAOE 差距另有来源，论文 4.4/4.6 的解释需要重写。

用法（在 /root/autodl-tmp/bevfusion 下，GPU 空闲时）：
    python tools/recalibrate_bn_and_eval.py 300

耗时：每组约 3 分钟重估 + 约 8 分钟评估，两组共约 25 分钟。
"""
import json
import subprocess
import sys
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

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
CFG = 'runs/s0_cp_bnfix_v1/input_config.yaml'
GROUPS = [('unfixed', 'runs/s0_cp_b2/epoch_6.pth'),
          ('fixed', 'runs/s0_cp_bnfix_v1/epoch_6.pth')]
OUT = Path('runs/bn_recalibration')
OUT.mkdir(exist_ok=True)

src = Path('mmdet3d/models/vtransforms/depth_lss.py').read_text()
assert 'BN_CP_FIX_V1' in src, '必须在已打补丁的代码上运行'

print('N_BATCHES=%d' % N, flush=True)
dist.init()
torch.cuda.set_device(0)
configs.load(CFG, recursive=True)
cfg = Config(recursive_eval(configs), filename=CFG)

print('building dataset ...', flush=True)
ds = build_dataset(cfg.data.train)
dl = build_dataloader(ds, samples_per_gpu=2, workers_per_gpu=4, dist=False,
                      shuffle=True, seed=0)
net = build_model(cfg.model)
model = MMDataParallel(net.cuda(), device_ids=[0])

# 评估配置：剥离教师，纯学生
ecfg = Config(recursive_eval(configs), filename=CFG)
ecfg['model']['type'] = 'BEVFusion'
for k in ('teacher_model_path', 'teacher_ckpt', 'distill'):
    ecfg['model'].pop(k, None)
ecfg['resume_from'] = ecfg['load_from'] = None
eval_cfg = OUT / 'eval_student.yaml'
import yaml
eval_cfg.write_text(yaml.safe_dump(dict(ecfg), allow_unicode=True, sort_keys=False))

results = {}
for tag, ckpt in GROUPS:
    print('\n=== %s : %s ===' % (tag, ckpt), flush=True)
    load_checkpoint(net, ckpt, map_location='cpu')

    # 重置全部 BN 并切到累积平均（无偏参考）
    n_bn = 0
    for m in net.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None
            n_bn += 1
    print('  已重置 %d 个 BatchNorm，开始重估 ...' % n_bn, flush=True)

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
            if done % 50 == 0:
                print('   %s %d/%d' % (tag, done, N), flush=True)

    # 保存重估后的纯学生权重
    state = {k[7:] if k.startswith('module.') else k: v
             for k, v in net.state_dict().items()}
    state = {k: v for k, v in state.items() if not k.startswith('teacher.')}
    out_ckpt = OUT / ('recalibrated_%s.pth' % tag)
    torch.save({'meta': {'recalibrated_from': ckpt, 'n_batches': N},
                'state_dict': state}, out_ckpt)
    print('  已保存 %s' % out_ckpt, flush=True)

    # 评估
    log = OUT / ('eval_%s.log' % tag)
    print('  评估中 ...', flush=True)
    with log.open('w') as f:
        rc = subprocess.run([sys.executable, 'tools/test.py', str(eval_cfg),
                             str(out_ckpt), '--eval', 'bbox'],
                            stdout=f, stderr=subprocess.STDOUT).returncode
        f.write('\nEVAL_EXIT_CODE=%d\n' % rc)
    txt = log.read_text(errors='replace').replace('\r', '\n')
    import re
    rec = {}
    for key in ('mAP', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE', 'NDS'):
        found = re.findall(r'\b%s:\s*([0-9.]+)' % key, txt)
        if found:
            rec[key] = float(found[-1])
    rec['exit'] = rc
    results[tag] = rec
    print('  %s -> %s' % (tag, rec), flush=True)

# ── 对照 ──
BASE = {'S0（未开检查点）': dict(mAOE=0.7247, NDS=0.4886691, mAP=0.5137257),
        'S0-CP（原始评估）': dict(mAOE=0.7910, NDS=0.4800992, mAP=0.5139128),
        'S0-CP-Fix（原始评估）': dict(mAOE=0.7524, NDS=0.4840, mAP=0.5152)}
print('\n' + '=' * 72)
print('BN 重估前后对照')
print('=' * 72)
print('%-26s %-9s %-9s %-9s' % ('组', 'mAP', 'mAOE', 'NDS'))
for k, v in BASE.items():
    print('%-26s %-9.4f %-9.4f %-9.4f' % (k, v['mAP'], v['mAOE'], v['NDS']))
for tag, _ in GROUPS:
    r = results.get(tag, {})
    if 'mAOE' in r:
        print('%-26s %-9.4f %-9.4f %-9.4f'
              % ('重估后 ' + tag, r['mAP'], r['mAOE'], r['NDS']))
if all('mAOE' in results.get(t, {}) for t, _ in GROUPS):
    gap = abs(results['unfixed']['mAOE'] - results['fixed']['mAOE'])
    print('\n重估后两组 mAOE 差距 = %.4f（重估前为 0.0386）' % gap)
    print('判读：差距显著缩小 → 支持“差异源于滑动统计”的因果解释；')
    print('      差距依旧 → 该差异另有来源，论文 4.4/4.6 需重写。')

(OUT / 'result.json').write_text(json.dumps(
    dict(n_batches=N, baseline=BASE, recalibrated=results), indent=2, ensure_ascii=False))
print('\n详细结果: %s' % (OUT / 'result.json'))
print('BN_RECALIBRATION_DONE')
