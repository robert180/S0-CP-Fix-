"""只读实验状态采集：输出单行 JSON，供 Codex 巡检与 Claude 判读。

用法（在 /root/autodl-tmp/bevfusion 下）：
    python tools/exp_status.py runs/s0_cp_bnfix_v1 experiments/bnfix_v1/s0_driver.log

不修改任何文件，不占 GPU，秒级返回。可安全地在训练进行中反复调用。
"""
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs/s0_cp_bnfix_v1')
DRIVER = Path(sys.argv[2] if len(sys.argv) > 2 else 'experiments/bnfix_v1/s0_driver.log')
DONE_MARK = sys.argv[3] if len(sys.argv) > 3 else 'BNFIX_EXPERIMENT_DONE'
EPOCHS = 6
EVAL_EPOCHS = (4, 5, 6)
KEYS = ['mAP', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE', 'NDS']

out = {'run': str(RUN), 'now': time.strftime('%Y-%m-%d %H:%M:%S')}


def running(pattern):
    try:
        return subprocess.run(['pgrep', '-f', pattern],
                              capture_output=True, text=True).returncode == 0
    except Exception:
        return None


out['proc'] = {
    'driver': running('[r]un_bnfix_experiment.py'),
    'train': running('[t]ools/train.py'),
    'eval': running('[t]ools/test.py'),
}

try:
    out['manifest_status'] = json.loads((RUN / 'manifest.json').read_text())['status']
except Exception as exc:
    out['manifest_status'] = 'ERR:%s' % exc

driver_txt = DRIVER.read_text(errors='replace') if DRIVER.exists() else ''
out['finished'] = DONE_MARK in driver_txt
out['driver_tail'] = driver_txt.strip().splitlines()[-4:] if driver_txt.strip() else []

logs = glob.glob(str(RUN / '2*.log'))
if logs:
    newest = max(logs, key=os.path.getmtime)
    last = None
    nan_loss = grad_overflow = n_iter_lines = 0
    for line in open(newest, errors='replace'):
        if 'Epoch [' not in line:
            continue
        n_iter_lines += 1
        if re.search(r'loss[a-z_/-]*: *-?(nan|inf)', line):
            nan_loss += 1
        if re.search(r'grad_norm: *-?(nan|inf)', line):
            grad_overflow += 1
        m = re.search(r'Epoch \[(\d+)\]\[(\d+)/(\d+)\].*?time: ([\d.]+)', line)
        if m:
            last = m
    if last:
        ep, it, tot, sec = int(last[1]), int(last[2]), int(last[3]), float(last[4])
        done, total = (ep - 1) * tot + it, EPOCHS * tot
        out['progress'] = {
            'epoch': ep, 'epochs': EPOCHS, 'iter': it, 'iters_per_epoch': tot,
            'pct': round(done / total * 100, 2), 'sec_per_iter': sec,
            'eta_hours': round(max(total - done, 0) * sec / 3600, 2),
        }
    out['log_file'] = os.path.basename(newest)
    out['log_age_min'] = round((time.time() - os.path.getmtime(newest)) / 60, 1)
    out['health'] = {
        'iter_log_lines': n_iter_lines,
        'nan_loss_lines': nan_loss,           # 必须为 0
        'grad_overflow_lines': grad_overflow,  # AMP 稳态，约 20/轮，正常
    }

out['checkpoints'] = sorted(p.name for p in RUN.glob('epoch_*.pth'))

out['eval'] = {}
for e in EVAL_EPOCHS:
    f = RUN / ('eval_epoch_%d.log' % e)
    if not f.exists():
        continue
    txt = f.read_text(errors='replace').replace('\r', '\n')
    rec = {}
    for k in KEYS:
        found = re.findall(r'\b%s:\s*([0-9.]+)' % k, txt)
        if found:
            rec[k] = float(found[-1])
    codes = re.findall(r'EVAL_EXIT_CODE=(-?\d+)', txt)
    rec['exit'] = int(codes[-1]) if codes else None
    out['eval']['epoch_%d' % e] = rec

try:
    out['gpu'] = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True).stdout.strip()
except Exception:
    out['gpu'] = None

try:
    df = subprocess.run(['df', '-BG', '--output=avail', '/root/autodl-tmp'],
                        capture_output=True, text=True).stdout.split()
    out['disk_avail'] = df[-1] if df else None
except Exception:
    out['disk_avail'] = None

print(json.dumps(out, ensure_ascii=False))
