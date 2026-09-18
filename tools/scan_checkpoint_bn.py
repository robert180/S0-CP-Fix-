# -*- coding: utf-8 -*-
"""静态扫描 v2：以"被检查点包裹的函数体内是否调用了带滑动统计的归一化层"为判据。

对每个 checkpoint 调用：
  1. 定位被包裹的函数体（嵌套函数 / self 方法 / self 子模块 / lambda）
  2. 收集函数体内引用的 self.<attr>
  3. 在所属类中判定这些 attr 的归一化类型：
     BN      明确的 BatchNorm 类，或 build_norm_layer 且该类 __init__ 默认 norm_cfg 为 BN/SyncBN
     NON_BN  LayerNorm / GroupNorm，或 build_norm_layer 且默认 norm_cfg 为 LN/GN
"""
import ast
import json
import os
import re
import sys
from collections import defaultdict

BN_CLASS = re.compile(r'\b(?:Sync)?BatchNorm[123]d\b|\bSyncBatchNorm\b|\bNaiveSyncBatchNorm')
NONBN_CLASS = re.compile(r'\bLayerNorm2?d?\b|\bGroupNorm\b|\bRMSNorm\b')
CFG_BN = re.compile(r"type=['\"](?:BN[123]?d?|SyncBN|naiveSyncBN[123]?d?)['\"]")
CFG_NONBN = re.compile(r"type=['\"](?:LN|GN|LN2d)['\"]")
NORMISH = re.compile(r'norm|bn|batch_norm', re.I)
CKPT = {'checkpoint', 'checkpoint_sequential'}


def seg(node, lines):
    try:
        return '\n'.join(lines[node.lineno - 1:node.end_lineno])
    except Exception:
        return ''


def dotted(call):
    f, parts = call.func, []
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
    return '.'.join(reversed(parts))


def scan(path, repo):
    try:
        text = open(path, encoding='utf-8', errors='replace').read()
        tree = ast.parse(text)
    except Exception:
        return []
    lines = text.split('\n')
    out = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        init = next((f for f in cls.body
                     if isinstance(f, ast.FunctionDef) and f.name == '__init__'), None)
        init_src = seg(init, lines) if init else ''
        methods = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}
        # self.<attr> 的构造表达式（用于判断子模块内部是否含 BN）
        attr_ctor = {}
        if init is not None:
            for asg in ast.walk(init):
                if isinstance(asg, ast.Assign):
                    for t in asg.targets:
                        if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                                and t.value.id == 'self'):
                            attr_ctor[t.attr] = seg(asg.value, lines)
        # 类默认 norm 语义
        cls_norm = ('BN' if (BN_CLASS.search(init_src) or CFG_BN.search(init_src))
                    else 'NON_BN' if (NONBN_CLASS.search(init_src) or CFG_NONBN.search(init_src))
                    else '?')
        if BN_CLASS.search(init_src) or CFG_BN.search(init_src):
            cls_norm = 'BN'          # BN 证据优先（同时出现时以 BN 为准，后续人工复核）

        for call in [n for n in ast.walk(cls) if isinstance(n, ast.Call)]:
            if dotted(call).split('.')[-1] not in CKPT or not call.args:
                continue
            a = call.args[0]
            body, tname = None, None
            if isinstance(a, ast.Name):
                tname = a.id
                for fn in ast.walk(cls):
                    if isinstance(fn, ast.FunctionDef) and fn.name == tname:
                        body = fn
                        break
            elif isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name) \
                    and a.value.id == 'self':
                tname = 'self.' + a.attr
                body = methods.get(a.attr)
                if body is None:
                    body = init          # self.<module> 被整体包裹，退回看 __init__
            elif isinstance(a, ast.Lambda):
                tname, body = 'lambda', a
            if body is None:
                continue
            bsrc = seg(body, lines)
            attrs = {n.attr for n in ast.walk(body)
                     if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                     and n.value.id == 'self'}
            # 一层展开：被包裹函数调用的同类方法，其函数体也纳入
            for extra in list(attrs):
                if extra in methods and methods[extra] is not body:
                    sub = methods[extra]
                    bsrc += '\n' + seg(sub, lines)
                    attrs |= {n.attr for n in ast.walk(sub)
                              if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                              and n.value.id == 'self'}
            # 被包裹函数调用的 self.<attr>，其构造表达式内若含 BatchNorm 也算命中
            ctor_bn = [x for x in attrs if BN_CLASS.search(attr_ctor.get(x, ''))]
            touches_norm = (bool(BN_CLASS.search(bsrc)) or any(NORMISH.search(x) for x in attrs)
                            or bool(ctor_bn))
            if ctor_bn:
                cls_norm = 'BN'
            verdict = ('BN' if (touches_norm and cls_norm == 'BN') else
                       'NON_BN' if (touches_norm and cls_norm == 'NON_BN') else
                       'NO_NORM' if not touches_norm else 'UNCERTAIN')
            out.append(dict(repo=repo, file=os.path.relpath(path, repo), line=call.lineno,
                            cls=cls.name, target=tname, verdict=verdict,
                            norm_attrs=sorted(set(list(ctor_bn) +
                                                  [x for x in attrs if NORMISH.search(x)]))[:4]))
    return out


allhits = []
for root in sys.argv[1:]:
    for dp, dn, fns in os.walk(root):
        dn[:] = [d for d in dn if d not in ('.git', 'tests', 'test', 'docs', '.github')]
        for fn in fns:
            if fn.endswith('.py'):
                allhits += scan(os.path.join(dp, fn), root)

# 去重（同一 类+文件+行 只算一次）
uniq = {}
for h in allhits:
    uniq[(h['repo'], h['file'], h['line'], h['cls'])] = h
allhits = list(uniq.values())
json.dump(allhits, open('scan_v2.json', 'w'), ensure_ascii=False, indent=1)

by = defaultdict(lambda: defaultdict(int))
for h in allhits:
    by[h['repo']][h['verdict']] += 1
cols = ['BN', 'NON_BN', 'NO_NORM', 'UNCERTAIN']
print('%-22s %s   合计' % ('仓库', ' '.join('%-9s' % c for c in cols)))
tot = defaultdict(int)
for r in sorted(by):
    print('%-22s %s   %d' % (r, ' '.join('%-9d' % by[r].get(c, 0) for c in cols),
                             sum(by[r].values())))
    for c in cols:
        tot[c] += by[r].get(c, 0)
print('%-22s %s   %d' % ('总计', ' '.join('%-9d' % tot[c] for c in cols), sum(tot.values())))

print('\n=== 判定为 BN 的命中（受影响） ===')
for h in sorted(allhits, key=lambda x: (x['repo'], x['file'], x['line'])):
    if h['verdict'] == 'BN':
        print('  %-14s %-46s:%-5d %-26s %s' %
              (h['repo'], h['file'], h['line'], h['cls'], ','.join(h['norm_attrs'])))
