# 上游提交文本（三份）

**提交策略**：PyTorch #96136 已准确描述现象但三年半无人回应，不宜新开重复 issue。
因此 (A) 在 #96136 下补充评论，提供它缺少的量化数据与可用修复；
(B)(C) 给 mmdetection / mmsegmentation 新开 issue，告知下游其 `with_cp` 受此影响——
这两个仓库的用户目前无从知晓，属于新信息。

建议顺序：先 A（最可能推动上游），再 B、C。

---

## 最小复现脚本（三份都用它，贴在 issue 里）

```python
# bn_checkpoint_repro.py  —  requires only torch + torchvision
import torch, torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.models.resnet import BasicBlock

def run(mode):
    torch.manual_seed(0)
    blk = BasicBlock(16, 16).train()
    torch.manual_seed(1)
    x = torch.randn(4, 16, 8, 8, requires_grad=True)
    before = {n: int(m.num_batches_tracked)
              for n, m in blk.named_modules() if isinstance(m, nn.BatchNorm2d)}
    if mode == "none":
        out = blk(x)
    else:
        out = checkpoint(blk, x, use_reentrant=(mode == "reentrant"))
    out.sum().backward()
    delta = {n: int(m.num_batches_tracked) - before[n]
             for n, m in blk.named_modules() if isinstance(m, nn.BatchNorm2d)}
    rm = {n: m.running_mean.clone()
          for n, m in blk.named_modules() if isinstance(m, nn.BatchNorm2d)}
    return delta, rm

base_d, base_rm = run("none")
for mode in ("none", "reentrant", "non-reentrant"):
    d, rm = run(mode)
    drift = max((rm[n] - base_rm[n]).abs().max().item() for n in rm)
    print(f"{mode:>14}  num_batches_tracked delta={sorted(set(d.values()))}  "
          f"running_mean drift vs no-checkpoint={drift:.3e}")
```

输出：

```
          none  num_batches_tracked delta=[1]  running_mean drift vs no-checkpoint=0.000e+00
     reentrant  num_batches_tracked delta=[2]  running_mean drift vs no-checkpoint=6.921e-02
 non-reentrant  num_batches_tracked delta=[2]  running_mean drift vs no-checkpoint=6.921e-02
```

---

## A. 在 pytorch/pytorch#96136 下补充的评论

> Still reproduces on torch 2.14.0 / torchvision 0.29.0. Adding quantitative data and a
> working workaround, since this issue has had no activity since it was opened.
>
> **1. The effect has a closed form.** Both forwards see the same input tensor, so the batch
> statistics `b` are identical, and the two updates compose into a single one:
>
> ```
> r₂ = (1−m)²·r₀ + [1−(1−m)²]·b
> ```
>
> which is exactly one update with an effective momentum
>
> ```
> m_eff = 1 − (1−m)² = 2m − m²
> ```
>
> For the default `m = 0.1` this gives `m_eff = 0.19`. Consequences, assuming i.i.d. batch
> statistics: the steady-state variance of the running estimate, `m·σ²/(2−m)`, rises from
> `0.05263σ²` to `0.10497σ²` (**×1.99**), and the effective number of averaged samples,
> `(2−m)/m`, drops from **19 to 9.5**. I verified the equivalence numerically at
> `m ∈ {0.01, 0.05, 0.10, 0.30}`; the difference between "two updates at m" and "one update
> at 2m−m²" is 6e−8…2e−7 (float32 round-off), while the difference against the correct single
> update is six orders of magnitude larger.
>
> Note this only inflates the *variance* of the running estimates — the expectation is
> unchanged, so it is not a systematic bias.
>
> **2. How far the running stats actually drift in real training.** I trained the same
> detector twice on nuScenes (6 epochs, 61790 iters/epoch, batch 2 × grad-accum 4, AMP),
> once with the double update and once with it fixed. To measure contamination I froze each
> checkpoint's weights, reset all BN running stats, set `momentum=None` (cumulative average),
> ran 300 batches to obtain an unbiased reference, and measured how far the stored stats
> deviate from it (mean shift in units of reference std):
>
> | BN group | layers | unfixed | fixed | ratio |
> |---|---:|---:|---:|---:|
> | inside the checkpointed region | 5 | 0.0248 | 0.0129 | **1.93** |
> | rest of the network (control) | 46 | 0.1550 | 0.1522 | **1.01** |
>
> The control group ratio of 1.01 rules out seeds / data order / cuDNN nondeterminism as the
> cause — only the checkpointed layers move.
>
> **2b. But the end-to-end cost is negligible, at least here.** To avoid overselling this: I
> also ran a paired within-run test — same weights, same data order, same seed, trained
> statistics kept, each layer's configured momentum kept, and the only difference being whether
> the affected layers are updated once or twice per iteration. Across all seven nuScenes
> metrics the two arms differ by at most 0.0018, with mixed signs (mAOE −0.0015, NDS +0.0002).
> The 0.0386 mAOE gap I see between two *independent* training runs is therefore run-to-run
> variance, not this bug. So the case for fixing it is correctness and reproducibility — the
> same code should record the same running statistics with and without `with_cp` — not accuracy.
> I'd rather state that plainly than have someone merge this expecting a metric win.
>
> **3. How widespread the pattern is.** Scanning six widely used repositories for
> `checkpoint(...)` calls whose wrapped region touches a normalization layer with persistent
> running stats, then hand-checking each hit: **20 of 33 call sites are affected**, covering
> ResNet / Res2Net / ResNeSt / TridentNet / DetectoRS / EfficientNet / MobileNetV2-V3 /
> U-Net / CGNet / DenseNet / VoVNet. Transformer backbones (Swin, MiT, ViT) are not affected
> because LayerNorm keeps no running state. `torchvision`'s own DenseNet triggers it through
> its public `memory_efficient=True` flag.
>
> **4. `use_reentrant=False` is affected too, and it breaks the existing workaround.** Both
> passes of the non-reentrant implementation run with grad enabled. Any mitigation that keys
> off `torch.is_grad_enabled()` — including FairScale's `patch_batchnorm`, the only packaged
> mitigation I could find — therefore returns early on *both* passes and silently does nothing
> there. That condition is sound for FairScale's own `CheckpointFunction`, which runs the first
> pass under an explicit `torch.no_grad()`; it stops being sound the moment the same helper is
> paired with `torch.utils.checkpoint` under the now-recommended `use_reentrant=False`.
>
> Discriminating on **call order** instead is correct on both paths. 5 training iterations on a
> single conv+BN block, lr=0, fixed seed, reporting `num_batches_tracked` / `sum(|running_mean|)`:
>
> | implementation | `use_reentrant=True` | `use_reentrant=False` |
> |---|---|---|
> | no checkpointing (reference) | 5 / 0.03902 | — |
> | stock `cp.checkpoint` | 10 / 0.05563 | 10 / 0.05563 |
> | FairScale `patch_batchnorm` | 5 / 0.03902 | 10 / 0.05563 |
> | call-order discriminator (below) | 5 / 0.03902 | 5 / 0.03902 |
>
> ```python
> def bn_safe_checkpoint(function, *args, module=None, **kwargs):
>     if module is not None and module.training:
>         function = _protect(function, module)
>     return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)
>
> def _protect(function, module):
>     # A fresh closure per checkpoint call, so the flag tracks the two passes
>     # of that one call. Works under use_reentrant=False, where a grad-mode
>     # test would suppress both updates.
>     state = {"first_pass_done": False}
>     def wrapper(*args, **kwargs):
>         if not state["first_pass_done"]:      # first forward: update normally
>             state["first_pass_done"] = True
>             return function(*args, **kwargs)
>         saved = []                            # recomputation: protect the buffers
>         try:
>             for layer in module.modules():
>                 if isinstance(layer, nn.modules.batchnorm._BatchNorm):
>                     for name in ("running_mean", "running_var", "num_batches_tracked"):
>                         v = getattr(layer, name, None)
>                         if v is not None:
>                             saved.append((layer, name, v))
>                             setattr(layer, name, v.detach().clone())
>             return function(*args, **kwargs)
>         finally:
>             for layer, name, v in reversed(saved):
>                 setattr(layer, name, v)
>     return wrapper
> ```
>
> BN stays in training mode and keeps normalizing by batch statistics, so forward outputs and
> gradients are bit-identical; only the module attributes are rebound, with no in-place write
> to the original tensors, so autograd version counters are untouched. Measured training cost
> in a 370740-iteration run: 0.467 vs 0.470 s/iter — no measurable overhead.
>
> One design note: this keeps the *first* forward's statistics, whereas FairScale keeps the
> recomputed ones. When the two passes are bit-identical it makes no difference; under AMP I
> measure a relative L2 of ~1.16e−3 between them, and keeping the first matches what a
> non-checkpointed run records, since that is the pass whose output actually feeds the network.
>
> **5. A one-line self-check** for anyone wanting to know whether their model is affected:
> record `num_batches_tracked` per BN layer before and after one training iteration; any delta
> greater than 1 marks a contaminated layer.

---

## B. open-mmlab/mmdetection 新 issue

**Title**

```
with_cp=True updates BatchNorm running stats twice per iteration (upstream pytorch#96136)
```

**Body**

> ### Describe the bug
>
> When `with_cp=True`, the block's `_inner_forward` is executed twice per training iteration
> (once under `no_grad`, once during recomputation). BatchNorm's running statistics are updated
> on both passes, so the same batch is accumulated twice. This does not affect training-time
> normalization, forward outputs or gradients — it only corrupts the running statistics used at
> inference time.
>
> This is the downstream manifestation of pytorch/pytorch#96136, which has been open since
> March 2023 with no response. I am filing here because mmdetection users have no way of
> knowing their `with_cp` runs are affected.
>
> ### Affected call sites in this repository
>
> Verified by resolving each wrapped region and checking the class's default `norm_cfg`:
>
> | File | Line | Class | Backbone |
> |---|---:|---|---|
> | `mmdet/models/backbones/resnet.py` | 88 | `BasicBlock` | ResNet-18/34 |
> | `mmdet/models/backbones/resnet.py` | 296 | `Bottleneck` | ResNet-50/101/152 |
> | `mmdet/models/backbones/res2net.py` | 154 | `Bottle2neck` | Res2Net |
> | `mmdet/models/backbones/resnest.py` | 268 | `Bottleneck` | ResNeSt |
> | `mmdet/models/backbones/trident_resnet.py` | 172 | `TridentBottleneck` | TridentNet |
> | `mmdet/models/backbones/detectors_resnet.py` | 107 | `Bottleneck` | DetectoRS |
> | `mmdet/models/backbones/efficientnet.py` | 109 | `InvertedResidual` / `EdgeResidual` | EfficientNet |
> | `mmdet/models/layers/inverted_residual.py` | 126 | `InvertedResidual` | MobileNetV2/V3 |
>
> Swin (`mmdet/models/backbones/swin.py`) is **not** affected — LayerNorm keeps no running state.
>
> ### Reproduction
>
> [贴上面的最小复现脚本]
>
> `use_reentrant=False` behaves identically, so switching checkpoint implementations does not
> help — both need a second forward.
>
> ### Effect size
>
> The double update is algebraically equivalent to raising BN's effective momentum from `m` to
> `2m − m²` (0.1 → 0.19 at the default), which roughly doubles the steady-state variance of the
> running estimates and halves their effective sample count. In a 6-epoch detection training run
> I measured the stored statistics of affected layers deviating from an unbiased reference
> **1.93×** as much as in a fixed run, while 46 unaffected BN layers in the same network showed
> a ratio of 1.01.
>
> ### Suggested fix
>
> Protect the buffers during recomputation rather than disabling tracking on the first pass —
> see the snippet in pytorch/pytorch#96136. Measured cost: none (0.467 vs 0.470 s/iter).
>
> Happy to send a PR if the maintainers agree on the approach.

---

## C. open-mmlab/mmsegmentation 新 issue

同 B，替换受影响清单为：

| File | Line | Class | Backbone |
|---|---:|---|---|
| `mmseg/models/backbones/resnet.py` | 90 | `BasicBlock` | ResNet |
| `mmseg/models/backbones/resnet.py` | 301 | `Bottleneck` | ResNet |
| `mmseg/models/backbones/resnest.py` | 261 | `Bottleneck` | ResNeSt |
| `mmseg/models/backbones/cgnet.py` | 164 | `ContextGuidedBlock` | CGNet |
| `mmseg/models/backbones/unet.py` | 81 | `BasicConvBlock` | U-Net |
| `mmseg/models/utils/inverted_residual.py` | 95 | `InvertedResidual` | MobileNetV2 |
| `mmseg/models/utils/inverted_residual.py` | 209 | `InvertedResidualV3` | MobileNetV3 |

未受影响（对照）：`swin.py`、`mit.py`、`vit.py` 使用 LayerNorm；
`cgnet.py:47` 的 `GlobalContextExtractor` 无归一化层。

---

## 提交前的注意事项

1. **不要在 mmdet/mmseg 的 issue 里声称这是新发现的 bug**——正文已写明是 pytorch#96136 的下游表现，
   这样既准确，也避免维护者认为重复报告。
2. **B/C 末尾的 PR 意向是可选的**。如果不打算写 PR，删掉那句，以免后续被追问。
3. 实测数字（1.93 / 1.01 / 0.467 vs 0.470）来自你的 nuScenes 实验。若不愿在论文见刊前公开细节，
   可以把表格换成"在一个 6 epoch 的检测训练中观察到约 1.9 倍的偏离差异"这类粗粒度表述，
   或等论文录用后再提交并附引用。
4. 提交后记下 issue 编号，写进论文 4.5 节作为实用价值的直接证据。

---

# D. PR 补丁（已生成并验证）

两个 `.patch` 文件可直接用 `git apply` 打到各自仓库的 main 分支：

```bash
git clone https://github.com/open-mmlab/mmdetection.git && cd mmdetection
git checkout -b fix/bn-stats-under-gradient-checkpointing
git apply /path/to/mmdetection_bn_safe_checkpoint.patch
git add -A && git commit -m "Fix BatchNorm running stats being updated twice under with_cp"
```

mmsegmentation 同理，换对应 patch 文件与分支名。

## 补丁内容

| 仓库 | 新增 | 改动调用点 | 净变更 |
|---|---|---:|---|
| mmdetection | `mmdet/models/layers/bn_safe_checkpoint.py`（72 行） | 8 处 | 9 files, +89 −18 |
| mmsegmentation | `mmseg/models/utils/bn_safe_checkpoint.py`（72 行） | 9 处 | 7 files, +88 −16 |

每个调用点的改动都是一行：

```diff
-            out = cp.checkpoint(_inner_forward, x)
+            out = bn_safe_checkpoint(_inner_forward, x, module=self)
```

`cp` 在全部调用点被替换后即成为未使用 import，补丁一并移除；`bn_safe_checkpoint` 经各仓库
`__init__.py` 导出，import 已按各自的 isort 配置归入第一方组。

## 已完成的验证

1. **单元验证**（`test_bn_safe_checkpoint.py`，torchvision BasicBlock/Bottleneck）：
   修复后 BN 每迭代更新 **1** 次，输出、梯度、running_mean、running_var 与"不使用检查点"
   **逐位相同**（全部 0.000e+00）；未修复时为 2 次。
2. **reentrant 与 non-reentrant 都覆盖**。早期版本用 `torch.is_grad_enabled()` 判别，在
   `use_reentrant=False` 下两次前向都处于 grad 模式，会把两次更新都抑制掉（变成 0 次）——
   现改为按调用次序判别首次前向，两种模式均正确。
3. **drop-in 安全性**：不传 `module=` 时退化为原 `cp.checkpoint` 行为；`eval()` 模式下不做
   任何额外操作，输出与直接前向一致。
4. **端到端**：直接加载打了补丁的 `mmdet/models/backbones/resnet.py`，构造真实的
   `BasicBlock` / `Bottleneck`（`with_cp=True`）跑一次训练迭代，BN 更新 1 次，
   running_mean 与 `with_cp=False` 路径差 0.000e+00。
5. **Lint**：两个仓库改动文件的 isort 与 flake8 均通过（`mmseg/models/backbones/unet.py`
   的两处 E231 为仓库原有，补丁前同样报出，非本补丁引入）。
6. **开销**：交替 5 轮 × 40 次取中位数，6.387 → 6.360 ms/iter，相对 −0.4%，在噪声范围内。

## 提 PR 时建议附上的说明

> The wrapped region is executed twice per training iteration, and BatchNorm updates its
> running statistics on both passes, so the same batch is accumulated twice — equivalent to
> raising BN's momentum from `m` to `2m - m**2` (0.1 → 0.19 at the default). This is the
> downstream effect of pytorch/pytorch#96136.
>
> The helper protects the buffers during the recomputation pass, keeping the statistics from
> the pass whose output actually feeds the network. Forward outputs and gradients are
> bit-identical to the unpatched code; only the running statistics change, and they now match
> a `with_cp=False` run exactly. Passing no `module=` falls through to the stock behaviour, so
> the helper is a safe drop-in.
>
> **Prior art.** This is not a new idea: FairScale ships `patch_batchnorm`
> (`fairscale/nn/checkpoint/checkpoint_utils.py`), which uses forward hooks to disable
> `track_running_stats` while `torch.is_grad_enabled()` is false. That condition holds for
> FairScale's own `CheckpointFunction`, which runs the first pass under an explicit
> `torch.no_grad()`. It does **not** hold for `torch.utils.checkpoint` with
> `use_reentrant=False`, where both passes run with grad enabled — both hooks return early and
> the suppression silently does nothing. Since PyTorch now recommends `use_reentrant=False`,
> this helper discriminates on call order instead, which is correct on both paths.
>
> Measured over 5 training iterations on a single conv+BN block (lr=0, fixed seed), reporting
> `num_batches_tracked` / `sum(|running_mean|)`:
>
> | implementation | `use_reentrant=True` | `use_reentrant=False` |
> |---|---|---|
> | no checkpointing (reference) | 5 / 0.03902 | — |
> | stock `cp.checkpoint` | 10 / 0.05563 | 10 / 0.05563 |
> | FairScale `patch_batchnorm` | 5 / 0.03902 | 10 / 0.05563 |
> | this helper | 5 / 0.03902 | 5 / 0.03902 |
>
> Repro script attached (`fairscale_vs_ours.py`), torch 2.14.0.

## 一处需要你判断的

`module=self` 会让 helper 遍历 `self.modules()` 找 BN。对 `BasicBlock` 这类小模块开销可忽略
（实测 −0.4%），但若维护者担心大模块上的遍历成本，可改为在 `__init__` 里缓存 BN 列表。
我没有预先这样做，因为那会增加改动面，而实测开销并不存在。若审阅时被提出，再改不迟。
