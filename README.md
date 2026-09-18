# Gradient checkpointing corrupts BatchNorm running statistics

Code and data for the paper *面向紧凑多模态BEV检测器的训练实现审计：梯度检查点导致的
BatchNorm统计污染及其修复*.

## The defect

`torch.utils.checkpoint` runs the wrapped function **twice** per training
iteration — once to produce the output, once to rebuild the graph during
backward. BatchNorm updates its running statistics on every forward, so any
BatchNorm inside a checkpointed region accumulates the same batch twice.

Two updates with the same batch statistics `b` compose into one:

```
r₂ = (1−m)²·r₀ + [1−(1−m)²]·b      →      m_eff = 2m − m²
```

At PyTorch's default `m = 0.1` that is `m_eff = 0.19`: the steady-state
variance of the running estimate roughly doubles (0.05263σ² → 0.10497σ²) and
the effective number of averaged samples drops from 19 to 9.5. The expectation
is unchanged, so this is added variance, not bias.

Forward outputs and gradients are unaffected. Only the statistics used at
inference change.

Both `use_reentrant=True` and `use_reentrant=False` are affected — upgrading
does not help. See [pytorch/pytorch#96136](https://github.com/pytorch/pytorch/issues/96136)
(open since 2023) and [facebookresearch/fairscale#1035](https://github.com/facebookresearch/fairscale/issues/1035).

## What this repository adds

The defect and the idea of suppressing one update are **not new** — FairScale
has shipped `patch_batchnorm` for years. What is new here:

1. **The existing mitigation breaks on the path PyTorch now recommends.**
   FairScale keys off `torch.is_grad_enabled()`, which is true only for its own
   reentrant `CheckpointFunction`. Under `use_reentrant=False` both passes run
   with grad enabled, both hooks return early, and the suppression silently does
   nothing. Discriminating on **call order** instead is correct on both paths.

   5 iterations on one conv+BN block, `num_batches_tracked / sum(|running_mean|)`:

   | implementation | `use_reentrant=True` | `use_reentrant=False` |
   |---|---|---|
   | no checkpointing (reference) | 5 / 0.03902 | — |
   | stock `cp.checkpoint` | 10 / 0.05563 | 10 / 0.05563 |
   | FairScale `patch_batchnorm` | 5 / 0.03902 | **10 / 0.05563** |
   | this fix | 5 / 0.03902 | 5 / 0.03902 |

   Reproduce: `python tools/compare_fairscale.py`

2. **Quantified contamination in a real training run**, with an internal
   control group. Freezing the weights and re-estimating the statistics by
   cumulative averaging gives an unbiased reference; the stored statistics
   deviate from it by 1.93× more in the 5 affected layers than before the fix,
   while the 46 unaffected layers give a control ratio of 1.01.

3. **A negative result on the end-to-end effect** (below).

4. **Applicability boundaries**, measured rather than assumed (below).

5. **Working patches** for MMDetection and MMSegmentation, in `patches/`.

## The end-to-end effect is negligible — read this before citing gains

Three independent 6-epoch runs differ by 0.0386 in mAOE between the unfixed and
fixed variants. **That difference is not caused by the defect.**

A paired within-run experiment settles it: same weights, same data order, same
seed, trained statistics kept, each layer's configured momentum kept (29 of the
51 BatchNorm layers use 0.01, 22 use 0.1; the 5 affected ones are 0.1). The only
difference is whether those 5 layers are updated once or twice per iteration.
The 46 layers outside the region receive identical updates in both arms and
cancel out.

| arm | mAP | mAOE | NDS |
|---|---|---|---|
| single (1×/iter) | 0.4782 | 0.7983 | 0.4560 |
| double (2×/iter) | 0.4786 | 0.7968 | 0.4562 |
| **Δ** | +0.0004 | **−0.0015** | +0.0002 |

Every metric moves by ≤ 0.0018, with mixed signs, and mAOE moves *opposite* to
the hypothesis. The isolated effect is ~25× smaller than the 0.0386 gap between
runs. So:

> **Fix this for correctness and reproducibility — the same code should produce
> the same running statistics with and without checkpointing — not because it
> buys accuracy. On this network it does not.**

Reproduce: `python experiments/paired_bn.py 50`

## A remedy that does not work

It is tempting to think an affected run can be salvaged without retraining:
freeze the weights and re-estimate the running statistics on the training set.
It does not work. With cumulative averaging over 300 batches, mAP collapses from
~0.515 to 0.29–0.33 for both variants. The weights were verified bit-identical
before and after (281 non-BN parameters unchanged, 153 BN buffers changed), so
this is a real property of the operation, not a bug in the script.

Affected runs need retraining with the fixed code.

Reproduce: `python experiments/recalibrate_bn_and_eval.py 300`

## Applicability boundaries

| setting | defect | this fix |
|---|---|---|
| eager, `use_reentrant=True` | present (2×/iter) | correct, bit-identical |
| eager, `use_reentrant=False` | present (2×/iter) | correct, bit-identical |
| `SyncBatchNorm` | present (2×/iter) | correct, bit-identical |
| `torch.compile` (inductor) | **absent** — AOTAutograd functionalizes the buffer mutation | unnecessary; causes a graph break |
| CUDA Graph capture | not applicable — checkpointing saves RNG state, which capture rejects | — |
| FSDP | untested (needs a GPU *and* torch ≥ 1.12; no such environment available) | — |

Reproduce: `python tools/boundary_compile.py` (CPU) and
`python tools/boundary_gpu.py` (needs a GPU).

Practically, the defect affects **eager-mode training** — which is what
MMDetection, MMSegmentation and BEVFusion all do.

## Using the fix

```python
from bn_safe_checkpoint import checkpoint as bn_safe_checkpoint

def forward(self, x):
    def _inner(x):
        return self.relu(self.bn(self.conv(x)))
    return bn_safe_checkpoint(_inner, x, module=self)   # pass the owning module
```

Passing no `module=` falls through to the stock implementation, so it is a safe
drop-in. BatchNorm stays in training mode and keeps normalizing by batch
statistics, so forward outputs and gradients are bit-identical; only the module
attributes are rebound, with no in-place write to the original tensors, so
autograd version counters are untouched.

Do **not** apply it on a `torch.compile` path — it is unnecessary there and
costs you the compiled region.

**Overhead.** On the real workload it is not measurable: 0.467 vs 0.470 s/iter
over a 370 740-iteration run, and −0.4% on a patch microbenchmark. On a *tiny*
block the picture is less flattering — `tests/test_bn_safe_checkpoint.py` on CPU
with a single ResNet Bottleneck reports +3.3%, because walking `self.modules()`
to find the BatchNorm layers is a fixed cost that only disappears against real
compute. If that matters for your case, cache the BatchNorm list in `__init__`
instead of walking the module tree on every call.

## Is my model affected?

```bash
python tools/detect_contamination.py     # self-test, exits 1 if it reproduces
```

Or, in your own training loop — this needs no knowledge of where checkpointing
is used:

```python
from tools.detect_contamination import snapshot, report

before = snapshot(model)
loss.backward()                  # one full training iteration
print(report(model, before))     # any layer advancing != 1 is contaminated
```

## Layout

```
bn_safe_checkpoint.py            the fix
tests/                           verification suite for the fix
tools/
  detect_contamination.py        framework-agnostic self-check
  scan_checkpoint_bn.py          static scanner for checkpoint+BN call sites
  runtime_verify.py              torchvision models on current PyTorch
  compare_fairscale.py           FairScale vs call-order discrimination
  boundary_compile.py            torch.compile / eager boundaries (CPU)
  boundary_gpu.py                SyncBatchNorm / CUDA Graph boundaries (GPU)
patches/                         MMDetection + MMSegmentation patches
experiments/
  paired_bn.py                   the paired within-run experiment
  recalibrate_bn_and_eval.py     the recalibration negative result
  results/results.json           every number reported above
  results/scan_results_zh.md     scan results (Chinese)
  results/upstream_reports_zh.md drafted upstream reports (Chinese)
```

The scripts under `experiments/` carry hard-coded paths into the BEVFusion run
directory (`runs/s0_cp_bnfix_v1/...`) and are meant to document exactly what was
run, not to be portable. Everything under `tools/` is standalone.

## Environment

Experiments: torch 1.10.2+cu113, single RTX 4090 24 GB, nuScenes, batch 2 ×
grad-accum 4, AMP, 6 epochs (~49 h per run).
Boundary checks: torch 2.14.0 (CPU) and torch 1.10.2 (GPU).

## Still to do

- Pick a license (the patches touch Apache-2.0 OpenMMLab code).
- Upstream: report to `pytorch/pytorch#96136`, open issues and PRs on
  MMDetection and MMSegmentation. Drafted text is in
  `experiments/results/upstream_reports_zh.md`.
