# 梯度检查点 + BatchNorm 模式扫描结果

扫描日期 2026-09-17。判据：定位每处 `torch.utils.checkpoint` 调用，解析被包裹函数体
（含一层方法展开与子模块构造表达式追溯），判断范围内是否引用带持久滑动统计的归一化层；
判据无法唯一确定的逐个回查所属类及其父类的 norm 配置默认值。

扫描范围：6 个仓库，共 33 处检查点调用，**20 处受影响**。

## 受影响（20 处，覆盖 12 类结构）

| 仓库 | 文件 | 行 | 类 | 结构 | 归一化来源 |
|---|---|---:|---|---|---|
| BEVFusion (MIT) | `mmdet3d/models/vtransforms/depth_lss.py` | 104 | `DepthLSSTransform` | DepthLSS 视图变换 | `nn.BatchNorm2d` |
| MMDetection3D | `projects/PETR/petr/vovnetcp.py` | 288 | `_OSA_module` | VoVNet | `nn.BatchNorm2d` |
| MMDetection | `mmdet/models/backbones/resnet.py` | 88 | `BasicBlock` | ResNet-18/34 | `norm_cfg=dict(type='BN')` |
| MMDetection | `mmdet/models/backbones/resnet.py` | 296 | `Bottleneck` | ResNet-50/101/152 | `norm_cfg=dict(type='BN')` |
| MMDetection | `mmdet/models/backbones/res2net.py` | 154 | `Bottle2neck` | Res2Net | `继承 _Bottleneck，BN` |
| MMDetection | `mmdet/models/backbones/resnest.py` | 268 | `Bottleneck` | ResNeSt | `norm_cfg=dict(type='BN')` |
| MMDetection | `mmdet/models/backbones/trident_resnet.py` | 172 | `TridentBottleneck` | TridentNet | `norm_cfg=dict(type='BN')` |
| MMDetection | `mmdet/models/backbones/detectors_resnet.py` | 107 | `Bottleneck` | DetectoRS | `norm_cfg=dict(type='BN')` |
| MMDetection | `mmdet/models/backbones/efficientnet.py` | 109 | `InvertedResidual/EdgeResidual` | EfficientNet | `norm_cfg=dict(type='BN')` |
| MMDetection | `mmdet/models/layers/inverted_residual.py` | 126 | `InvertedResidual` | MobileNetV2/V3 | `norm_cfg=dict(type='BN')` |
| MMSegmentation | `mmseg/models/backbones/resnet.py` | 90 | `BasicBlock` | ResNet | `norm_cfg=dict(type='BN')` |
| MMSegmentation | `mmseg/models/backbones/resnet.py` | 301 | `Bottleneck` | ResNet | `norm_cfg=dict(type='BN')` |
| MMSegmentation | `mmseg/models/backbones/resnest.py` | 261 | `Bottleneck` | ResNeSt | `norm_cfg=dict(type='BN')` |
| MMSegmentation | `mmseg/models/backbones/cgnet.py` | 164 | `ContextGuidedBlock` | CGNet | `norm_cfg=dict(type='BN')` |
| MMSegmentation | `mmseg/models/backbones/unet.py` | 81 | `BasicConvBlock` | U-Net | `norm_cfg=dict(type='BN')` |
| MMSegmentation | `mmseg/models/backbones/unet.py` | 142 | `DeconvModule` | U-Net 上采样 | `norm_cfg=dict(type='BN')` |
| MMSegmentation | `mmseg/models/backbones/unet.py` | 216 | `InterpConv` | U-Net 上采样 | `norm_cfg=dict(type='BN')` |
| MMSegmentation | `mmseg/models/utils/inverted_residual.py` | 95 | `InvertedResidual` | MobileNetV2 | `norm_cfg=dict(type='BN')` |
| MMSegmentation | `mmseg/models/utils/inverted_residual.py` | 209 | `InvertedResidualV3` | MobileNetV3 | `norm_cfg=dict(type='BN')` |
| timm | `timm/models/densenet.py` | 80 | `DenseLayer` | DenseNet | `norm_layer=BatchNormAct2d` |

## 不受影响（对照）

| 仓库 | 文件 | 行 | 类 | 结构 | 原因 |
|---|---|---:|---|---|---|
| MMDetection | `mmdet/models/backbones/swin.py` | 375 | `SwinBlock` | Swin | norm_cfg=dict(type='LN') — 不受影响 |
| MMSegmentation | `mmseg/models/backbones/swin.py` | 373 | `SwinBlock` | Swin | LN — 不受影响 |
| MMSegmentation | `mmseg/models/backbones/mit.py` | 292 | `TransformerEncoderLayer` | SegFormer/MiT | LN — 不受影响 |
| MMSegmentation | `mmseg/models/backbones/vit.py` | 118 | `TransformerEncoderLayer` | ViT | LN — 不受影响 |
| OpenPCDet | `pcdet/models/backbones_image/swin.py` | 362 | `SwinBlock` | Swin | LN — 不受影响 |
| MMDetection | `mmdet/models/necks/hrfpn.py` | 86 | `HRFPN` | HRFPN | norm_cfg=None 默认无 norm |
| MMSegmentation | `mmseg/models/backbones/cgnet.py` | 47 | `GlobalContextExtractor` | CGNet SE 块 | 无归一化层 |

## 规律

受影响的全部是以 BatchNorm 为默认归一化的**卷积骨干**；不受影响的 Swin / MiT / ViT 使用 LayerNorm，
不维护滑动统计。触发条件与任务、数据集无关，只取决于检查点包裹范围内是否存在有状态归一化层。

注：扫描仅覆盖可静态解析的包裹模式，经由通用卷积封装间接引入归一化的情形未追溯，**20 处应视为保守下界**。

## 运行时验证（PyTorch 2.14 + torchvision 0.29，官方模型）

| 模型与设置 | BN 更新次数 | running_mean 与基线最大差 |
|---|---:|---:|
| ResNet BasicBlock，不使用检查点 | 1 | — |
| ResNet BasicBlock，use_reentrant=True | 2 | 6.92e-02 |
| ResNet BasicBlock，use_reentrant=False | 2 | 6.92e-02 |
| ResNet Bottleneck，不使用检查点 | 1 | — |
| ResNet Bottleneck，use_reentrant=True | 2 | 3.06e-02 |
| ResNet Bottleneck，use_reentrant=False | 2 | 3.06e-02 |
| DenseNet _DenseLayer，memory_efficient=False | 1 | — |
| DenseNet _DenseLayer，memory_efficient=True | 1~2 | 1.40e-02 |

**三点结论**：该现象在最新版 PyTorch/torchvision 上依然存在；`use_reentrant=False` 与 `True` 表现完全一致，
升级检查点实现无法规避；torchvision 官方 DenseNet 的 `memory_efficient` 开关本身即触发该现象。

## 自查判据

```python
before = {n: int(m.num_batches_tracked) for n, m in model.named_modules()
          if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}
# ... 跑一次完整的训练迭代（前向 + 反向）...
for n, m in model.named_modules():
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        d = int(m.num_batches_tracked) - before[n]
        if d > 1:
            print('受污染:', n, '每迭代更新', d, '次')
```

---

## 修复补丁状态（2026-09-18）

已按本文方案为两个仓库实现补丁并验证，patch 文件见
`mmdetection_bn_safe_checkpoint.patch` / `mmsegmentation_bn_safe_checkpoint.patch`。

| 仓库 | 新增 helper | 改动调用点 | lint | 端到端验证 |
|---|---|---:|---|---|
| mmdetection | `mmdet/models/layers/bn_safe_checkpoint.py` | 8 | isort/flake8 通过 | BasicBlock/Bottleneck 实测 BN 1 次、running_mean 差 0 |
| mmsegmentation | `mmseg/models/utils/bn_safe_checkpoint.py` | 9 | isort/flake8 通过 | 同上实现，共用同一 helper |

两个 patch 均已在全新克隆上通过 `git apply --check`。

**实现过程中发现的陷阱**：以 `torch.is_grad_enabled()` 判别重计算，在 `use_reentrant=False`
下两次前向都处于梯度使能状态，会把两次更新同时抑制，导致滑动统计**完全不更新**（比原缺陷更糟）。
正确做法是按调用次序判别首次前向。此坑已写入论文 4.5 节。

---

## 生态内已有缓解措施的覆盖情况（2026-09-18 复核）

| 实现 | 是否处理 BN | 判别条件 | 非重入路径下是否有效 |
|---|---|---|---|
| FairScale 0.4.13 `patch_batchnorm` | **是** | `torch.is_grad_enabled()` | **否**（见下表实测） |
| DeepSpeed 0.19.7 激活检查点 | 否 | — | — |
| PyTorch 2.14 `torch.utils.checkpoint` | 否 | — | — |
| MMDetection / MMSegmentation / BEVFusion | 否（直接调官方接口） | — | — |

PyTorch 官方 docstring 中关于重计算不等价的告警，指向的是"全局变量导致两次调用行为不同"，
**未提及持久缓冲被更新两次**这一副作用。

受控实测（单个 conv+BN 块，lr=0，固定种子，5 次训练迭代；
记 `num_batches_tracked` ／ `sum(|running_mean|)`）：

| 实现 | `use_reentrant=True` | `use_reentrant=False` |
|---|---|---|
| 不使用检查点（参照） | 5 ／ 0.03902 | — |
| 官方 `cp.checkpoint` | 10 ／ 0.05563 | 10 ／ 0.05563 |
| FairScale `patch_batchnorm` | 5 ／ 0.03902 | **10 ／ 0.05563** |
| 本文方案（调用次序判别） | 5 ／ 0.03902 | 5 ／ 0.03902 |

复现脚本：`fairscale_vs_ours.py`（torch 2.14.0，CPU 即可跑，约 3 秒）。

**对论文的影响**：抑制一次更新的思路并非首次提出，FairScale 早已实现。论文已相应改写
（§2.3 新增生态覆盖段、§3.4 补充适用范围比较、§4.5 新增表 8、结论段明确不主张首次发现权），
新颖性表述收缩为四点：定量刻画、真实训练污染测量、端到端影响评估、**已有缓解方案在非重入
路径下的正确性边界**。最后一点是此前任何报告与实现都没有的。
