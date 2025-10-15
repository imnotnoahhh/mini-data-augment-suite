1. **学习率与阶段重置**  
   - 单因子 → Sobol → RSM → 定稿各阶段 **分别重置** optimizer 与 LR scheduler（保持同一 base LR 规则：`0.1@BS=128` 线性缩放到 `0.2@BS=256`）。  
   - **定稿训练从随机初始化开始**，不沿用探索阶段权重，保证公平可比。

2. **验证切分与噪声控制**  
   - 在 CIFAR‑100@20% 训练子集内再划 **10% 作验证集**（分层抽样、固定 index），test 仅用于最终报告。  
   - 每项实验 **3 seeds × 固定 split**，报告 **mean ± std / 95% CI**；CI 同样用于早停判断。

3. **“同起点”公平性**  
   - 任一对比策略（你的组合、RA、TrivialAugment）都 **从相同初始化**、**相同数据划分**独立训练；不复用对方或先前阶段的权重。
   - Baseline 统一配置：模型、优化器、调度器、EMA、训练轮数与主实验完全一致（ResNet-18 / SGD(momentum=0.9, weight_decay=5e-4) / Cosine LR / 200 epochs / EMA decay=0.999），仅允许在增强策略的搜索预算上做对等或更宽松设置。

4. **EMA / 梯度裁剪 开关**  
   - **EMA 默认开**（decay=0.999），验证/提交使用 EMA 权重。  
   - **梯度裁剪默认关**；仅在不稳定时启用 `clip_grad_norm_`，**max_norm=5.0**（AMP 下先 `scaler.unscale_(optimizer)` 再裁剪）。

5. **变换顺序与接口一致性**  
   - 推荐顺序（v1 API）：`RandomCrop → HorizontalFlip → Rotation → RandomAffine(scale) → ColorJitter → ToTensor → RandomErasing → Normalize`。  
   - 注意 `RandomErasing` 作用于 **Tensor**，`Normalize` 放在其后；若使用 `torchvision.v2`，确保语义等价。

6. **Sobol 阶段省算力策略**  
   - 先用 **1 seed × 64 组**筛选 **Top 10–15%** 组合，再仅对 **Top‑k 组合补 3 seeds**；其余组合不补测。

7. **RSM 关键因子选择与设计**  
   - 从 Sobol 数据用 **逐步回归（AIC/BIC）/岭回归/Permutation importance** 选 **K=2–4** 个关键因子。  
   - 采用 **CCD**；**中心点重复 ≥3 次** 估计纯误差/LOF；最优点出炉后做 **±微扰** 稳健性复测（3 seeds）。

8. **报告与公平性说明**  
   - 主文明确：**20% 档学一次策略，10%/40% 零调参迁移**（learn‑once, apply‑many）。  
   - 对比方法给 **等预算**（或更宽松）搜索/超参；在文中列 **总 GPU 小时** 与关键超参。

9. **日志与可复现**  
   - 记录并导出：抽样 index、全部 transform 参数、随机种子、学习率曲线、每次实验的指标 CSV。  
   - 推荐：
     ```python
     import torch, torch.backends.cudnn as cudnn
     cudnn.benchmark = True                 # 吞吐优先
     torch.use_deterministic_algorithms(False)
     ```
     在附录说明非完全确定性来源，并用 3‑seed 平滑处理。

10. **A10 实用贴士**  
   - 开启 AMP，并设置：
      ```python
      torch.set_float32_matmul_precision("high")
      ```
   - 显存吃紧时先把 **batch 256→128**（LR 同步减半），再考虑梯度累积；避免同时改动过多超参。

## 🌐 数据集 × 模型实验矩阵
- **数据集覆盖**（均采用「20% 训练子集 × 10% 验证」分层划分原则）：  
  1. **CIFAR-100**（32×32，自然图像，100 类）——主线实验；  
  2. **STL-10**（96×96，10 类，训练标注少 + 10 万 unlabeled）——验证高分辨率/少标注场景的泛化；  
  3. **Tiny-ImageNet-200**（64×64，200 类）——验证更细粒度、多类别场景。
- **模型覆盖**（统一使用 torchvision 实现，保持优化器与训练设置一致）：  
  - ResNet-18（主线全流程：单因子 → Sobol → RSM）；  
  - ResNet-50；  
  - ConvNeXt-Tiny（ConvNeXt 家族的轻量版本，补充最新 CNN 架构）。  
- **运行策略**：  
  - **主线**：仅在「CIFAR-100 × ResNet-18」上执行完整三阶段 DOE 与消融；  
- **泛化确认**：将主线最终确定的增广组合迁移到其余 4 组（两数据集 × 两个额外模型，分别对 STL-10/Tiny-ImageNet-200 × ResNet-50/ConvNeXt-Tiny），仅执行定稿训练（无需重复搜索），验证性能增益与稳定性；  
  - 若迁移结果出现异常，再增量开展局部微调（需在补充材料中说明）。

## 🧱 实现架构草案
- **配置管理**：使用 `configs/{stage}.yaml` 描述数据源、模型、优化器、增强范围、训练时长，并在 `configs/models.yaml` 注册可用 backbone（ResNet-18/50、ConvNeXt-Tiny）；CLI 入口仅接收 `stage` 与 `seed`（以及 dataset/model 覆盖），其余从配置派生，避免人工偏差。
- **数据管线**：`data/splits.py` 负责生成/读取 split（见下节），`data/transforms.py` 暴露 `make_single_factor`, `make_sobol_combo`, `make_rsm_combo` 工厂，确保增强参数与 CSV 字段一一对应。
- **训练循环**：`engine/trainer.py` 封装 epoch 级逻辑（AMP、EMA、梯度裁剪、指标聚合），`engine/evaluator.py` 专管验证/测试，保持指标计算一致。
- **实验调度**：`orchestration/run_stage.py` 读取阶段配置，展开组合（单因子档位 / Sobol CSV / RSM 设计矩阵），逐一调用 trainer 并写入日志；失败 run 自动重试一次并标记。
- **结果存档**：所有输出位于 `outputs/{stage}/{run_id}/`，包含 `metrics.csv`、`summary.json`、`config.yaml`、`transform.json`、`best.ckpt`；`run_id = stage_seed_comboId`，与论文表格索引一致。

### 数据集拆分脚本
- `scripts/build_subset_split.py --dataset {cifar100, stl10, tiny-imagenet-200} --train-ratio 0.2 --val-ratio 0.1 --seed {0,1,2}`：
  1. 加载目标数据集的有标注训练集（默认缓存于项目目录 `data/raw/`），按类分层抽样 **20%** 作为 `train_subset`；  
  2. 在 `train_subset` 内分层抽取 **10%** 作为验证集（若类别样本不足，至少保留 1 张验证图像）；  
  3. 将 `{ "train_indices": [...], "val_indices": [...] }` 写入 `artifacts/splits/{dataset}/{dataset}_seed{seed}.json`；  
  4. 生成 `artifacts/splits/{dataset}/metadata.json`（记录 torchvision 版本、生成时间、脚本 git hash 以及每个 seed 的哈希/样本量）；  
  5. 对 `seed ∈ {0,1,2}` 循环执行，保证主线与对照实验共享同一划分。
- 主训练入口以 `load_split(dataset, seed)` 强制读取；若文件缺失或哈希不匹配直接报错，禁止自动重采样。
- 若需批量生成划分，可运行 `scripts/build_all_splits.sh`（可通过环境变量 `SEEDS="0 1 2"` 指定需要的 seed，`DATA_ROOT`/`TINY_ROOT` 自定义缓存位置）。

### 增强预览输出
- 预览生成工具：`utils/preview.py` 提供 `save_transform_preview`；命令行脚本 `scripts/generate_previews.py` 读取 JSON 配置批量输出对比图。
- CIFAR-100：在每个阶段（单因子 / Sobol / RSM / Final baseline）正式训练前，遍历当期全部增广配置，调用预览函数输出不少于 **8 张** 对比。默认将样本缩放到 **96×96**（最近邻插值），若未开启 `--save-individual`，拼接图存放于 `artifacts/previews/cifar100/{stage}/`；若开启，则每个配置写入 `artifacts/previews/cifar100/{stage}/{config_label}/`，并生成 `{label}_sampleXX_base/_aug.png` 文件，方便后续自定义拼图。
- STL-10、Tiny-ImageNet-200：在迁移评估阶段，为最终确定的组合增广各输出 **2 张** 对比图（同样缩放到 96×96，目录结构与上相同，支持 `--save-individual`）。
- 推荐流程：阶段脚本在读取配置后先执行 `generate_previews.py`（或直接调用 `save_transform_preview`），确认增广效果正确再进入训练循环。

### 本地自检顺序
- `SEEDS="0 1 2" ./scripts/build_all_splits.sh`
- `python scripts/generate_previews.py --dataset cifar100 --stage single_factor --config-file artifacts/preview_configs/cifar100_single_factor.json --split-seed 0 --target-size 96 --save-individual`
- `python scripts/transform_smoke_test.py --dataset cifar100`
- `python scripts/train_smoke_test.py --dataset cifar100 --model resnet18 --steps 2`
- `python scripts/verify_environment.py`
- `python orchestration/run_stage.py --stage single_factor`

上述命令在本地通过后，再在服务器上重复 smoke test（步骤 3–5），确认 CUDA 环境一致后启动正式训练。

### 日志与指标落盘
- 指标表遵循「单因子实验输出结构规范」，所有阶段 append 写入 `outputs/{stage}/{run_id}/metrics.csv`；行包含 `stage`、`combo_id`、`seed`、`epoch` 等附加列，方便聚合。
- `summary.json` 存储关键摘要：`best_val_acc`、`best_epoch`、`stop_reason`、`ema_enabled`、`lr_schedule_snapshot`。单因子阶段需额外汇总成「run 级摘要表」（按 `combo_id × seed` 聚合关键字段）。
- `events.log` 记录早停、NaN、梯度爆炸、重试等事件，配合 `stderr` 追踪训练失败原因；在汇总报表中需要列出是否触发早停、对应原因，并与 `events.log` 交叉校验。
- `analysis/gather_metrics.ipynb` 汇总 CSV，计算 mean/std/CI 并生成论文图表；Notebook 在仓库中给出模板以保证流程可复现。该 Notebook 现要求输出下列「必选」可视化与统计：
  1. **操作 × 强度统计表**：字段包含 `Top-1 / Top-5 / Macro-F1 / Loss / mean / std / 95% CI / EarlyStop / Note`，确保 3 seeds 的 `mean±std±95%CI` 均计算完成。
  2. **每个操作的强度曲线**：Top-1、Macro-F1、Loss 至少三条折线图，均需显示 95% CI（误差棒或阴影带）。
  3. **最佳强度对比图**：按操作选取性能最优的强度档，绘制 Top-1（附带 CI）的柱状或折线图；可选在同图补充 Macro-F1。
  4. **训练过程曲线**：基于 `metrics.csv` 绘制 Loss 与 Accuracy（Top-1/Top-5）随 epoch 变化的曲线。
  5. **混淆矩阵**：对每个 run 输出验证集混淆矩阵可视化，并在摘要表中记录宏平均 F1（与 `summary.json` 对齐）。
  6. **学习率曲线**：根据 `lr_schedule_snapshot` 绘制 Cosine 退火曲线，用于核对调度器行为。
- 所有生成的图表与表格应存入 `artifacts/reports/single_factor/`（或配置指定的位置），命名中包含操作名/强度，确保后续阶段可复用。
- `scripts/transform_smoke_test.py`、`scripts/train_smoke_test.py`、`scripts/verify_environment.py` 分别用于本地变换/训练/依赖自检，出厂前需全部通过。

### 公平性约束的落地
- `set_seed(seed)` 同时作用于 Python、NumPy、PyTorch（含 CUDA），写入 `summary.json` 的 `rng_state_hash`。
- `run_stage.py` 在每次训练前验证：split 文件 hash、模型初始化权重 hash 与对照组一致，若不一致终止并标记失败。
- 所有基线与自定策略共享 `ExperimentConfig.signature`（聚合模型、优化器、调度器参数），增强策略是唯一可变项。

### 对照方法配置
| **方法** | **增强策略** | **搜索/试验预算** | **其余配置** |
|:-:|:-:|:-:|:-:|
| 自定组合 | 单因子 → Sobol 64 组 → RSM CCD | \(64 + \text{CCD 样本}\) 组合 × 3 seeds | 共享主配置 |
| RandAugment | `N=2`, `M∈[5,15]`（7 档，与单因子强度匹配） | 与单因子阶段等档位，共 7 档 × 3 seeds | 学习率/优化器一致 |
| TrivialAugment | `max_level∈{5,7,9,11,13,15,17}` | 7 档 × 3 seeds | 共享主配置 |
| NoAug 基线 | 仅标准化 + RandomCrop + Flip | 1 组 × 3 seeds | 共享主配置 |
> 以上对照均使用相同 split、初始化与训练轮数；若额外增加实验，需在报告中注明新增 GPU 小时。

# On the Effectiveness of Simple Data Augmentation Combinations for Low-Data Image Classification
### 实验逻辑：单因子筛选 → Sobol 全局均匀采样 → RSM 局部优化

---

## 单因子筛选逻辑

| **Geometric** | Random Crop, Horizontal Flip, Rotation, Scaling |
| **Color-based** | Brightness, Contrast， Saturation, Hue |
| **Occlusion** | Random Erasing |

---

## 📘 Geometric 类（几何变换）

| **操作** | **参数** | **合法区间（官方语义）** | **推荐档位 (共 7 档，低→高)** | **说明** |
|:-:|:-:|:-:|:-:|:-:|
| **RandomCrop** | padding | 任意 int ≥ 0（补边后裁剪） | {0, 2, 4, 6, 8, 10, 12} | 保持输出尺寸为 32×32，仅调节补边大小；过大会引入模糊边缘。 |
| **RandomHorizontalFlip** | p | [0, 1] | {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0} | 概率越大，水平翻转越频繁。 |
| **RandomRotation** | degrees | 任意 float，采样自 [−d, +d] | {0, 5, 10, 15, 20, 25, 30} | 角度过大会切掉主体边缘；30° 对 CIFAR-100 已较强。 |
| **Scaling (RandomAffine.scale)** | δ，最终 scale=(1−δ,1+δ) | δ ∈ [0,0.35] | {0.00,0.05,0.10,0.15,0.20,0.25,0.30} | 通过 δ 控制对称缩放幅度；0 表示不缩放，δ≥0.3 可能导致主体截断。 |

---

## 🎨 Color-based 类（颜色变换）

| **操作** | **参数** | **合法区间（官方语义）** | **推荐档位 (7 档)** | **说明** |
|:-:|:-:|:-:|:-:|:-:|
| **Brightness** | b ≥ 0 → 采样范围 [max(0, 1−b), 1+b] | b ≥ 0 | {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6} | 调整整体亮度；0.6 已接近人眼可感知极限。 |
| **Contrast** | c ≥ 0 → 采样范围 [max(0, 1−c), 1+c] | c ≥ 0 | 同上 | 调整对比度；≥0.5 时类内差异明显增强。 |
| **Saturation** | s ≥ 0 → 采样范围 [max(0, 1−s), 1+s] | s ≥ 0 | 同上 | 控制色彩饱和度；0.5 已非常鲜艳。 |
| **Hue** | h ∈ [0, 0.5] → [−h, +h] | 0 ≤ h ≤ 0.5 | {0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30} | 超过 0.3 会出现明显色偏。 |

---

## 🩶 Occlusion 类（遮挡）

| **操作** | **参数** | **合法区间（官方语义）** | **推荐档位 (7 档)** | **说明** |
|:-:|:-:|:-:|:-:|:-:|
| **RandomErasing** | scale=(min,max)∈ (0,1]；ratio>0 | scale 表示擦除面积比例 | scale_max ∈ {0.04, 0.08, 0.12, 0.18, 0.24, 0.30, 0.36} ；设 `scale=(0.02, scale_max)`，`ratio=(0.3, 3.3)`，`p=0.5` | 0.36≈1/3 图像面积，再高易破坏主体结构。 |

---

## 🧮 单因子实验早停逻辑

### 🎯 目标
在**全合法区间内确定性能峰值**（Top-1 / F1 / Loss），避免无意义强度的重复实验。

### 🚦 阶段一：极端哨兵点筛选
1. 选取每个操作的 **最弱（min）、中间（mid）、最强（max）** 三个档位；  
2. 在固定训练配置下（3 seeds）评估三点；  
3. 若 `upper(CI_max) < lower(CI_mid)`，则判定高强端性能退化，**仅保留 min–mid 区间进入阶段二**；否则保留全区间。

### ⏩ 阶段二：顺序细扫 + 置信区间早停
从弱到强逐档测试，每档跑 3 seeds 并计算平均性能及 95% 置信区间：  
\[
CI_t = \overline{x_t} \pm 1.96 \cdot \frac{s_t}{\sqrt{3}}
\]

触发早停条件之一时终止该操作实验：  
\[
\text{(1) } \overline{ACC}_t < \overline{ACC}_{t-1} \ \text{且} \ \overline{ACC}_{t-1} < \overline{ACC}_{t-2}
\]  
\[
\text{(2) } \mathrm{upper}(CI_t) < \mathrm{upper}(CI_{t-1})
\]

> 建议至少保留 **3–4 档数据**，以便后续 RSM / Sobol 建模。

### ⚙️ 实现要点
- 每个强度档位运行 **3 个随机种子**（seed=0,1,2）；  
- 每次实验输出 **Top-1、Top-5、macro-F1、val-loss** 四项指标；  
- 计算 **均值、标准差与置信区间**；  
- 当触发早停时记录 **触发档位及原因**。

---

## 📊 单因子实验输出结构规范

| **Operation** | **Strength** | **Top-1 Acc** | **Top-5 Acc** | **Macro-F1** | **Loss** | **Mean (Top-1)** | **Std (Top-1)** | **CI_low** | **CI_high** | **EarlyStop** | **Note** |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Brightness | 0.00 | 71.2 | 93.5 | 70.4 | 1.42 | 71.2 | 0.5 | 70.6 | 71.8 | False | Sentinel |
| Brightness | 0.10 | 73.8 | 94.1 | 72.7 | 1.39 | 73.8 | 0.4 | 73.3 | 74.3 | False |  |
| Brightness | 0.20 | 74.1 | 94.3 | 73.0 | 1.37 | 74.1 | 0.6 | 73.3 | 74.9 | ✅ True | CI下降 |

**字段定义与说明**

| **字段** | **含义说明** | **备注 / 用途** |
|:-:|:-:|:-:|
| **Operation** | 数据增强操作名称（如 Brightness / Rotation） | 每个操作单独生成一张表 |
| **Strength** | 当前档位参数值 | 对应合法区间中的 7 档之一；Scaling 使用 δ（scale=(1−δ,1+δ)) |
| **Top-1 Acc / Top-5 Acc / Macro-F1 / Loss** | 三种精度指标与验证集平均损失 | 逐 seed 取值与聚合结果均需保留 |
| **Mean / Std** | 3 seeds 的均值、标准差 | 对所有四个指标计算；Top-1 必须在表格主列呈现 |
| **CI_low / CI_high** | 95% 置信区间上下界 (\(\overline{x} \pm 1.96 \cdot s / \sqrt{3}\)) | 同时给出四个指标的 CI，表格主列至少展示 Top-1 的区间，附录可放其余指标 |
| **EarlyStop** | 是否触发早停 (True/False) | CI 或性能趋势下降时置 True |
| **Note** | 特殊标记（如 Sentinel / CI下降 / Truncated / ConfMat） | 记录早停原因、混淆矩阵摘要、异常事件等补充信息 |

---

## 🔷 Sobol 全局均匀采样（多因子阶段）

**一句话总结**：Sobol 采样是一种低差异（low-discrepancy）序列，能在多维空间产生更均匀的覆盖，比完全随机更稳定，适合 9–12 维的增强组合空间。

### 一、准备输入空间
你的 Sobol 空间维度 = 因子数（例如 9 个增强操作）。  
每个维度的区间由单因子实验结果确定（早停后保留的有效区间）；其中 `scale_delta` 表示相对于 1.0 的对称缩放幅度，后续构造 transform 时会转换为 `(1-δ, 1+δ)`：

```python
param_ranges = {
    "brightness": (0.0, 0.4),
    "contrast": (0.0, 0.5),
    "saturation": (0.0, 0.5),
    "hue": (0.0, 0.15),
    "crop": (0, 6),
    "flip": (0.1, 0.9),
    "rotation": (0, 20),
    "scale_delta": (0.0, 0.3),
    "erase": (0.02, 0.24),
}
```

### 二、生成 Sobol 采样点
建议使用 `scipy.stats.qmc.Sobol`（支持直接缩放到自定义范围）；`torch.quasirandom.SobolEngine` 亦可。

```python
from scipy.stats import qmc
import pandas as pd

# 1️⃣ 初始化 Sobol 采样器
sampler = qmc.Sobol(d=len(param_ranges), scramble=True)

# 2️⃣ 生成样本点，比如 64 个组合
sample = sampler.random_base2(m=6)  # 2^6 = 64
sample_scaled = qmc.scale(sample,
                          [v[0] for v in param_ranges.values()],
                          [v[1] for v in param_ranges.values()])

# 3️⃣ 转成表格方便查看
df = pd.DataFrame(sample_scaled, columns=param_ranges.keys())
df.head()
```

> 说明：Sobol 序列生成的是 [0,1] 区间的点；`qmc.scale` 将其线性映射到各增强操作的取值范围内。

### 三、实验执行方式
每一行代表一个「增强组合」，训练时逐行读取构造组合变换：

```python
from torchvision import transforms as T

CROP_CHOICES = [0, 2, 4, 6, 8, 10, 12]

def make_transform(row):
    crop_idx = int(round(row["crop"]))
    crop_idx = max(0, min(crop_idx, len(CROP_CHOICES) - 1))
    crop_padding = CROP_CHOICES[crop_idx]
    return T.Compose([
        T.RandomCrop(32, padding=crop_padding),
        T.RandomHorizontalFlip(p=float(row["flip"])),
        T.RandomRotation(degrees=float(row["rotation"])),
        T.RandomAffine(
            degrees=0,
            scale=(max(0.5, 1 - row["scale_delta"]), 1 + row["scale_delta"])
        ),
        T.ColorJitter(
            brightness=row["brightness"],
            contrast=row["contrast"],
            saturation=row["saturation"],
            hue=row["hue"]
        ),
        T.ToTensor(),
        T.RandomErasing(
            p=0.5,
            scale=(0.02, row["erase"]),
            ratio=(0.3, 3.3)
        )
    ])
```

执行时读取：
```python
for i, row in df.iterrows():
    transform = make_transform(row)
    # 调用训练函数
    acc, loss = train_with_transform(transform)
```

> 注：`scale_delta` 先映射到 `(1-δ, 1+δ)` 再传给 `RandomAffine`，并额外夹紧下界以规避极端采样导致的非正缩放。
>
> Sobol 采样得到的 `crop` 为连续值，需如上所示先映射到离散合法档位（{0,2,4,6,8,10,12}）；其他离散字段可按同样方式 snap 到允许列表，保证实验记录与实现一致。

### 四、Sobol 相比随机采样的优势

| **方法** | **分布特性** | **方差** | **适用场景** |
|:-:|:-:|:-:|:-:|
| **随机采样** | 完全随机，可能局部密集 | 高 | 适合快速探索 |
| **LHS（拉丁超立方）** | 各维度均匀，但不保证全局均匀 | 中等 | 参数个数少时效果好 |
| **Sobol（低差异序列）** | 全局均匀、覆盖性最强 | 最低 | 推荐用于高维组合空间 |

> Sobol 非常适合你的情况（9~12 个增强因子），既能覆盖所有组合方向，又不需要指数级试验量。

### 五、实验记录建议结构

| **ID** | **Brightness** | **Contrast** | **Crop** | **Flip** | **Rotation** | **Scale δ** | **Scale (low,high)** | **Erase** | **Top-1** | **Top-5** | **F1** | **Loss** |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 | 0.21 | 0.34 | 4 | 0.62 | 10 | 0.08 | (0.92, 1.08) | 0.16 | 77.3 | 93.8 | 74.9 | 1.22 |
| 1 | 0.38 | 0.29 | 2 | 0.41 | 15 | 0.12 | (0.88, 1.12) | 0.20 | 78.1 | 94.2 | 75.6 | 1.19 |

> CSV 中保存 `scale_delta`，而 `transform.json` 存 `(scale_low, scale_high)`，实现与日志可互相还原。

---

## 🟣 RSM（响应面法）局部优化

**目的**：在 Sobol 得到的“表现较好”的区域内，构建**二次响应面**，寻找近似最优组合，并定量解释各因子的主效应与交互效应。

### 1) 候选子空间选择
- 在 Sobol 结果中选取 **Top 10%** 的组合；
- 统计每个因子在这些高分样本中的分布，确定 **关键因子 K（2–4 个）**；
- 将这些关键因子的取值**标准化到 [-1, 1]**，作为 RSM 自变量。

### 2) 二次模型形式（带交互）
\[
y = \beta_0 + \sum_{i=1}^{K}\beta_i x_i + \sum_{i=1}^{K}\beta_{ii} x_i^2 + \sum_{i<j}\beta_{ij} x_i x_j + \epsilon
\]
- 其中 \(x_i\) 为标准化因子（-1, 0, +1 等编码）；  
- \(y\) 为目标（如 Top-1 或 macro-F1）。

### 3) 设计点（建议）
- **中心点**：重复 \(n_c=3\) 次用于估计噪声与纯误差；  
- **轴点（star points）**：每个因子在 \(\pm \alpha\)，\(\alpha ≈ 1\)（若需旋转中心复原，可取 \(\alpha = \sqrt{K}\)）；  
- **边点（factorial points）**：在 \((-1, +1)\) 的全因子或半复合设计（CCD）。

**样本量建议**（仅对关键因子 K）：  
- K=2：中心 3 + 轴点 4 + 边点 4 → **11–15** 次；  
- K=3：中心 3 + 轴点 6 + 边点 8 → **17–23** 次；  
- K=4：中心 3 + 轴点 8 + 边点 16 → **27–35** 次。

### 3.5) 工具链与数据流
- Sobol 阶段输出保存为 `sobol_results.csv`（每行包含组合参数、指标、seed）。  
- `analysis/build_rsm_design.py`：读取 Sobol CSV，选择 Top 10% & 关键因子，调用 `statsmodels.api.OLS` 拟合二次模型，输出 `design_matrix.csv` 与 `coefficients.json`。  
- 设计矩阵列格式：`factor`、`coding`（-1/0/+1）、`delta`（原始尺度偏移）、`replicate_id`。  
- `orchestration/run_stage.py --stage rsm` 读取设计矩阵，恢复真实增强参数（含 `scale_delta→(1±δ)`），调用训练入口并写 `metrics.csv`。  
- `analysis/rsm_report.ipynb` 使用 `statsmodels.stats.anova.anova_lm` 生成方差分析表，绘制响应面与等高线，支撑论文解释性。

### 4) 拟合与寻优
- 用最小二乘拟合 \(\beta\)；
- 若二次项正定，最优点 \(\hat{x}^* = -\frac{1}{2} H^{-1} g\)（\(H\) 为二次项矩阵，\(g\) 为一阶系数向量）；
- 将 \(\hat{x}^*\) 反标准化回原始强度空间，得到**候选最优组合**。

### 5) 验证与稳健性
- 在验证集上复现 3 seeds（或更多）；
- 对最优点的 \(\pm\) 微扰做敏感性测试，输出**可行区间**与**置信带**；
- 可补做一个**消融**（去掉某一因子或交互项）。

---

## ✅ 实施建议（全流程）

| **步骤** | **内容** | **样本数** | **是否自动** |
|:-:|:-:|:-:|:-:|
| 单因子 | 确定每个增强的有效区间 | 每因子约 7 档 × 3 seeds | 半自动（早停） |
| Sobol | 全局均匀采样组合 | 32–128 组（2^m） | 全自动 |
| RSM | 局部拟合与微调 | 取 Sobol Top 10% | 可选（自动/手动） |

---
