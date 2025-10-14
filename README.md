# 实验代码骨架概览

> 结构基于 `implementation_plan.md` 中的约定，用来确保实际代码与论文计划保持一致。后续填充逻辑时请保持同名同路径。

## 目录一览
- `configs/`：阶段与模型配置模板（计划中提到的 `configs/{stage}.yaml`、`configs/models.yaml`）。当前含占位文件，后续请为 `single_factor`, `sobol`, `rsm` 等阶段分别添加配置。
- `data/`：数据相关模块。
  - `splits.py`：训练/验证划分读取或生成逻辑。
  - `transforms.py`：单因子、Sobol、RSM 阶段的数据增强工厂。
- `engine/`：训练与评估核心。
  - `trainer.py`：封装训练循环（AMP、EMA、梯度裁剪等占位）。
  - `evaluator.py`：验证/测试评估入口。
- `orchestration/`：阶段调度脚本。
  - `run_stage.py`：统一 CLI，按阶段展开组合并写日志。
- `scripts/`：独立脚本集合。
  - `build_subset_split.py`：生成/校验 CIFAR-100 / STL-10 / Tiny-ImageNet-200 的子集划分（默认缓存到 `data/raw/`）。
  - `build_all_splits.sh`：批量调用上面脚本，可通过 `SEEDS="0 1 2"` 控制需要的 seed。
  - `verify_environment.py`：快速检查依赖、数据集、模型前向是否正常。
  - `transform_smoke_test.py`：验证 `data/transforms.py` 中的工厂函数可以正常执行。
  - `train_smoke_test.py`：跑极简训练循环（默认 2 step）确认优化器/反向传播正常。
  - `generate_previews.py`：根据配置文件生成「原图 vs. 增广后」对比图。
- `analysis/`：实验结果分析工具。
  - `gather_metrics_template.ipynb`：CI/图表分析的 Notebook 模板占位。
- `data/raw/`：存放下载的数据集原始文件（脚本自动创建）。
- `artifacts/splits/`：按数据集分类存放划分结果（例如 `artifacts/splits/cifar100/`）。
- `outputs/`：阶段输出目录（`metrics.csv`、`summary.json`、`transform.json`、`best.ckpt`）。

## 约定
1. **文档同步**：如需新增模块，请先更新 `implementation_plan.md` 并在此 README 反映路径，避免计划与实现偏离。
2. **占位说明**：当前所有 `.py` 文件仅提供接口注释，真实逻辑需要在后续步骤实现。
3. **日志结构**：在填充逻辑时，务必遵循计划中关于 CSV/JSON 字段的定义，例如 `scale_delta` 与 `(scale_low, scale_high)` 的存储方式。
- `utils/preview.py`：封装预览图生成逻辑，供阶段脚本调用。
- `artifacts/preview_configs/`：示例配置，描述不同阶段的增广组合用于预览。

## 预览图生成

要求：
- CIFAR-100 的每个阶段（单因子 / Sobol / RSM / Final baseline）在正式训练前，为每个增广配置生成 8 张「原图 vs. 增广」对比图。
- STL-10 与 Tiny-ImageNet-200 在迁移评估阶段，各抽 2 组样本输出最终组合增广的对比图。

使用方式示例：

```bash
# 为 CIFAR-100 单因子阶段输出 8 张预览（示例配置）
python scripts/generate_previews.py \
  --dataset cifar100 \
  --stage single_factor \
  --config-file artifacts/preview_configs/cifar100_single_factor.json \
  --split-seed 0 \
  --target-size 96 \
  --save-individual

# 为 STL-10 的最终组合生成对比（默认 2 张）
python scripts/generate_previews.py \
  --dataset stl10 \
  --stage final \
  --config-file artifacts/preview_configs/stl10_final.json \
  --split-seed 0 \
  --target-size 96 \
  --save-individual
```

配置文件格式（JSON 列表）：

```json
[
  {"label": "brightness_0.2", "type": "single_factor", "operation": "brightness", "strength": 0.2},
  {"label": "sobol_cfg_01", "type": "sobol", "params": {"brightness": 0.2, "contrast": 0.1, ...}}
]
```

`type` 支持 `single_factor`、`sobol`、`rsm`、`identity`；生成的图片默认保存在 `artifacts/previews/{dataset}/{stage}/`。仓库预置的配置如下：

- `artifacts/preview_configs/cifar100_single_factor.json`
- `artifacts/preview_configs/cifar100_sobol.json`
- `artifacts/preview_configs/cifar100_rsm.json`
- `artifacts/preview_configs/cifar100_final.json`
- `artifacts/preview_configs/stl10_final.json`
- `artifacts/preview_configs/tiny-imagenet-200_final.json`

CIFAR-100 默认每个配置输出 8 张，其它数据集默认 2 张，可通过 `--num-images` 覆盖。`--target-size` 默认 96，可按需调整（内部使用最近邻插值放大）。若指定 `--save-individual`，脚本会为每个样本分别保存原图与增广图（文件名格式为 `{label}_sampleXX_base.png` 与 `{label}_sampleXX_aug.png`），并存放在 `artifacts/previews/{dataset}/{stage}/{config_label}/` 子目录下；若不加该参数则继续输出整张上下拼接的对比图到 `artifacts/previews/{dataset}/{stage}/`。

## 运行单因子训练

单因子阶段的训练入口位于 `orchestration/run_stage.py`，读取 `configs/single_factor.yaml` 后依次遍历所有单因子组合（默认来自 `artifacts/preview_configs/cifar100_single_factor.json`），对每个强度×种子的组合启动一次完整训练。示例命令：

```bash
python orchestration/run_stage.py --stage single_factor
```

默认行为：
- 使用配置中设定的 `num_workers=8`（适配 A10 + 8 核 CPU）；
- 自动启用 AMP、EMA、梯度裁剪等选项，并记录 `metrics.csv`、`summary.json`、`transform.json`、`config.yaml`、`best.ckpt`；
- 输出目录为 `outputs/single_factor/{config_label}_seed{seed}/`，命名规则与预览配置一致；
- 建议在训练前先运行 `generate_previews.py` 确认增广效果，确保阶段实验符合预期。

## 本地自检 & 运行顺序

在迁移到服务器之前，建议按以下步骤在本地验证环境与配置：

```bash
# 1. 生成所有数据划分（默认种子 0/1/2）
SEEDS="0 1 2" ./scripts/build_all_splits.sh

# 2. 生成单因子阶段的增广预览（96×96，单张输出）
python scripts/generate_previews.py \
  --dataset cifar100 \
  --stage single_factor \
  --config-file artifacts/preview_configs/cifar100_single_factor.json \
  --split-seed 0 \
  --target-size 96 \
  --save-individual

# 3. 增强流水线 smoke test（确保单因子/Sobol/RSM 工厂可执行）
python scripts/transform_smoke_test.py --dataset cifar100

# 4. 训练最小循环 smoke test（验证 DataLoader/反向传播）
python scripts/train_smoke_test.py --dataset cifar100 --model resnet18 --steps 2

# 5. 依赖与模型前向检查
python scripts/verify_environment.py

# 6. 正式启动单因子阶段训练
python orchestration/run_stage.py --stage single_factor
```

迁移到 GPU 服务器后，再重复步骤 3–5，确认 CUDA 环境一致。所有命令均已在脚本中默认配置为 A10 + 8 核 CPU + 24GB 显存，可根据需要进一步调整。
