# CONTRACT: 小数据图像分类代码接口

## 1. 目录约定
```
project/
  configs/
  manifests/
  reports/
  src/
    __init__.py
    data.py
    augment.py
    models.py
    train.py
    evaluate.py
    search.py
    utils/
      __init__.py
      logging.py
```

## 2. CLI 入口
- 生成清单：`python -m src.data --make-manifests --datasets configs/datasets.yaml`
- 单算子网格：`python -m src.search --mode sweep --op <op>`
- 贪心叠加：`python -m src.search --mode greedy`
- 训练：`python -m src.train --phase <explore|confirm>`
- 评估统计：`python -m src.evaluate`

## 3. 数据接口
- `src.data.get_dataloaders(config, stage, combo)` 返回 train/val/test DataLoader 字典。
- `src.data.build_manifest(args)` 生成 `manifests/<dataset>/*`。
- Dataset 样本格式：`{"image": Tensor[C,H,W], "target": int, "index": dict}`。

## 4. 增强接口
- `src.augment.build_pipeline(cfg, combo=None)` 返回 torchvision transform。
- 顺序：Resize → Flip → Affine → Color → Blur/Gray → Erasing → ToTensor/Normalize。
- 支持单算子调度与组合调度。

## 5. 模型接口
- `src.models.create_model(arch_cfg, num_classes, compile=True)`。
- 支持 ResNet-18/50、ConvNeXt-Tiny、ViT-T/16。
- 返回 `(model, optimizer, scheduler, scaler)`。

## 6. 训练器
- `src.train.train_phase(phase, configs, combo)` 执行全流程训练并记录 JSONL。
- 训练循环需写入 `reports/tables/*.csv` 与 `checkpoints/`。

## 7. 搜索器
- `src.search.run_sweep(args)` 单算子扫描，写入 `reports/tables/single_op_sweep.csv`。
- `src.search.run_greedy(args)` 贪心叠加，写入 `reports/tables/greedy_path.csv`。

## 8. 日志
- 每次运行调用 `src.utils.logging.create_run_logger()`。
- JSONL 字段：`run_id, timestamp, config_hash, commit, seed, metrics`。

## 9. 兼容性
- Python >= 3.10, PyTorch >= 2.2。
- 支持单 GPU (L20 48GB)；DataLoader workers=10。

本契约与 `docs/Runbook.md` 联动，确保自动化脚本与手动复现一致。
