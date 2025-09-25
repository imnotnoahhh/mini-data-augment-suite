# Runbook: 小数据图像分类 L20 单卡实验流程（速查）

> **使用方式**：先阅读 `docs/实验操作手册.md` 了解背景与细节，再按本 Runbook 执行命令。

## 0. 准备工作
1. 激活虚拟环境并安装依赖：
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. 确认数据目录：
   ```
   data/
     cifar-100-python/
     tiny-imagenet-200/
   ```
3. 若修改 `configs/*.yaml`，请记录变更以便复现。

## 1. 生成数据清单与 k-shot 抽样
```bash
python3 -m src.data --make-manifests --datasets configs/datasets.yaml
```
- 输出：`manifests/<dataset>/splits.yaml`、`kshot_*_seed*.csv`、索引 JSONL。
- 出错排查：检查数据路径、文件完整性与访问权限。

## 2. 基线健康检查（无增强）
示例（CIFAR-100, ResNet-50, K=20, seed=0）：
```bash
python3 -m src.train \
  --dataset cifar100 \
  --architecture resnet50 \
  --phase explore \
  --kshot 20 \
  --seed 0
```
- 输出目录：`logs/jsonl/<run_id>.jsonl`、`checkpoints/<run_id>/best_val.pt`、`last.pt`。
- 若性能异常，检查学习率放缩、显存占用、数据是否损坏。

## 3. 单算子步进实验（探索期）
对每个算子执行：
```bash
python3 -m src.search \
  --mode sweep \
  --dataset cifar100 \
  --architecture resnet50 \
  --phase explore \
  --kshot 20 \
  --seeds 0 1 2 \
  --op hflip
```
- 汇总输出：`reports/tables/single_op_sweep.csv`。
- 需要对 Tiny-ImageNet-200 同步执行一遍。

## 4. 贪心叠加搜索
```bash
python3 -m src.search \
  --mode greedy \
  --dataset cifar100 \
  --architecture resnet50 \
  --phase explore \
  --kshot 20 \
  --seed 0 \
  --topk 6 \
  --epsilon 0.4
```
- 输出：`reports/tables/greedy_path.csv`。
- 对 Tiny-ImageNet-200 亦需重复。

## 5. 确认期复现实验
对贪心阶段选出的组合（如 `combo_A`）在 10 个种子上训练：
```bash
for seed in 0 1 2 3 4 5 6 7 8 9; do
  python3 -m src.train \
    --dataset cifar100 \
    --architecture resnet50 \
    --phase confirm \
    --kshot 20 \
    --seed $seed \
    --combo-id combo_A
done
```
- 指标会自动写入 `reports/tables/confirm_10x.csv`。
- 对 Tiny-ImageNet-200 重复一次。

## 6. 统计汇总
```bash
python3 -m src.evaluate --task sweep --dataset cifar100 --architecture resnet50 --phase explore --kshot 20
python3 -m src.evaluate --task confirm --dataset cifar100 --architecture resnet50 --kshot 20
```
- 生成 `single_op_sweep_summary.csv` 与 `confirm_summary.csv`。
- 按需对 Tiny-ImageNet-200 再运行一遍，并汇总到 `reports/report_template.md`。

## 7. 归档与提交
1. 打包：`logs/`、`checkpoints/`、`reports/tables/`、`configs/`（若有改动）。
2. 在记录中注明：数据版本、`git_commit`、关键增强组合与统计结论。
3. 更新 `reports/report_template.md` 填写表格与图表。

## 8. 常见问题速查
| 症状 | 排查建议 |
| --- | --- |
| 训练损失 NaN | 调整学习率、批大小或梯度裁剪；确认数据未损坏 |
| DataLoader 卡住 | 调低 `configs/train.yaml` 的 `num_workers` 或关闭 `persistent_workers` |
| 显存不足 | 减小批大小，确保 ViT 梯度检查点开启 |
| 统计脚本缺记录 | 确认对应 CSV 已生成且 `dataset/model/phase/kshot` 匹配 |

完整背景说明请参阅 `docs/实验操作手册.md`。
