# 报告模板：小数据图像分类增强实验

## 1. 摘要
- 数据集、模型、增强搜索范围、主要发现一句话总结。

## 2. 实验设置
- 数据集切分与 K-shot 设定。
- 模型与微调策略。
- 训练协议（探索/确认）。

## 3. 单算子响应曲线
- 表 1：`reports/tables/single_op_sweep.csv`
- 图 1：Top-1 vs 强度曲线。

## 4. 贪心叠加路径
- 表 2：`reports/tables/greedy_path.csv`
- 文字分析：阈值、稳定性、失败案例。

## 5. 跨数据集与 K-shot 复核
- 表 3：`reports/tables/confirm_10x.csv`
- 图 2：不同 K 的性能趋势。

## 6. 统计检验
- 配对 t 检验/Wilcoxon 结果。
- Cohen's d 与 95% CI。

## 7. 讨论
- 甜蜜点的解释。
- 泛化能力与潜在局限。
- 未来工作。

## 附录
- Mixup/CutMix 对照。
- ColorJitter 二维切片。
- RandomResizedCrop ratio 分析。
