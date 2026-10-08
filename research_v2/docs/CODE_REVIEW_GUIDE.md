# 核心代码与审阅路线

入口为 [详细研究报告](../REVIEW_REPORT_ZH.md)。本文件说明如何定位实现和重现证据，不要求审阅者接受报告结论。

## 1. 最短阅读路径

1. `registry.py`：44 个固定规格以及两项 Blend 的精确定义。
2. `scenario_inference.py`：场景数据结构、后验联合质量、外层边际、条件质量、EB 和信息几何。
3. `risk_envelopes.py`：各种目标、不利权重与支持函数；特别检查条件组质量是否固定。
4. `risk_center.py`：加权 QP、上下界、主对偶与独立原始求解；`native_conic.py` 是大场景锥实现。
5. `new_generators.py` 与 `experiment_v2.py`：信息来源、数据生成、清单、调参和评估。
6. `g3_experiments.py`、`selection_repair.py`：未知结构压力检验与明确标为探索性的补充。

旧版 `../core.py` 提供 `solve_qp`、`MaskLikelihood` 和仿射风险族。新代码没有复制另一套未经审计的底层 oracle。

## 2. 数据流

```text
masked observations + public family + fixed reference prior
                 |
     nested scenarios / explicit importance correction
                 |
       complete-mask likelihood (once)
                 |
 joint posterior -> outer marginal / conditional posterior
                 |
 RiskSpec -> risk envelope -> weighted QP / independent conic solver
                 |
 weights + numerical bounds + status + diagnostics
                 |
 evaluator-only covariance/oracle -> true synthetic regret
```

`ScenarioSet` 保存 Q、均值、节点、参考对数质量、组编号、持仓约束和场景 oracle 界。`PosteriorState` 保存联合和条件对数质量及 ESS。`RiskSpec` 决定聚合方式。`DecisionResultV2` 区分目标值、数值上下界、gap、可行残差、对手权重、时间和状态。

## 3. 优先检查的函数与不变量

| 位置 | 核查内容 |
|---|---|
| `scenario_inference.posterior_from_loglik` | 完整似然只更新一次；按联合质量导出条件分布；没有将后验样本再无修正乘似然 |
| `new_generators.importance_scenarios` | 25% 均匀提议成分、混合提议密度、外层积分权重、真实参数不进入拟合 |
| `Envelope.blocks` / `support` | 条件化固定 alpha；熵从 log 质量计算；排序随当前持仓更新 |
| `tail_weights` | 概率质量尾部的部分边界原子；不要与固定整数 Top-k 混淆 |
| `Envelope` filter 分支 | 过滤后 k=ceil(rho*K)，有效比例 k/K；奇数 K 回归检查 |
| `Envelope` budget 分支 | 最大化 LP 原始值只给支持函数下近似；真正上界来自修复 LP 对偶与数值余量 |
| `risk_center.bounds` | 原始上界减 oracle 下界；对偶下界减 oracle 上界；惩罚项符号一致 |
| `risk_center.project_envelope` | 对手满足概率、cap、组质量和证据预算约束后才能作下界 |
| `ScenarioSet.prepare` | 缓存签名涵盖 Q、mu、上限、交易成本和旧持仓 |
| `experiment_v2.select` | 只使用开发未来实现损失；最终准则已修复，历史 selection.json 不被替换 |
| `selection_repair` | 不把复用已查看确认数据的补充称为新独立确认 |
| `g3_experiments` | 过去拟合与评分，未来只评价；oracle 用条件协方差目标而非实现冲击 |

函数签名没有 evaluator 真值是一项必要检查，不足以单独证明无泄漏；还需沿调用链核对 family、Y、先验拟合及场景生成。合成真值保存在结果中供审计，不传给 `solve_center`。

## 4. 结果与版本

精确清单：`configs/*_manifest.json`；原冻结候选：`configs/selection.json`；修订候选：`configs/selection_repair.json`。主要结果摘要是 `results/final_risk_comparison.csv`；逐实例原始字节通过压缩分片恢复。

所有正式顶层 JSONL 都公开，包括启发式、多起点诊断、负面结果和开放 gap。历史重跑及重复 worker 分片仍留在原机器，并在 `publication/local_archive_inventory.json` 列出 hash；没有用这些旧行扩充最终样本量。历史 hash 不代表每个旧源代码版本都可恢复。

## 5. 可执行检查

在仓库根目录、已安装 `requirements.txt` 的 Python 环境中运行：

```bash
# 验证公开压缩记录，不必先解压
python -m research_v2.publication verify
# 恢复正式结果；拒绝覆盖内容不同的既有文件
python -m research_v2.publication restore
# 检查清单、配对完整性、持仓可行性、状态与已保存的 gate
python -m research_v2.verify_delivery
```

以下命令独立重新执行数值检查，会更新对应 gate JSON，建议在单独 checkout/副本运行：

```bash
python -m research_v2.gates
python -m research_v2.native_gate
python -m research_v2.theory_checks
```

预期不是“所有输出都是成功”：开发有四条开放 gap，补充有两条；不连续启发式状态是 heuristic，没有虚构全局 gap。独立锥检查中某些直接路径会触发回退，这同样应保留。

完整重跑顺序在 [README](../README.md)。保留冻结 configs 和历史 results 后再启动，避免把新运行时间写入原账本。重新开发选择与重现历史确认是两件不同的事。

## 6. 建议审阅者优先提出的反驳

- 二阶 epi 微分的一般结果是否已直接推出概率尺度定理？若是，应用层的新内容还剩什么？
- 256 节点尾部误差是否足以解释条件/全局差异？应如何给统一积分误差而非只看 ESS？
- G2 的对角组内结构、低秩跨块结构和有限开发面板是否压弱了条件化的必要性？
- G3 后验集中来自真实信息、场景不足、伪后验评分过尖，还是三者共同作用？
- 两模型信息下界只匹配阶且候选族给定，这样的理论规模是否足以作为论文主贡献？

报告没有预设这些质疑都能得到有利答案。
