# research_v5b_ashare

真实历史风险实验与高斯机制桥接。规范见 `PROTOCOL_AMENDMENT_ZH.md`，原协议位于 `../research_v5_ashare/INPUT_PROTOCOL_ZH.md`；冻结参数见 `protocol_frozen.json`。结果完成后见 `REPORT_ZH.md`。

训练只用原始 NaN。仅未来评价和研究账本使用估值沿用近似，不能将本目录的净值当作精确可交易回测。

## 环境与私有数据

```powershell
$env:ASHARE_RUN_ROOT = 'D:\private_runs\ashare_v5b'
$env:ASHARE_CACHE_ROOT = 'D:\private_runs\ashare_v5\data'
$env:ASHARE_SOURCE = 'D:\licensed_data\a_share_research.parquet'
$env:ASHARE_MEMBERSHIP = 'D:\licensed_data\verified_index_history'
```

`ASHARE_CACHE_ROOT` 是旧v5只读审计缓存，必须与源文件SHA256匹配。若从零重建，可令缓存目录为新私有目录的 `data/`，先运行 `audit-data`；本轮复用了既有缓存和过去递推面板。不可在公开仓库存放证券级数据。

依赖使用原v4a虚拟环境的 NumPy/SciPy/Pandas/PyArrow/CVXPY/Clarabel/HiGHS/scikit-learn/psutil/matplotlib/zstandard。最终交付检查保存具体版本。

## 阶段命令

在仓库根目录运行，所有数值阶段通过资源监督器。建议顺序运行；不要同时启动两个长任务。

```powershell
python -B -m research_v5b_ashare.supervisor prepare
python -B -m research_v5b_ashare.supervisor calibrate
python -B -m research_v5b_ashare.supervisor validate
python -B -m research_v5b_ashare.supervisor benchmark
python -B -m research_v5b_ashare.supervisor validate-information
python -B -m research_v5b_ashare.supervisor validate-f2
python -B -m research_v5b_ashare.supervisor gate0
python -B -m research_v5b_ashare.supervisor freeze
python -B -m research_v5b_ashare.supervisor r1
python -B -m research_v5b_ashare.supervisor semisynthetic
python -B -m research_v5b_ashare.supervisor batch R2
python -B -m research_v5b_ashare.supervisor information
python -B -m research_v5b_ashare.supervisor batch R3b
python -B -m research_v5b_ashare.supervisor batch R4-N
python -B -m research_v5b_ashare.supervisor batch R4-M
python -B -m research_v5b_ashare.supervisor batch F2
python -B -m research_v5b_ashare.supervisor batch F2-library
python -B -m research_v5b_ashare.supervisor evaluate
python -B -m research_v5b_ashare.supervisor report
python -B -m research_v5b_ashare.supervisor archive
python -B -m research_v5b_ashare.supervisor delivery-check
```

本轮完整指数在测试前因成本外推排除，计时删失证据保存在 `output/full_index_budget_bound.json`。不能把未完成计时当成快速求解。主清单、源哈希和开关已冻结；恢复任务时直接使用私有归档中的清单，**不要在已看到测试结果后重新运行 freeze 改变开关**。

实例原子落盘，已完成实例跳过。需要修复实现错误时先保留旧记录到修订目录，再按同一清单重算受影响项，不按结果补跑。旧v5的271.6875 CPU秒已计入24小时总上限；19.2小时为优化上限，4.8小时保留评价与归档。

N=100阶段可在第二个计算槽空闲后运行`python -B -m research_v5b_ashare.supervisor prefetch`。它仅调用相同前端预先准备冻结任务缓存，不运行决策或评价；与主决策进程合计最多两个计算进程。通常仍建议按上述命令顺序复现。

## 模块边界

- `risk.py`、`data.py`：原始历史、EM、预序评分与面板；不读取估值填补。
- `valuation.py`、`evaluator.py`：研究估值视图、独立未来QP及交叉拟合评价。
- `decision.py`：公共v4a A1-B适配、共同预算基线、操作性回退和guarded版本。
- `bridge.py`、`information.py`：已知模型半合成与等标量观测实验，真模型身份不进入决策接口。
- `proxy_book.py`：可分割总收益单位、收盘调仓及成本重放。
- `runner.py`、`reporting.py`：固定清单跑批、聚合、同步月份块推断和私有无损归档。

所有公开结果均为聚合或无证券标识的实验记录。数据源、证券级持仓/收益、模型矩阵和似然保存在独立私有目录；不会自动推送GitHub。
