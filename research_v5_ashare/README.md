# research_v5_ashare

状态：**开发标签不足，尚未冻结或运行主测试。** 先阅读 [中文阶段报告](REPORT_ZH.md) 和 [交付状态](output/delivery_status.json)。本目录不是一套已经完成 R1—R4 的市场实验。

基础版本：`f19453377892cbc7bb9aa8a79f3cfc117e74eb0b`。原始协议与 SHA256 分别保存于 `INPUT_PROTOCOL_ZH.md` 和 `protocol_frozen.json`；后者明确标记 `NOT_FROZEN_CALIBRATION_BLOCKED`，不应被当作有效参数冻结。

## 可复现运行

在仓库根目录、含 `output/versions.json` 所列依赖的 Python 环境中运行。所有路径由环境变量指定；私有输出目录不得放入公开仓库。

```powershell
$env:ASHARE_RUN_ROOT = 'D:\private_runs\ashare_v5'
$env:ASHARE_SOURCE = 'D:\licensed_data\a_share_research.parquet'
$env:ASHARE_MEMBERSHIP = 'D:\licensed_data\verified_index_history'
python -B -m research_v5_ashare.supervisor audit-data
python -B -m research_v5_ashare.supervisor gate0
python -B -m research_v5_ashare.supervisor timing
python -B -m research_v5_ashare.supervisor calibrate
python -B -m research_v5_ashare.supervisor label-audit
```

如果校准不足，不运行验证或测试。当前交付的描述性几何诊断可以独立重现：

```powershell
python -B -m research_v5_ashare.supervisor r1-geometry
python -B -m research_v5_ashare.supervisor n100-attempt
python -B -m research_v5_ashare.supervisor report
python -B -m research_v5_ashare.supervisor archive
python -B -m research_v5_ashare.supervisor delivery-check
```

请顺序启动阶段，**不要并行启动多个 supervisor**。监督器汇总 Windows 启动器及其子进程 CPU；普通阶段在 19.2 小时配额前停止，最终化阶段才可使用 4.8 小时预留，提前保留 120 秒退出余量。一次最多一个数值计算进程，BLAS 单线程。失败阶段也计费，已完成实例按文件跳过。公开汇总可重复生成；证券级记录仅在 `ASHARE_RUN_ROOT`。

若源数据更换，应使用新的私有运行目录，不能复用旧 frontend/面板缓存。归档不重复打包大型可重建源缓存，但保留源哈希和索引元数据；要恢复全部内容，需要相同授权源数据。当前 `validate` 有实现但未运行，不意味着主测试跑批已经实现。

## 实现边界

- `data.py`：只读审计、日历、资格过滤、共同排序面板。
- `risk.py`：零均值 EM、收缩风险库、严格逐步预测评分。
- `decision.py`：复用 v4a 公共 A1-B 的单位适配、共同输入、回退与 guarded 结果。
- `evaluator.py`：未来完整标签和独立 PSD QP 上下界，与决策模块分离。
- `ledger.py`：可测试的原始股数/现金原语；不是已完成的真实市场账本。
- `pipeline.py`：实际已执行阶段及可恢复实例记录。
- `checks.py`：已实现范围的18项检查；未实施的 R3/真实账本检查明确列为待办。
- `reporting.py`：聚合报告、图表、未完成清单、私有无损归档校验。

尚未实施完整 R2/R3/R4 跑批、Greedy+1swap、实用基线调度和时间块统计；原因与恢复条件见报告。不得将未计算的六项检验空值当成不显著结果，不得将均匀权重几何诊断冒充预定公共阈值下的真实达标率实验。

## 数据与隐私

公开目录没有证券级收益、持仓、风险矩阵、成分明细或授权行情。协议中的数据源由用户持有，重现者需自行取得合法访问。没有自动推送、付费数据下载或外部计算任务。
