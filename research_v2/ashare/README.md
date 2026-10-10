# A 股真实数据研究

本目录实现条件尾部算法的真实数据研究。行情、逐证券持仓和逐日账本保存在仓库外的私人目录；`output/` 仅包含不带证券明细的汇总、检查结果和图表。已完成实验的主要结果、推导说明和限制见[详细报告](REPORT_ZH.md)。

## 研究的问题

**当股票历史不完整、共同观测不均匀时，保留较充分信息的概率权重，只对仍不确定的风险模型做条件尾部重加权，能否比模型平均或标准协方差收缩得到更低、且对场景构建稳定的样本外组合风险？**

我们分别检验平均风险、较差时期表现和条件信息处理的独立作用。外层 bootstrap 描述组内风险，内层描述跨组风险；这一操作性分解不等于已经证明真实市场参数的可识别性。实验不声称找到了最小充分统计量，也不将未来实现风险写成真实总体协方差 regret。

本轮完成 456 个主测试及延伸决策窗口，另有场景精度和 50 只股票检查。**目前没有支持条件尾部稳定优于同信息 Model Averaging 的证据。** 主测试期条件尾部二阶损失比 EM＋收缩高约 9.8%–19%；场景重建的随机性大于微小的方法差异。代码、冻结配置、汇总结果和失败诊断一并公开，便于复核这些负面结果。

## 复现顺序

从仓库根目录运行，Python 环境沿用项目环境，另需 `pyarrow`、`pandas`、`scikit-learn`、`psutil`。全部 BLAS 线程设为 1，主实验至多 4 个工作进程。

```powershell
python -m research_v2.ashare.data
python -m research_v2.ashare.gates
python -m research_v2.ashare.run initialize
python -m research_v2.ashare.run timing --hours 1
python -m research_v2.ashare.run freeze
python -m research_v2.ashare.audit temporal
python -m research_v2.ashare.run scores --hours 1
python -m research_v2.ashare.run choose-power
python -m research_v2.ashare.run tune --hours 1
python -m research_v2.ashare.run choose-tuning
python -m research_v2.ashare.run test --hours 6
python -m research_v2.ashare.run audit --hours 1.5
python -m research_v2.ashare.run fifty --hours 1.5
python -m research_v2.ashare.bounds_audit
python -m research_v2.ashare.finalize_records
python -m research_v2.ashare.evaluate test
python -m research_v2.ashare.evaluate test --actions
python -m research_v2.ashare.evaluate fifty
python -m research_v2.ashare.audit integration
python -m research_v2.ashare.mechanism
python -m research_v2.ashare.sample_audit
python -m research_v2.ashare.figures
python -m research_v2.ashare.figures --actions
python -m research_v2.ashare.verify
```

默认数据路径和私人结果路径定义于 `common.py`，可用 `ASHARE_SOURCE`、`ASHARE_PRIVATE` 环境变量替换。输入要求与现有研究 Parquet 相同的字段语义，不能用无分红的未复权价格直接替换收益率。缓存记录文件大小、修改时间、行数与日历。换输入或修改算法后，应使用新的私人目录；缓存不是跨版本自动失效的数据库。

如果只复现已经冻结的测试结果，保留本目录 `configs/base.json`、`scale.json` 和 `selected.json`，在新的 `ASHARE_PRIVATE` 中执行数据准备、gates、test、audit、fifty 和各评价命令即可；不要重新运行 initialize／freeze／choose 系列去覆盖发布的配置。完整重新调参的流程使用上面的全部命令，计时选规模可能因机器而变化。`verify.py` 验收完整研究流程，测试阶段单独复现时会如实报告尚未重跑的验证任务。

已有单窗口输出使运行可恢复；算法修改后不可无说明地混用旧缓存。本轮修复与未覆盖事项见报告。缓存内的场景 oracle 仅指有限模型内最优目标值，不是未来真实最优组合。

本轮验证阶段曾遇到大似然惩罚造成的数值问题，`repair_lp.py` 只用于修复当时已产生的旧记录，原始记录保留在私人归档目录。全新运行已直接使用等价的冗余约束移除，不需要再执行该修复。`bounds_audit.py` 则利用最大似然场景 regret 非负这一恒等事实，给全局证据惩罚目标补充解析下界 0；它不修改持仓、调参或收益，只改进数值最优性判断，并保留原数值界。

## 信息权限和单位

- 股票选择和拟合只访问决策时点及以前。验证期目标的未来评价必须在 2019 年结束。
- 拟合函数只接收过去 252 日收益；拟合 189 日、评分 63 日。未来收益只由评分幂次选择器和回测评估器读取。
- EM、逆波动率基线可使用全部 252 日历史，避免给基线更少信息。LW 对照使用回看期所有资产同时有记录的完整行，要求至少 60 行；其适用性不外推为缺失数据 LW 理论保证。
- 数值优化使用开发期固定风险尺度归一化。结果中的 `objective`、`gap` 是归一化有限场景目标；真实评价的二阶损失使用原始小数收益。
- bootstrap 参考质量与幂次高斯评分构成近似或广义后验。固定温度、有限场景最坏值以及模型间尾部均不构成真实风险置信证书。

## 执行与统计

`book.py` 维护现金与市值持仓，先应用本次开盘的估值变化和已知分红，再卖后买。无法成交部分不假定完成。主账本连续跨越换股和窗口边界；验证损失另用每窗口新账户，明确分开。缺报价的估值近似和无法核实的退出不能通过补零收益掩盖。

`--actions` 评价另行加入一项已核实的换股事件：600837 于 2025-03-17 按 1:0.62 转为 601211，依据事前公布的发行人公告。`actions.py` 给出公开来源、分数股假设及避免重复估值的测试。它不改变股票池、拟合、参数或目标持仓；原账本完整保留，修正账本用 `_actions` 命名。该选项只修复这一已识别事件，不代表拥有完整公司行动库。报告以修正账本为主要评价。

方法独立计时只覆盖场景优化器。基础基线最初使用零占位，`finalize_records.py` 将其标注为未单独测量并导出为 null，保留原占位值；不能把它读成基线没有计算成本。整窗口 CPU／墙钟计时包含基线计算。

主要配对比较先对同日三个面板平均，再用 63 日区块共同重采样 2,000 次。没有把面板或滚动窗口当作独立市场路径。仅对两项条件尾部、两项基线、两类股票组共八项比较进行 Holm 校正。可靠报价子集是敏感性诊断，其不连续日期的复合收益不解释为可执行策略轨迹。

## 范围

本轮固定长仓、全额预算、目标单股权重不超过 20%，不做收益预测。不完整历史来自可观察记录长度，不推断缺失原因或真实上市时间。股票池来自当时记录，不声称缺少权威退市和代码变更表时已完全消除幸存者偏差。真实数据没有可观察的总体协方差 regret。
