# 平均代价受控的决策兼容性研究

论文写作与后续交接请先读 [论文证据留存备忘录](PAPER_EVIDENCE_MEMO_ZH.md)：逐项保存正面结果、逐族证据、可引用措辞和必须补齐的缺口。该文档不改变原始审计状态。

研究问题：相对 MA 的后验平均 regret 只允许增加 10% 或 25% 时，能否保留无预算 A1 的实际达标率优势？对应三个实验分别核验精确多资产价格、非对称新族的成功—代价前沿与信息平台、同目标精确求解的计算成本。

本目录从 `edac208d27e8751faab5a2ca94a1cd267cd4439b` 建立，历史研究不修改。结果与判断见 `REPORT_ZH.md`，条件性证明见 `THEORY.md`，用户协议原文见 `INPUT_PROTOCOL_ZH.md`。

## 复现

在仓库根目录运行。版本清单为本目录 `requirements.txt`；冻结算法哈希保存在 config_frozen.json。本机独立解释器为 `D:/Portfolio_Optimization_paper/.venv-v4a/Scripts/python.exe`。最多两个计算进程，包入口强制 BLAS 单线程。

```powershell
python -B -m research_v4a.gates
python -B -m research_v4a.secondary_tie_audit
python -B -m research_v4a.experiments S1
python -B -m research_v4a.boundary_checks
python -B -m research_v4a.experiments freeze
python -B -m research_v4a.experiments development
python -B -m research_v4a.experiments confirmation
python -B -m research_v4a.repair_decisions development
python -B -m research_v4a.repair_decisions confirmation
python -B -m research_v4a.platforms
python -B -m research_v4a.computation benchmark
python -B -m research_v4a.computation perturbation
python -B -m research_v4a.enumeration_audit
python -B -m research_v4a.precision_audit
python -B -m research_v4a.numerical_boundaries
python -B -m research_v4a.refine_certificates
python -B -m research_v4a.provenance
python -B -m research_v4a.posterior_precision
python -B -m research_v4a.summarize
python -B -m research_v4a.audit
python -B -m research_v4a.delivery_check
python -B -m research_v4a.archive
```

首次本轮运行按上述顺序建立冻结配置；复现实验使用已交付 config_frozen.json，不重新选择阈值。已有结果先通过汇总与审计阅读，重做实验请复制到新的工作区并使用新的空 results 目录，避免覆盖已发布记录。每个完整数据集有完成清单；硬中断可能留下半个数据集，必须先保留其部分记录并标明 incomplete，再继续，不能重复追加后无声去重。

## 接口及证书

公开入口 `research_v4a.api.solve_budgeted_dcl(family, posterior, epsilon, beta, geometry_cache=None, posterior_price_cache=None, time_limit_seconds=30)` 返回持仓、预算和成功区间、主质量 gap、全局二级价格 gap、残差与耗时。`price_oracle` 返回指定模型集合共同达标的最低平均增量区间。真实模型身份只属于观测模拟及评价器。`api.py` 对全部目标矩阵完全相同的族提供结构性 L*=0 认证，仍将浮点持仓的零预算认证保留开放；不根据后验下溢判定结构性零。冻结实验族均不满足该退化分支，因此直接调用同一通用求解器。

价格缓存必须与后验、完整族、约束和 ε 的哈希匹配；几何缓存可以跨数据使用，价格缓存不能。β=∞ 的新版全局破同分和旧 A1 的索引破同分分开保留。零预算退回 MA，空成功集合合法。

全局主／次目标同值时，通用求解器使用固定模型变量顺序并保留首个认证 incumbent，只有严格较低的平均代价上界才替换；枚举最后以集合编号排序。不同求解器不要求返回相同持仓，要求目标区间与可行证书一致。

## 数据与解释

Separated 在两个独立时点分别观察两组资产；Joint 在一个时点观察全资产，另一个为空。二者每块观察的标量数相同。不得从未观测的跨组收益估计后验。

主目标为给定数据的后验 ε-成功质量，不是赚钱概率。后验平均预算不保证每个真实模型的平均损失也满足同一比例。ε 曲线是按不同容忍度重新求解的政策族，不能称为同一组合的 regret CDF。

全部超时、数值不确定性和二级未闭合状态保留。float64 弱对偶加舍入余量属于数值证书；并非严格有向舍入证明。预算近零或后验下溢时不能添加 floor 后仍声称求解原问题。

计时字段说明：`cold_seconds` 是通用结果容器中的单次调用经过时间；S2 会跨后验复用合法几何缓存，不能把该字段直接称为真正冷启动。真正的冷启动／缓存消融只比较 S3 对应方法。S1 对称组价格先行构建，其结果行的 `cold_seconds` 只计封装开销；完整价格预计算计时另在 S1_boundaries.json，禁止用封装耗时声称对称约简速度。

调用／割计数是对应缓存对象的累计计数；S3 缓存逐查询增量须对同一 case 按预算顺序作差。旧 A1 包装器中的零调用计数表示该内部计数未暴露，不用于调用数比较。零样本的两种 mask 是相同先验的设计视图，均只求解一次后评价全部真模型，不作为额外独立风险族或统计样本。

实验台账在 results/runtime_ledger.jsonl，包含失败的计算阶段、Python 导入消耗和高精度核验。环境安装、Git 文件操作和交互文字编辑不属于实验 CPU。调度器同时检查已结束阶段和仍在运行的实验进程，6 小时上限留出最终写入余量。依赖安装只影响独立 v4a 环境。

台账的 `source_hash` 是阶段结束时的磁盘源码快照，不能据此推断一个已运行数小时的 Python 进程加载了哪个版本。初始确认进程继续使用已加载的旧 MILP 模块；正式重算使用 `validated_algorithm_sha256` 和 `numerical_revision=scaled-mass-v2` 标识，审计逐文件核验这些哈希。初始数据流只提供固定观测及未受修复影响的其他基线，不能用其结束时磁盘哈希冒充已经重算。

本轮压缩归档因 CPU 预算耗尽未执行，不能从 Git 提交独立重建完整原始决策流。全部原始 JSONL 仍在本工作区 results 目录；关键小型原始记录与汇总纳入提交，大型初始和正式决策流只在本地保存。archive.py 已准备逐行无损 `.jsonl.partNNNN.zst` 分片与 SHA256 清单，但新版压缩流程尚未运行验证。不要删除本工作区。交付缺口见 DELIVERY_STATUS_ZH.md。

数值修订分两部分，全部保留原始记录。第一部分收紧枚举在无限预算下因 `inf <= inf` 产生的过宽上界，S1 / S3 收紧证书另存 `*_refined_certificates.json`，原始平台文件另存 `S2_platforms_initial.jsonl`。第二部分是实质性的 MILP 数值修复：初始未缩放目标曾漏掉约 1e-8 的后验质量，使目标上界低估约 1e-7。`scaled-mass-v2` 将质量目标与质量约束乘 1e6，并用离散质量间隔限定全局二级认证；所有 S2 的 A1-B 和旧 A1 都在同一批数据上重算，完整 S3 同步重跑。其他基线原样保留。正式统计只读 `S2_*_validated`；不能混用初始结果来增加样本量。`frozen_sources/`、原始配置、修订说明、新哈希、反例和独立枚举回归检查均已归档。

以上命令用于干净目录的复现。交付代码已经包含数值修复，`repair_decisions` 在复现时会再次校验并生成正式 validated 数据流；它不会重抽观测。历史上为了保留在途确认进程的其他基线，先完成初始数据流再统一重算，额外 CPU 已计入总预算。只有重做历史数值修复的额外 S3 时才使用 `--numerical-repair`，它不提高总预算。

后验精度：公开 API 可接受 `log_posterior=` 作为精度证据。输入中出现零概率且无结构性认证时，返回 `posterior_precision_open`，不能将其当成结构零。对实验的有限正密度、高斯似然，用保存的日志计算误差界并单列 posterior_precision.json；汇总扩大相应数值区间，原始决策及实际成功评价保持不变。原始状态是名义浮点后验下的结果，须连同这些资格修订阅读。

严格预算资格：`certificate_qualification.py` 统一使用 `Δ_upper <= B_lower`，零预算也不例外。β=0 仍返回 MA；但其浮点表示若有正的 QP 误差界，只能称为协议的 MA 端点，不能认证该浮点持仓满足字面上的零增量。公开 API、审计和正式统计将没有已认证可行 incumbent 的主下界保守置零，并保留开放二级证书；零预算主目标取开放区间。原始名义字段和全部持仓原样保留，不借此更改任何实际成功事件或 10% / 25% 的配对比较。原始 JSONL 必须通过这些资格规则阅读；`config_frozen.json` 记录该证书修订与源码哈希。

预算检查点恢复时，只有完整数据集追加到正式决策流；在中断前产生的部分价格证书尝试也保留，因此证书流可以有重复尝试，正式决策流的唯一性则由完成清单及逐键审计保证。
