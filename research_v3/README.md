# 决策兼容性与证据闸门合成研究

本目录在独立分支 `research/decision-compatible-v3` 上研究有限模型 GMV。代码基于提交 `5f3bf8c45b8e0e3d8fa706c57006e45e01e6f147`，不改变 v2 或原聊天中的 A 股实验。实验状态与结论见 `RESEARCH_REPORT_ZH.md`，证明边界见 `THEORY_STATUS.md`。

## 复现顺序

在本工作区仓库根目录运行，使用根目录 requirements.txt 对应环境。本机 Python 为 `D:/Portfolio_Optimization_paper/.venv/Scripts/python.exe`。

```powershell
python -m research_v3.run_gates
python -m research_v3.experiments freeze
python -m research_v3.experiments E1
python -m research_v3.experiments E2
python -m research_v3.experiments E3
python -m research_v3.experiments E4
python -m research_v3.experiments confirmation
python -m research_v3.safety_reference
python -m research_v3.rare_precision
python -m research_v3.audit
python -m research_v3.observation_evidence
python -m research_v3.report
python -m research_v3.verify_delivery
python -m research_v3.archive
python -m research_v3.finalize
```

已交付配置是冻结协议。新机器应直接使用该配置跳过 freeze；freeze 仅用于新空结果目录的计时与规模选择，不用于在已查看结果后调节配置。再次运行完整实验前请使用新工作区保存本轮结果；原始 JSONL 采用逐数据集完成清单续跑，报告会检查重复键。硬中断发生在完成清单写入之前时，须先归档该数据集的部分行再续跑，不能无声删除或重复计算。

当前版本已修正 E4 的 ε 比例标签，无需重跑 metadata_correction。该脚本只用于本轮旧文件的可审计迁移，原文件压缩备份保存在 results/audit_history。运行器除这项标签更正外，与冻结版本哈希一致；五个算法模块哈希完全一致，验证器会重新检查。

Git 中逐实例记录以无损 `.jsonl.gz` 保存，`archive_manifest.json` 给出解压后的 SHA256。当前工作区同时保留原始 JSONL；报告和验证器在原始文件缺失时自动读取 gzip。先运行报告与验证即可审阅已有结果；不要在未解压已有输出的克隆上直接追加实验，因为它会产生只含追加部分的新原始文件。需要重新实验时使用完整新结果目录。

## 数据和接口

`compat_models` 构造公开有限候选族与高斯观测。`Family` 保存每个模型 oracle 的上下界。真实模型编号只传给模拟器与 evaluator，不传给 DCL、QP、后验或 evidence gate。高维联合样本使用高斯 Wishart 散布统计量生成精确似然；这与原始独立高斯样本的充分统计量同分布，不是用期望似然替代随机数据。

`compatible_qcqp` 在中心化、风险归一化坐标上调用 Clarabel，另用原始 QP 弱对偶公式核验。`dcl_master` 使用 HiGHS MILP 与经过证明的冲突割。`evidence_gate` 固定 pilot 候选后在独立 gate 数据检验，并仅在批准候选凸包内优化。

每次运行保存 CPU 与实际经过时间。最多两个计算进程，各自 BLAS 单线程。阶段 CPU 上限总和为 6 小时；无须把预算全部用完。初始交互式安装检查与少量调试命令未计入实验 ledger，不能把 ledger 宣称为整个研究会话的完整计费时间。

## 结果解释

成功概率、后验均值 regret、频率学错误批准率是不同目标。epsilon_ratio=1 的数值边界单独报告，不进入主要排名。超时、求解失败及模糊成功保留；不得把 solver 状态字符串直接当成证书。求解器证书是 float64 独立弱对偶检查，不是严格区间算术。

E2 的开发与确认使用独立数据种子；非对称和隐藏曲率模型也使用独立族种子。标准 M=d 正单纯形结构本身固定，不把其确认称作新几何族。有限个生成族的重复实验只支持这些固定设计，不能估计整个金融市场或所有风险族的泛化误差。

交付核验入口为 results/delivery_verification.json；最终 CPU、进程经过时间与观测到的总经过跨度分别记录在 results/final_manifest.json。核验失败历史与修正原因完整保留。本轮未运行 E5、G3、真实回测或公开推送。
