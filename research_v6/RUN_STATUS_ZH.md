# 执行状态与断线恢复

状态：**执行中，尚未形成最终实验结论**。

- 工作区：`D:\Portfolio_Optimization_paper\worktrees\portfolio-research-v6`。
- 分支：`research/decision-sufficient-information-v6`。
- 原冻结提交：`ad03493`；原配置和777,600条确认清单不改。
- 当前数值修订：`symmetric-covariance-v3`，见`implementation_revision.json`；34项Gate通过，20项数据流缓存比对通过。
- 私有纯合成运行目录：`D:\Portfolio_Optimization_paper\private_runs\synthetic_v6`。
- 确认使用固定100次重复档。旧确认/校准/终端保存在修订目录，当前顶层目录只存修复后有效记录。
- `results/runtime.json`只包含已退出阶段；实时CPU还需计入当前监督器的子进程。总预算12 CPU小时，失败/重算同样计入。

恢复时先核查是否已有`research_v6.supervisor`及其`experiments`进程；最多两个计算进程，不要重复启动同阶段。当前阶段采用原子记录，已有记录跳过。不要重新freeze；核对核心文件哈希应使用原配置及实现修订记录。

依次完成确认、terminal、stress、report、audit、archive。报告未完成时不能把局部汇总当作完整冻结结果。最终工作只本地提交，不自动推送或合并main。
