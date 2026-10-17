# research_v6：决策充分信息与自适应观测

独立纯合成探索，基于`d07e06b739ad2fce00aa35e4b233e19fcb352231`。不读取行情，不使用历史校准族，不更新旧冻结结果。

**定位：这是一次算法与论文问题表述的可能性尝试，不是建议立即替换现有算法，也不是已决定的论文转向。** 请独立评估保留原方案、只采纳部分机制、或继续新方向三种选择。2026-10-17按用户要求直接发布到独立实验分支，不创建PR、不合并main。实验结束时的本地交付说明保留为历史记录。

给GPT／Claude的完整审阅任务见[审阅prompt](REVIEW_REQUEST_ZH.md)。审阅者尚未给出结论；不得把这个prompt或发布本身视为外部认可。

先读[综合判断](INTERPRETATION_ZH.md)和[完整报告](REPORT_ZH.md)，再查[理论状态](THEORY_STATUS_ZH.md)、[执行状态](RUN_STATUS_ZH.md)及[开发日志](DEVELOPMENT_LOG_ZH.md)。终端A1-B不继承序贯认证保证，oracle采样只作参照。

## 运行

从仓库根目录使用既有v4a环境（NumPy、SciPy、Clarabel、CVXPY、HiGHS、mpmath、numba、psutil、zstandard、matplotlib、pandas）。确切版本见`results/environment.json`。以下为复用已冻结配置的重现流程，设置全新的独立运行目录：

```powershell
$env:V6_RUN_ROOT='D:\research_runs\synthetic_v6'
python -B -m research_v6.supervisor gate
python -B -m research_v6.supervisor diagnose
python -B -m research_v6.supervisor prepare confirmation
python -B -m research_v6.supervisor calibrate
python -B -m research_v6.supervisor confirm
python -B -m research_v6.supervisor terminal
python -B -m research_v6.supervisor stress
python -B -m research_v6.supervisor stress-profile
python -B -m research_v6.supervisor report
python -B -m research_v6.supervisor audit
python -B -m research_v6.supervisor archive
```

不要重新freeze覆盖确认方案；未压缩或gzip清单存在时，冻结器都会拒绝覆盖。开发阶段原始命令为`prepare development`、`develop`、`freeze`，其计时与原结果已保存，此处不重复冻结。默认顺序执行监督器；总CPU上限43,200秒，优化阶段34,560秒，留120秒退出余量。动作成本、标量数量、CPU与wall时间分开报告。续跑跳过原子落盘的记录，错误修订需保留原记录。

本次S3终端阶段按配额只完成部分清单。复现的CPU边界可能因硬件产生不同的截断位置；比较科学结果时使用已交付的完成/未完成清单，不将新增运行与本次确认统计混合。

`models.py`定义公开族/动作；`certificates.py`定义有理弱对偶证书；`likelihood.py`定义完整mask似然和区间；`information.py`解信息分配LP；`sequential.py`分离模拟器与策略；`terminal.py`比较信息耗尽后的决策。真模型只由模拟器/evaluator或明确标记的oracle分配参照读取。
