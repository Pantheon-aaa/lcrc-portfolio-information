# research_v6：决策充分信息与自适应观测

独立纯合成探索，基于`d07e06b739ad2fce00aa35e4b233e19fcb352231`。不读取行情，不使用历史校准族，不更新旧冻结结果。

阅读`THEORY_STATUS_ZH.md`与`DEVELOPMENT_LOG_ZH.md`，完成后结果位于`REPORT_ZH.md`。终端A1-B不继承序贯认证保证，oracle采样只作参照。

## 运行

从仓库根目录使用既有v4a环境（NumPy、SciPy、Clarabel、CVXPY、HiGHS、mpmath、numba、psutil、zstandard、matplotlib）。设置独立运行目录：

```powershell
$env:V6_RUN_ROOT='D:\research_runs\synthetic_v6'
python -B -m research_v6.supervisor gate
python -B -m research_v6.supervisor diagnose
python -B -m research_v6.supervisor prepare development
python -B -m research_v6.supervisor develop
python -B -m research_v6.supervisor prepare confirmation
python -B -m research_v6.supervisor freeze
python -B -m research_v6.supervisor calibrate
python -B -m research_v6.supervisor confirm
python -B -m research_v6.supervisor terminal
python -B -m research_v6.supervisor stress
python -B -m research_v6.supervisor report
python -B -m research_v6.supervisor audit
python -B -m research_v6.supervisor archive
```

不要重新freeze覆盖确认方案。默认顺序执行监督器；总CPU上限43,200秒，优化阶段34,560秒，留120秒退出余量。动作成本、标量数量、CPU与wall时间分开报告。续跑跳过原子落盘的记录，错误修订需保留原记录。

`models.py`定义公开族/动作；`certificates.py`定义有理弱对偶证书；`likelihood.py`定义完整mask似然和区间；`information.py`解信息分配LP；`sequential.py`分离模拟器与策略；`terminal.py`比较信息耗尽后的决策。真模型只由模拟器/evaluator或明确标记的oracle分配参照读取。
