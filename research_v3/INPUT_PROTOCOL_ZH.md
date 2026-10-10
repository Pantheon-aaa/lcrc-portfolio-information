# ICML 组合优化研究：论文背景、两个候选方向的形式化与 Codex 预热实验协议

> **研究状态：2026-10-10；候选研究方案，非已完成定理或已验证新算法。**  
> 基础仓库：[Pantheon-aaa/lcrc-portfolio-information](https://github.com/Pantheon-aaa/lcrc-portfolio-information)，核对版本 `4d3974ed06ac0139e44e18fc9be32ee388c0613d`。现有 `research_v2` 是基线及理论基础；建议在新目录 `research_v3/` 隔离实现，不修改或覆盖 v2 的冻结测试记录。  
> **计算边界：** 不训练神经网络，笔记本 CPU、低维风险参数、3—20 只股票/少量风险场景做机制实验；需要扩展时再使用既有 G2/G3。  
> **本文中的命题状态：** 标为“可直接证明”的，是在写明条件下可由标准工具严格推出的引理或反例；标为“候选主定理”的，是下一阶段需要完整证明的研究目标。构造新目标、套用已有定理，不能直接宣称 ICML novelty。

---

## 0. 一页总览

**核心背景：** 股票风险依赖联合收益；部分资产没有足够共同历史时，不同协方差矩阵可以解释同一份数据。估计整个矩阵未必必要，因为多个模型可能支持同一个近优组合。但一些没有被排除的模型会限制新资产进入持仓，甚至它们的后验概率已很小，仍会影响决策。我们要从“学习市场参数”转向“学习足以作出好决策的信息”。

| | 方向 A：Decision-Compatible Learning (DCL) | 方向 B：Evidence-Gated Portfolio Learning (EGPL) |
|---|---|---|
| **精确问题** | 选择一个在**最大后验模型质量**上满足 `regret ≤ ε` 的组合 | 什么观测证据足以**安全解锁某个持仓调整**？ |
| 主要评价 | `ε-成功概率`、后验质量、频率学失败概率 | 错误解锁概率、解锁延迟、条件 regret、所需联合样本数 |
| 直接方法 | 后验兼容子集选择 + 凸 QCQP 可行性 oracle | 带复合零假设的似然比/e-value gate + 联合 QP |
| 已有理论接口 | `K_j(ε)` 交集、Helly、小冲突证书、不可识别模型 | 不同临界锥、后验概率 `p_t/t` 的三阶段、联合观察信息界 |
| **不是创新的部分** | 机会约束、最大可行子系统、Bayes 0–1 决策、Helly | 二模型似然比检验、e-value、Bayes 买入阈值、鲁棒安全约束 |
| 应争取的真正增量 | 连续多资产 **decision-compatible** 目标的计算与信息复杂度，特别是**多模型联合冲突** | **多方向/未知干扰参数**中的决策相关证据门槛、统一有效性与近下界可达性 |
| 建议 | **理论主问题 + 小 M 精确求解验证** | **理论对应算法 + 二模型与多方向机制验证** |

**止损标准：** 如果 A 仅仅是把熟悉的 chance constraint 用在 GMV 上，或者 B 仅仅把 MA 的 KKT 阈值改写为 likelihood ratio，而没有更一般的定理/算法增量，则不要将其作为论文方法创新。

---

# Part I. 文章背景（拟写 Intro 的中文版）

## 1. 为什么“缺失信息”是一个组合学习问题，而不只是缺失数据问题？

股票组合依赖资产**之间**的关系。一只股票自己的波动率可以从单独历史中估计，但资产之间能否分散风险，必须依赖共同交易日期的信息。在新上市、历史长度不均、停牌或跨市场记录不同步的环境里，有些资产对有大量重叠观测，有些资产对只有很短的重叠历史。因此，**不同联合风险模型可能与相同的可观测数据相容**。

传统做法是先恢复协方差矩阵，再求最小方差或均值—方差组合：

\[
\widehat\Sigma\ \longrightarrow\ \widehat w=\arg\min_{w\in\mathcal W}\frac12w^\top\widehat\Sigma w.
\]

但这隐含了一个通常没有经过检验的要求：为了找到好持仓，我们是否一定要学会全部协方差？有些未知参数可能根本不改变最优组合；反之，某些看似很小的未知相关性，可能恰好决定某只资产能否进入持仓。

机器学习里的 Decision-Focused Learning 已强调预测误差与决策 regret 不等价，PEAR（ICML 2026）进一步说明了固定局部活跃面上曲率和切空间如何筛除与决策无关的预测误差。然而，在我们的问题中，**被学习的协方差本身就是优化目标的 Hessian**，不同观测相容市场可以有不同曲率和临界锥，不能简单地以一个固定投影代表所有市场。

## 2. 为什么经典鲁棒优化和后验平均都不能完整回答问题？

一种方式是保留所有可能的协方差，最小化最大 regret：

\[
\min_{w\in\mathcal W}\sup_{\theta\in\Theta(D)}R_\theta(w).
\]

这是相对稳健组合的已有范式，并不是我们的新目标。它保护每个未排除市场，即使其中某个模型与数据已经很不一致，也可能形成长期的持仓阻碍。原 LCRC 的探索失败说明了一个实际障碍：若计算外覆盖太宽，新增联合信息根本不能改变优化结果。

另一种方式是模型平均：

\[
\widehat w_{\rm MA}\in\arg\min_w\mathbb E_{\theta\mid D}R_\theta(w).
\]

在正确有限模型后验及相同的损失准则下，它是**最优的 Bayes 均值决策**。因此新算法不能以“普遍在正确后验的均值 regret 上击败 MA”为研究目标。值得回答的是另一种不同的问题：**现有信息是否已足以保证 `regret ≤ ε`？若尚不够，哪些候选市场必须被区分，才允许改变持仓？**

## 3. 我们希望揭示的统计—几何障碍

一个错误模型即使后验概率不断下降，仍可能改变交易决策。这是因为正确调整可能带来二阶小量的收益改善，而在错误模型中越过持仓边界会造成一阶损失。已有 `research_v2` 的概率尺度定理刻画了三种极限：当错误模型质量相对调整尺度 `t` 衰减慢、同阶、快时，分别产生硬约束、软惩罚和消失项。其多资产高斯子类还把联合观察信息与达到指定 regret 的必要样本阶联系起来。

但这些结果并未自动给出一个**在一般多模型、多可行方向下工作，且优于恰当基线**的算法。特别是，当多个模型各自都有好决策，却不存在一个共同近优组合时，问题不是简单的参数点估计，而是**带有多个可接受答案的统计识别与凸可行集合的交叠结构**。

## 4. 本文拟研究的两个互补问题

**问题 A（决策兼容性）：** 如果只要求 `regret ≤ ε`，我们是否能够选择一个组合，让它在尽可能多的可信市场中达标？实现这个目标所需的统计信息是否少于精确识别市场参数？

**问题 B（证据释放边界）：** 如果当前持仓在一组候选市场中受到约束，哪些联合收益证据足以排除真正危险的反对模型，允许一次低 regret 的持仓调整？在多方向和未知曲率下能否得到紧的样本需求？

二者不是互斥方法：A 提供一个目标和“哪些市场在决策上冲突”的结构，B 提供一种利用证据触发候选持仓变化的统计机制。如果两者能被同一信息界连接，就会形成“信息必要性—算法可达性”的论文叙事；如果连接不上，也应依据预热实验明确择一。

**应用边界：** 以股票组合优化为核心；数学主实验用高斯/有限风险族研究可识别性，真实 A 股用于额外样本外验证。停牌、上市历史和非随机缺失不能自动套用外生 mask 的高斯保证。

---

# Part II. 统一数学模型和必须分清的三类目标

## 5. 观察模型与决策模型

- `d` 只股票，长期约束 `W={w∈R^d: 1ᵀw=1,0≤w_i≤u_i}`，也可额外含固定多面体约束。
- 风险情景 `θ∈Θ`，`Σ_θ ≻0`，决策目标 `f_θ(w)=½ wᵀQ_θ w−μ_θᵀw+g(w)`。GMV 取 `Q_θ=Σ_θ, μ_θ=0, g=I_W`。对所有模型 `g` 相同且闭凸多面体；`Q_θ` 在可行方向统一强凸。
- 每期看到 `Y_{r,S_r}`：收益向量仅在 mask `S_r` 的坐标被观察，收益在理论部分独立高斯，mask 固定或与收益、参数独立。完整观测似然为

\[
\ell_\theta(D)=-\frac12\sum_r \Big(\log\det\Sigma_{\theta,S_rS_r}+Y_{r,S_r}^{\top}\Sigma_{\theta,S_rS_r}^{-1}Y_{r,S_r}\Big)+\text{common constant}.
\]

- `V_θ=min_{w∈W} f_θ(w)`，`R_θ(w)=f_θ(w)-V_θ≥0`。
- 有限候选 `θ_j` 的预先确定参考质量 `b_j>0`。准确后验为

\[
\pi_j(D)=\frac{b_j\exp\ell_j(D)}{\sum_kb_k\exp\ell_k(D)}.
\]

  注意：真实不可识别时，相同观测分布的模型无法仅靠数据调整彼此的后验赔率。

### 三种目标不能混用

**均值 regret：** `E_{θ|D} R_θ(w)`，正确后验下 MA 为其最优解。

**近优成功概率：** `P_{θ|D}(R_θ(w)≤ε)`，方向 A 的目标；这是一种后验概率，**不能冒充 95% 频率学保证**。

**频率学 δ-正确性：** 对每个真实 `θ`，`P_{D~P_θ}(R_θ(ŵ(D))>ε)≤δ`，方向 B 的严格校准目标之一；需要额外检验论证和模型条件。

---

# Part III. 方向 A：Decision-Compatible Learning (DCL)

## 6. 核心定义：ε-近优与相容市场集合

对每个模型定义凸近优集合

\[
K_j(\varepsilon)=\{w\in W:R_j(w)\le\varepsilon\}.
\]

对模型子集 `G⊆{1,…,M}`，定义

\[
\Gamma(G)=\min_{w\in W}\max_{j\in G}R_j(w).
\]

`G` **ε-兼容** 当且仅当 `Γ(G)≤ε`，即 `∩_{j∈G}K_j(ε)≠∅`。

**方向 A 的精确后验问题**：

\[
\boxed{\quad\widehat w_A(D)\in\arg\max_{w\in W}\sum_{j=1}^M\pi_j(D)\mathbf1\{R_j(w)\le\varepsilon\}.\quad}\tag{A1}
\]

直观上：输出一个**在最多后验概率质量的模型下足够好**的持仓，而非平均 regret 最小或仅保护最坏市场。

### 命题 A0（可直接证明；不是独立 novelty）

\[
\boxed{\max_{w\in W}\sum_j\pi_j\mathbf1\{R_j(w)\le\varepsilon\}
=\max_{G:\Gamma(G)\le\varepsilon}\sum_{j\in G}\pi_j.}\tag{A2}
\]

**证明：** 任一持仓 `w` 诱导 `G(w)={j:R_j(w)≤ε}`，该集合有共同见证 `w`，故满足右边约束。反之，任一兼容 `G` 有公共 `w`，它至少让 `G` 达标。两侧互为上/下界。**因此从连续持仓选择转成“选择兼容场景子集”是严格等价，而非松弛。**

但 A1/A2 属于已有机会约束、最大可行子系统、Bayes 0–1 决策的特定表述，绝不能仅靠这条等价式宣称创新。

## 7. 一个完全在 GMV 中成立的反直觉反例（建议作为 E1）

**目的：** 不依靠抽象一维玩具模型，构造三只股票、长仓 GMV、相同可观测边际收益，而 MA 在所有模型都不达标，兼容性方法在 `2/3` 模型质量上达标。

令 `d=3`，`w0=(1,1,1)/3`，记 `e_j` 为标准基，

\[
v_j=\sqrt{\frac{3}{2}}(e_j-w_0),\qquad
(H_j)_{aa}=0,\quad (H_j)_{ab}=3(v_{j,a}+v_{j,b})\ (a\ne b),
\]

\[
\Sigma_j(t)=I_3+tH_j,\qquad j=1,2,3.
\]

当 `t>0` 足够小时，各 `Σ_j(t)` 正定，且**对角线全部等于 1**。如果每一行只看一只股票收益，三个世界给出相同的 `N(0,1)` 观测数据。先验均匀时后验始终 `(1/3,1/3,1/3)`。

因为 `Σ_j H_j=0`，MA 使用的平均协方差恰好是 `I_3`，因此

\[
w_{MA}=w_0,\qquad R_j(w_{MA})=\frac12t^2+O(t^3).
\]

三种模型的一阶最优响应构成等半径正三角形；任意两模型的共同最小最大 regret 为

\[
\Gamma(\{i,j\})=\frac38t^2+O(t^3),\qquad
\Gamma(\{1,2,3\})=\frac12t^2+O(t^3).
\]

固定 `ε_t=7t²/16`。因为 `3/8<7/16<1/2`，在足够小的 `t` 下：

- `w_MA` 在 **0/3 个模型中达标**；
- 任意一对模型存在共同达标组合，因此方向 A 达到 **2/3 后验成功质量**；
- 没有任何持仓能同时满足三模型，所以 2/3 是精确上限。

独立小数值核验 `t=0.02` 时，`ε≈0.000175`，MA 每个模型 regret 约 `0.0002103`，一对模型的最优折中约 `0.0001615`，满足上述排序。**Codex 必须自行重新计算并通过数值 gate；这些数字不是新的独立实验结果。**

### 更有解释力的随机化比较

三个模型在 singleton masks 下观测等价。任意给定持仓最多使两个模型成功。如果我们**均匀随机选择三组“两模型兼容组合”**，在每个真实模型下，成功概率均为 `2/3`，失败概率均为 `1/3`。由“任何动作至少在一个模型失败”可得，对任意数据依赖随机算法

\[
\max_jP_j\{R_j(\widehat w)>\varepsilon_t\}\ge\frac13.
\]

这说明 **最坏 ε-失败概率下的随机化可能有价值**，虽然在凸 regret 的最坏**期望值**问题中随机化不会改善 Jensen 下界。两种损失目标绝不能混在一张性能表中。

这组结果用到了仓库原有的正单纯形几何构造和经典 Helly/凸分析，本身仍需与机会约束和多个正确答案文献排重。

## 8. DCL 候选算法 A1：Compatibility-Cut（有限模型精确解）

**输入**：候选风险场景 `Σ_j, μ_j`、后验质量 `π_j`、ε、长仓约束 `W`。每个 `V_j` 用原仓库的 QP oracle 计算，并记录原始/对偶误差。

**主问题**：使用二元变量 `z_j∈{0,1}` 表示该模型是否必须满足 ε，求

\[
\max_{z\in\{0,1\}^M}\sum_j\pi_jz_j
\quad\text{s.t.}\quad \bigcap_{j:z_j=1}K_j(\varepsilon)\ne\varnothing.
\]

**建议实现：冲突割平面（constraint generation）**

```text
compatibility_cut(scenarios, posterior, epsilon):
    QP oracle => each V_j and numerical lower/upper bounds
    initialize MILP master: maximize sum(pi[j]*z[j]), z binary
    repeat:
        solve MILP master => selected set G
        solve convex feasibility QCQP:
            find w in W, s >= 0
            subject to R_j(w) <= epsilon + s for j in G
            minimize s
        if feasible with verified tolerance:
            among feasible witnesses optionally tie-break by posterior mean regret
            return w, posterior_good_mass, MILP_bound, QCQP_gap
        else:
            find a small incompatible subset C of G
            add conflict cut sum(z[j] for j in C) <= len(C)-1
    until time/gap budget
    report best feasible solution and global master upper bound
```

**数学依据**：`K_j(ε)` 是凸集；在 `k=dim aff(W)` 的决策空间内，任一空交的有限族存在至多 `k+1` 个集合的空交见证（Helly）。因此可以检查、缓存较小冲突集。**但这不保证整个离散子集选择多项式时间**；在 20 资产、256 情景上直接求精确 MILP 不应当是首轮目标。

**务必防止数值错误**：

1. 当 `Γ(G)` 与 ε 的差小于数值容差，标记 `indeterminate`，提高精度，而不是强行判可行/不可行。
2. 不能把 approximate `V_j` 当精确值。为了**验证达标**，应对 `f_j(w)-V_j` 使用 oracle **下界** `V_{j,L}`：`f_j(w)-V_{j,L}` 才是 regret 上界。
3. 当 `G` 兼容，最终输出持仓可对后验质量达到最优；不保证平均 regret 领先 MA。失去后验正确性时也无后验保证。
4. 对相同场景复制并平分质量，A1 的理论最优成功质量应保持不变；实现必须测试这个不变量。

### A2：scalable surrogate（作为 A1 的对照而非“新方法”）

为了扩大到 `M=128/256`，预热一个凸代理：

\[
\widehat w_{\rm hinge}\in\arg\min_{w\in W}\sum_j\pi_j[R_j(w)-\varepsilon]_+.
\tag{A3}
\]

它不是 A1 的等价改写，但有明确的缓冲界：对任意 `γ>0`，

\[
P_{\pi}\{R_j(w)>\varepsilon+\gamma\}
\le\frac{\sum_j\pi_j[R_j(w)-\varepsilon]_+}{\gamma}.
\]

在 E1/E2 同时比较 A1 精确解、A3 代理和 MA，检查可计算性与 ε-成功率差距。若 A3 只能接近 MA、不能恢复兼容性优势，先停止推广，不需要不断加新 surrogate。

## 9. DCL 的候选主理论（尚未完成）

**T-A1：Decision-compatibility information complexity。** 对指定的观察设计 `S_r`、模型族和 `ε`，给出达到

\[
\sup_\theta P_\theta\{R_\theta(\widehat w)>\varepsilon\}\le\delta
\]

的必要且（在一类非平凡多模型族上）可达的共同观察量。目标是由**冲突模型组**而不只是参数点识别复杂度刻画。需要考虑：所有模型对可能兼容，但整体不兼容；简单 pairwise KL 必然遗漏某些多模型障碍。

已可得到的起点：当有限群 `J` 的观测分布完全一致且 `∩_{j∈J}K_j(ε)=∅`，则所有规则有

\[
\max_{j\in J}P_j\{R_j(\widehat w)>\varepsilon\}\ge 1/|J|.
\]

近似相同时，选一个参考分布 `Q`，还可以用总变差距离得到

\[
\sum_{j\in J}P_j\{R_j(\widehat w)>\varepsilon\}
\ge 1-\sum_{j\in J}{\rm TV}(P_j,Q).
\]

这只是初步不可区分下界。**要成为 ICML 主定理，必须加入多模型信息量结构、统一可达算法或紧的概率界；不能将上述几行简单不等式冒充原创贡献。**

**T-A2：计算—统计误差分离。** 若候选模型后验为 `π`，算法使用 `\hatπ`，则对任一固定组合，后验 ε-成功概率误差不超过 `TV(π,\hatπ)`；使用 `\hatπ` 的最优组合相对真实后验最优成功值，差最多 `2 TV`。这是现成的有限质量扰动事实；要体现 novelty，需要考虑连续模型、在 `R=ε` 边界不连续时的正确逼近与求解复杂度。

**T-A3：随机化/凸 regret 的目标分离。** 在三模型 GMV 例中，证明“MA 最优平均 regret，但 ε-成功率 0；随机 pair-compatible 决策达到最小最坏失败概率 `1/3`”。这适合正文的反直觉 Proposition，不足以单独构成论文全部理论。

---

# Part IV. 方向 B：Evidence-Gated Portfolio Learning (EGPL)

## 10. 最小理论模型：为什么需要一个“证据闸门”？

考虑持仓增量 `x≥0`（多资产某个可行方向的局部标量化）。两个市场：

\[
f_{G,t}(x)=\frac h2x^2-a t x,\qquad
f_{B,t}(x)=\frac h2x^2+s x,
\quad a,s,h>0.
\tag{B1}
\]

`G` 希望买入 `x_G^*=at/h`，`B` 希望不买入 `x_B^*=0`。

若令坏模型后验概率为 `p_B`，**正确后验下的 MA 决策本身就是**

\[
\boxed{x_{\rm MA}=\frac{[(1-p_B)at-p_Bs]_+}{h}.}\tag{B2}
\]

MA 的买入条件是 `p_B/(1-p_B)<at/s`。二模型后验赔率为

\[
\frac{p_B}{1-p_B}
=\frac{b_B}{b_G}\exp(\ell_B(D)-\ell_G(D)).
\]

因此 MA 的买入门槛等价于

\[
\boxed{\ell_G(D)-\ell_B(D)>\log\!\frac{b_Bs}{b_Gat}.}\tag{B3}
\]

**重要反查：** 若“新算法”只是检查 (B3) 再使用 (B2)，那它就是 MA 的另一种实现，**没有算法创新**。下一阶段必须指定区别于 Bayes 均值的评价准则，例如**错误解锁概率、约束安全性、先验不确定下的校准保证**，并证明新增能力。

## 11. EGPL 候选算法 B1：双模型一侧似然闸门（严格、可实现的起点）

给定独立、外生 mask 的完整观察数据，记全数据似然比

\[
\Lambda(D)=\frac{p_G(D)}{p_B(D)}.
\]

设容错常数 `δ∈(0,1)`，预先确定 `t`。定义

\[
\boxed{
\widehat x(D)=
\begin{cases}
 at/h,&\Lambda(D)\ge (\delta t)^{-1},\\
 0,&\text{otherwise}.
\end{cases}}
\tag{B4}
\]

**命题 B0（可直接证明；不是新的似然检验）：** 若真市场为 B，

\[
P_B\{\widehat x=at/h\}
\le\delta t.
\]

因为在 B 下 `E_B[Λ]=1`，用 Markov 不等式即可。于是坏市场误入场的 regret 是 `O(t)`，乘以错误批准概率 `O(δt)` 后，得到 `O(δt²)` 的期望 regret。

另一方面，在 G 下，需要 `P_G{\Lambda<(\delta t)^{-1}}=o(1)` 才能达到 `o(t²)` 的期望 regret。若联合观测每条提供固定正的 KL 和合适集中条件，那么 `m=C log(1/t)` 级别的联合观察可使该错拒概率消失；仓库已有多资产嵌入定理覆盖了类似的匹配**阶**结论。

**解释：** B4 和 MA 的目标不同。B4 优先保证“真为 B 时不应错误买入”的一侧错误概率；它未必比 MA 有更小的平均 regret。它只是后续**多方向证据校准**的、可审计的统计原型。

## 12. EGPL 候选算法 B2：多资产、多个持仓方向的复合零假设闸门

一份数据上可能产生多个候选持仓，例如不同股票进入、退出或一组股票重新分配。令 `w_b` 为在开发样本上确定的基准持仓，`w_c` 为第 `c` 个候选组合；它们须满足共同可行域。

为了保证“批准候选不使真实目标比基准恶化超过 `ξ_c`”，定义**危险模型集合**：

\[
\mathcal B_c
=\{j:f_j(w_c)-f_j(w_b)>\xi_c\}.
\tag{B5}
\]

为候选 `c` 选择在**评估样本之前确定**的辅助概率密度 `q_c(D_{\rm test})`，例如支持改善模型的固定混合似然。批准条件是

\[
\boxed{
\min_{j\in\mathcal B_c}
\log\frac{q_c(D_{\rm test})}{p_j(D_{\rm test})}
\ge \log\frac{1}{\delta_c t}.
}\tag{B6}
\]

若危险集合为空，则该候选不需要统计闸门。对任一真实危险模型 `j∈B_c`，由 `E_j[q_c/p_j]=1`，有

\[
P_j\{\text{approve candidate }c\}\le\delta_c t.
\tag{B7}
\]

若预先固定 `C` 个候选且 `\sum_c\delta_c\le\delta`，用 union bound 可以同时控制多候选错误批准。选出批准候选后可按后验均值目标二次优化或在其凸包内优化；若所有批准候选对真实模型都是 `ξ`-安全的，**共同凸目标下它们的凸组合也不比基准恶化超过 `ξ`**。注意不同基准、不同惩罚 `g` 时必须重新检验这条凸性转移。

### 数据复用的硬性要求

- 若 `w_b,w_c,B_c,q_c` 均使用同一 `D_test` 选择，B6 的条件有效性**一般不成立**。首版必须使用 **pilot/fit 与 gate-test 两份独立数据**，或证明一个允许数据自适应选择的有效 e-process。
- 不能将 `q_c` 随意替换为以整个 `D_test` 拟合的“最好模型密度”，再继续使用 `E_j(q_c/p_j)=1`。
- 多次时间滚动使用同一门槛不自动控制整个交易期错误率；要另加顺序校准或错误预算分配。
- 在 G3 非平稳、收益相关缺失、时间依赖条件下，B7 **不能直接沿用**。初期只在严格符合条件的独立高斯实验上检验。

### B2 与经典 robust safe improvement 的区别在哪里？

B6 本身属于似然比/e-value 和安全候选批准的经典构造；新意不在这个统计检验。**有待贡献的部分**是如何利用 `a_j(v)`、临界锥、隐藏 Hessian 和模型兼容性，**自动提出只针对真正冲突方向的候选**，并证明多方向 gate 的样本量接近信息下界，而不是把所有风险参数都精确估计后才行动。

## 13. EGPL 候选主定理（下一阶段应力争完成）

**T-B1：方向相关的信息释放界。** 对固定参考组合 `w0` 和一组预算可行的临界方向 `v_1,...,v_L`，将阻碍方向 `v_l` 的模型集合定义为 `B(v_l)`，构造以完整 mask 联合观察计算的证据检验。证明一个有明确误批概率控制、可输出合法多资产持仓的统一算法。

**T-B2：匹配下界的 sample complexity。** 对真实模型 `θ`，若要求沿其正确方向达到 `o(t²)` regret，而在相冲突模型中只有 `O(t²)` regret，可将算法输出转化为区分观测分布的检验，得到信息下界。目标是在**不止两个已知模型、允许多个兼容答案**的非平凡子类中证明必要量与算法所用共同观察量匹配；当前仓库的两模型 Theorem 6 已给出一个可引用起点。

**T-B3：与 MA/CVaR 的严格分离。** 预热必须先寻找在**同一明确目标**（如指定错误批准率或最坏 ε-失败概率）下，EGPL 比 Bayes 均值或固定 CVaR 有理论上不能轻易复现的性质。不能拿 EGPL 的一侧风险与 MA 的平均 regret 横向比较来声称全面更强。若没有这样一个分离定理，B 应被定位为理论实验工具或安全对照。

---

# Part V. 如何让 A 和 B 汇聚，而不是同时做两篇松散文章？

## 14. 一个可能的统一接口：兼容性优先，证据用于排除关键冲突模型

方向 A 告诉我们：给定 ε，哪些模型可以共享好决策，哪些不能。方向 B 告诉我们：哪些观察证据能排除阻碍某个候选好决策的模型。

在有限候选集上可以形成下面的研究接口：

1. 用 A 产生一个候选 **兼容集合 `G`**，并找出共同持仓 `w_G`。
2. 对该候选持仓，构造**不兼容的反例模型**：`B_G={j:R_j(w_G)>ε}`。这是模型/候选依赖的危险集合；若 `w_G` 是用数据选出的，则下一步使用独立 gate 样本。
3. 在独立数据上，用 B 的似然证据检验是否已经足以排除这些危险模型。通过则发布 `w_G`；否则回到更保守的基准，或寻找另一个兼容集合。
4. 目标定理：区分产生冲突的模型所需信息可以小于恢复完整协方差所需信息；该算法达到相应近优成功/误批保证。

**注意**：这是候选整合路径，不是已经证明在高维连续风险族上有效的完整算法。其计算复杂度、重复检验预算和独立样本成本都需要实际测量。当前优先把 A 与 B **分别在小问题上验证**；只有都有清楚的非平凡结果才合并。

---

# Part VI. 给 Codex 的直接执行协议（先小实验，严禁先跑大回测）

## 15. 实验工程约束

- **基线仓库：** 保留 `research_v2/` 的所有冻结结果、确认种子及历史诊断。新代码放 `research_v3/`，除非明确复制，不重用已查看的 `confirm` 作为新的独立确认。
- **信息隔离：** 方法只接收 `(masked returns, fixed candidate risk family/prior, feasible set, ε/δ)`。真实 `θ`、未来收益/真协方差仅由 evaluator 持有。
- **共享底层：** 复用 `core.MaskLikelihood`、QP oracle / Clarabel；为新算法新增有限场景 posterior 模块、`Γ(G)` 可行性 oracle、MILP master、e-value gate。保存每一个原始求解 gap。
- **核心对照：** MA (`A-Mean`)、posterior CVaR (`A-Tail-P:0.5`)、full-set regret、合适的 MLE+QP；对 A 添加精确 chance oracle、hinge proxy；对 B 添加已知模型的 LR test 和 posterior-mean boundary action。
- **固定小规模：** 初期 `d=3,6,10,20`，离散 `M=2,3,6,10,12`。`M=128/256` 只对能扩展的代理方法测试，不强求精确整数求解。
- **数值规则：** 每次记录 primal/dual gap、可行残差、模型置信/后验误差及求解时间。`Γ(G)` 距 ε 在容差范围内时标注不可判定。启发式输出没有全局证书就不能写“exact”。
- **性能解释：** CPU 是此类矩阵统计/小型 QCQP 的自然运算环境；不要引入 NN 仅为制造对照。

## 16. Gate 0：代码正确性（先于一切统计比较）

| 检查 | 必须达到的条件 |
|---|---|
| 后验计算 | singleton 观测相容模型的 log-likelihood 完全相同；后验=先验（数值精度内） |
| 场景复制 | 复制同一模型并平分质量，A 的最优成功概率及加权尾部结果不变 |
| QCQP oracle | `Γ(G)` 的独立锥规划 / 二次规划核验，约束和 KKT 残差合格 |
| A1 一致性 | 三模型、每个子集暴力枚举结果与 MILP+冲突割一致 |
| A3 代理 | 检查 buffered hinge 概率上界的数值方向正确 |
| B4 显著性 | 固定错误市场下假阳性频率不大于 `δt`（有限重复只能做置信区间检验） |
| B6 数据隔离 | 将 gate 数据置换为新随机种子，不改变 pilot 候选库；真参数不进入算法接口 |
| 数值稳定 | 不允许用 solver `success=true` 替代真实 gap 检查；模糊 case 不强行 pass |

## 17. 实验 E1：三市场 ε-兼容反例（理论机制测试）

**目标**：精确验证方向 A 确实与平均 regret 有不同答案；不是为市场泛化造一个“新算法胜利”。

**生成**：采用第 7 节三资产协方差族，先 `t∈{0.005,0.01,0.02}`，`W=Δ_3`，三模型均匀先验。收益 `N(0,Σ_j(t))`，最初只给 singleton masks，故观测不可区分。

**阈值**：预先定 `ε_t=7t²/16`。测试 MA、A1、A3、max-regret、随机 pair center；对已知参数族还要暴力枚举 7 个非空子集验证 `Γ(G)`。

**应该观察到（解析预期，需重新运行）**：MA 的 ε-成功模型质量为 0，A1 为 2/3；MA 的后验平均 regret 不劣于 A1；随机 pair centers 在每个真实模型下成功概率为 2/3；三个观测分布提供相同信息，所以**任何方法的统计表现不应随 singleton 样本数量变化**。

**对照扩展**：加入真正联合观察后，用完整 mask 似然更新三模型后验。不要用“固定 `m`”简单比较不同 `t`：本模型两世界差异也随 `t` 缩小，单条 joint 记录 KL 约为二阶小量。应同时报告 `m×KL`；按预先设定的信息预算（例如 0、1、3、6 个 nats）选择 `m`，在预算过高时限制最大行数并报告未达到目标。

**停止**：若 `m=0` 的模型后验发生变化、A1 与穷举不一致或 MA 在 `t≤0.02` 与解析预期冲突，先修代码，不进行后续实验。

## 18. 实验 E2：多模型兼容集合与计算可行性

**目标**：判断 A 是否具有真正的高维/多模型价值，并确认“只有 pairwise 检查”会失败。

- `d∈{3,6,10}`，`M∈{3,6,10,12}`，保持模型族事先确定、协方差严格正定；先使用仓库中的 regular-simplex 和 hidden-curvature 构造，接着使用非对称扰动。
- 对每个模型预先计算独立 `w_j^*`；ε 网格用 `Γ(all)` 的事前理论量纲构造，例如 `ε/Γ(all)∈{0.25,0.5,0.75,1.0}`，且所有候选算法共同使用同一 ε。
- 比较：A1 精确 chance、A3 hinge、MA、posterior CVaR、minimax；同时记录 pairwise-compatible 比例、最小不可行冲突集大小、MILP 节点数及运行时间。
- 分开报告 `posterior_success_mass`（条件于单份观测）、`true_success_rate`（真市场跨数据重复）、平均 regret 及 p95 regret。
- 限制 `M≤12` 的精确 A1 单实例求解时间，例如 30—60 秒；超过就记录 `timeout + 上下界`，不能删除慢或失败样本。

**应当接受的负结果**：A1 只在 ε-成功指标有优势而在均值 regret 更差，是符合目标定义的现象；若一般非对称场景里 A1 都退化为 MA 或求解基本超时，应收缩 A 的算法贡献，转向更有结构的子类或停止。

## 19. 实验 E3：双市场“证据解锁”信息相变

**目标**：用与仓库 Theorem 6 完全一致的多资产风险生成模型，检验 B4 的错误解锁率与共同样本量需求。

**生成**：直接复用 `research_v2/embedded_boundary.py` 的 `d=10/20`，保持组内独立历史与跨组联合历史分离，`t∈{0.1,0.03,0.01}`；同一协方差必须用于收益生成和真实 GMV regret。

**方法**：B4（二模型固定阈值，`δ=0.05`）、MA、后验 CVaR `ρ=0.5`、完全信息 oracle（仅上限参照）、始终不交易；如适当加入 LCRC/稳健中心对照。

**横轴**：用 `m K(t)/log(1/t)` 与 `m` 同时表示；`m` 应预先取围绕阈值的一系列值。数据生成仅模拟模型本身，决策方法不可读取生成标签。

**指标**：

1. 坏市场 `false_unlock=P_B(选择买入)`、目标线 `δt`；
2. 好市场 `false_reject=P_G(拒绝买入)`；
3. `E_G R_G /t²` 和 `E_B R_B /t²`（分别计算，不能简单混在均值）；
4. 需要的 joint observations、平均/最大运行时长；
5. 检查 `m=0` 时组内样本加大仍然无法积累区分证据。

**模拟精度**：第一轮每个 `(t,m,true_model)` 至少 200 重复作绘图；对于小于 1% 的错误率不能把 0 次观测到错误直接当严格证明，应给二项区间并在需要时增加重复。理论保证来自似然比，而不是模拟 95% 的经验覆盖率。

**停止**：若 B4 与解析两模型构造不一致，或者 log-likelihood ratio 在 m=0 时产生虚假的信息积累，先修数据或似然实现。若 B4 仅表现为 MA 的一侧重新阈值化，且没有可推广的多方向增量，就不要将 B4 单独当作“新方法”。

## 20. 实验 E4：多方向综合可行性（只在 E1—E3 通过后）

- `d=10/20`，设计 3—6 个候选调整，分别涉及进入、退出、接近上限及跨组再配置；至少一组方向共享预算，不能把所有方向假设为彼此独立。
- pilot 样本生成候选 `w_c` 和危险模型 `B_c`；另取独立 gate 样本进行 B6。`δ_c` 分配预先固定，总和不超过 δ。
- 比较：EGPL-B2、多模型 A1（小 M）、MA、A3、后验 CVaR、保守基准。
- 输出：逐方向证据、批准记录、组合权重、交易约束可行性、错误批准率、真模型 regret。分析错误批准不是由数据复用、重复 gate 或错误 mask 造成。
- 优先在 `M≤10` 情景进行；如果执行安全与概率控制无法通过，停止，不推进 G2/G3。

## 21. E5：才考虑 G2/G3、以及后续 A 股

E1—E4 只测试正确规定的有限模型与观测关系，不能自动外推到非平稳市场。在明确得到一个**新评价目标下的有效机制 + 求解可靠 + 与强基线的可解释区别**之后，才扩大：

- **G2**：20 资产两组，`n_B∈{64,256}`，`m∈{0,4,16,64}`，同标量预算；不能重复使用原 `confirm` 做新独立确认。须另开未查看的新风险族与种子。
- **G3**：20/50 资产生成器未知，非平稳/厚尾与 informative mask 只是压力测试；原 G3 ESS 几乎坍缩至 1，必须先处理场景质量校准，否则不同聚合器的比较没有判别力。
- **A 股**：完成新合成独立确认后再进行滚动样本外风险验证。过去 2011—2026 日频数据的训练、持仓、未来标签、调参完全隔离；检查上市、停牌、除权、退市、交易可行性、换手和交易成本。真实数据没有可见总体协方差 oracle，不能报告合成意义的 true population regret。

---

# Part VII. 建议 Codex 产出与选择标准

## 22. 建议生成的代码/结果文件

```text
research_v3/
  README.md                         # 如何复现，准确说明哪些已执行
  background_formalization.md       # 从本文提炼的研究定位（可复用）
  compat_models.py                  # Σ_j 构造、观测设计、后验、oracle
  compatible_qcqp.py                # Gamma(G)、冲突搜索、数值上下界
  dcl_master.py                     # A1 master MILP、cut log
  dcl_surrogate.py                  # A3 与 buffered bound
  evidence_gate.py                  # B4/B6, likelihood ratios, safe split
  run_gates.py                      # Gate 0 全部检查
  run_E1.py                         # ε-兼容反例
  run_E2.py                         # 多模型冲突与计算复杂度
  run_E3.py                         # 双市场样本复杂度
  run_E4.py                         # pilot/gate 多方向（有条件启动）
  configs/                           # seed、epsilon、delta、模型族、mask
  results/                           # csv/json、图、逐实例错误/超时
  THEORY_STATUS.md                   # proved / conjecture / failed
  RESEARCH_REPORT_ZH.md              # 负面结果与主结论
```

这些是**待 Codex 创建的文件规格，不表示当前仓库已经存在这些路径。** 禁止覆盖 v2 原始 CSV/JSONL。

## 23. Codex 的工作顺序与停机线

**第一阶段（先 1—2 个短 CPU 会话）：** 实现 Gate 0 + E1，核对三模型 GMV 反例；不得修改生成器以“确保新算法胜利”。如果数值不能重现实质性分离，暂停并检查数学/代码。

**第二阶段：** E2 的小场景精确兼容求解器及与 MA/CVaR 的公平比较。关注 `posterior ε-success` 的确切提升、MILP 时间和数值正确性，不要优先追求 Sharpe。

**第三阶段：** E3 双市场 evidence gate。冻结阈值、生成器和错误率，检验 `m KL / log(1/t)` 及一侧风险。此阶段只确认机制，不能宣称新的样本复杂度定理已经超越仓库 Theorem 6。

**第四阶段（仅在前述基础上）：** E4 多方向 gate 与 A/B 结合。若始终无法给出 MA/CVaR 没有的新性质，**正式停止新算法扩展并重新评估问题**。

**明确禁止：** 只在一个特设样本上汇报 ε-success；用真实生成参数构造算法输入；把已看过的旧测试称为独立确认；忽略场景近似误差；依据平均 regret 胜负否定一个专门优化 δ-失败率的方法；依据 δ-失败率优势宣称平均 regret SOTA；为了发表增加复杂但缺少实质增量的算法模块。

## 24. 论文选择的预定 Gate

| 检查 | 选择 A 的理由 | 选择 B 的理由 | 都不能满足时 |
|---|---|---|---|
| 理论增量 | 多模型兼容/信息复杂度定理超越现有 chance & best-answer 文献 | 多方向证据门槛/未知干扰下的有效性与可达性 | 停止包装新方法 |
| 算法增量 | 可扩展的兼容冲突求解/一致代理，且不仅是通用 MILP 包装 | 决策相关 gate 有可证明的非平凡保障，不只是 MA 阈值 | 只保留机制论文或换题 |
| 合成实证 | E1 反例通过，E2 在未选择机制上有可靠表现 | E3 概率校准通过，E4 在冲突方向仍有效 | 不启动真实回测 |
| 数值和统计 | 小情景精确 gap、概率计算稳定、独立确认 | 信息样本需求、FP/FR 有足够重复，训练与 gate 分离 | 先修基础设施 |

---

# Part VIII. 最近邻文献与审稿风险（直接阅读链接）

下列是**已存在的理论/算法范式**。阅读目的是辨别可继承工具和必须超越的内容，不等于它们支持了我们全部候选主定理。

1. **Lee, Jin & Lee, “Decision-Focused Learning via Tangent-Space Projection of Prediction Error,” ICML 2026.** [PMLR 原文](https://proceedings.mlr.press/v306/lee26q.html)。已知局部目标结构下的 regret 投影及梯度计算；我们的隐藏曲率、模型不确定性和统计信息区别于其固定-base 局部结构。
2. **Degenne & Koolen, “Pure Exploration with Multiple Correct Answers,” NeurIPS 2019.** [官方页面](https://papers.nips.cc/paper/2019/hash/60cb558c40e4f18479664069d9642d5a-Abstract.html)。多个正确答案改变信息复杂度；**方向 A 最重要的近邻**，不能把“多个好答案”当作新思想。
3. **Christensen, Moon & Schorfheide, “Optimal Decision Rules When Payoffs are Partially Identified,” Review of Economic Studies 2026.** [期刊页](https://academic.oup.com/restud/advance-article-abstract/doi/10.1093/restud/rdag017/8502914)。可识别参数的平均统计风险与未识别回报的 minimax；我们的连续组合动作和兼容性/异质临界锥必须提供新结果。
4. **Hauser, Krishnamurthy & Tütüncü, “Relative Robust Portfolio Optimization.”** [预印本](https://arxiv.org/abs/1305.0144)。有限场景相对 regret 及稳健优化已有前例；不能重新命名其 regret center。
5. **Wu, Zhu & Zhou, “A Bayesian Risk Approach to Data-driven Stochastic Optimization,” SIAM Journal on Optimization.** [期刊页](https://epubs.siam.org/doi/10.1137/16M1101933)。后验均值、VaR/CVaR 风险优化已有理论；A 不可声称 Bayes chance 是新目标，B 不可声称后验风险加权是新算法。
6. **Wasserman, Ramdas & Balakrishnan, “Universal Inference,” PNAS 2020.** [预印本](https://arxiv.org/abs/1912.11436)。似然比/e-value 校准为经典工具；B 的 B4/B6 不是原创统计检验。
7. **Pagnoncelli, Reich & Campi, “Risk-Return Trade-off with the Scenario Approach in Practice: A Case Study in Portfolio Selection,” JOTA 2012.** [论文页](https://ideas.repec.org/a/spr/joptap/v155y2012i2d10.1007_s10957-012-0074-x.html)。用删除约束在投资组合中处理 chance constraints；是 A 的**直接优化近邻**。
8. **Elmachtoub & Grigas, “Smart ‘Predict, then Optimize’,” Management Science 2022.** [原文](https://pubsonline.informs.org/doi/10.1287/mnsc.2020.3922)。在**线性目标**下设计 SPO+ surrogate 并证明一致性；不能把 SPO+ 当成未知 Hessian 的 GMV 的直接同任务强基线。
9. **Rockafellar & Uryasev, “Optimization of Conditional Value-at-Risk.”** [论文 PDF](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf)。CVaR hinge/阈值与对偶风险包络为成熟工具。

**待专门排重：** 最大权重的可行子系统与 chance-constrained integer optimization；多答案 pure exploration 的信息博弈；局部强凸随机规划、二阶 epi calculus；安全策略改善和 e-process 的自适应检验。没有完成这四项逐定理核查之前，不能宣称“完全创新”。

---

## 最终建议

将本文作为 **“先证明决策兼容性的信息结构，再尝试用证据闸门达到该信息要求”** 的预热路线。**主论文问题优先 A，第一项可精确理论实验 E1；算法可行性优先 B，第一项统计机制实验 E3。** 其余 E2/E4 按 Gate 结果启动。

真正值得投稿的成果组合应是：

\[
\boxed{\text{多模型决策信息下界}
\; + \;\text{可达的持仓选择/释放算法}
\; + \;\text{独立 CPU 实证机制验证}.}
\]

若最终只有“Bayes chance objective + generic MILP”或“LR test + MA 的买入门槛”，应当如实判定理论—算法创新不足并调整方向。ICML 的可信贡献来自**真实的新问题、新界和新可达算法**，而不是证明复杂或参数搜索庞大。
