"""分块重排显著性检验：回答"这个回测结果是不是运气"。

本模块与 `robust_metrics.py` 的分工：那边全是【样本内描述统计】（RAR / R³ / Robust
Sharpe 都只是把曲线形状换个角度描述一遍），本模块是【统计检验】——它给出一个 p 值，
即"在策略毫无预测力的零假设下，纯靠运气拿到不低于实测值的成绩的概率"。

━━━ 一、重排什么？三个零假设是三个不同的问题 ━━━

这是全模块最重要的一段。"重排收益序列"和"重排价格序列后重跑策略"回答的**不是同一个
问题**，而且对 long-only 单标的 CTA 来说，两个常见做法各有一个致命缺陷：

  方案 A · 重排【策略日收益序列】
      H0：策略日收益可交换（顺序无信息）。
      致命缺陷：**夏普对收益序列的重排（近似）恒等不变**。分块重排只是把同一批日盈亏
      换个顺序，多重集合一个元素都没变；若按固定本金算算术收益，均值与标准差严格不变，
      夏普是数学恒等式意义上的不变量，零分布退化成一个点，功效为零。

      这里有个必须说清的实测细节：vnpy 的夏普不是算术口径，它走
      `log(balance_t / balance_{t-1})`，分母是滚动净值，于是重排会通过复利路径带来
      一点二阶残差 —— 零分布**不是**一个点，而是一个极窄的分布。这比严格不变更危险：
      它会安安静静吐出一个看起来正常的 p 值，而那个 p 值衡量的纯粹是复利记账顺序，
      与有没有 edge 毫无关系。

      ⚠ 残差大小 **随仓位占本金的比例增长**，这一点曾经害本模块漏判过。实测
      （n=500，5 个种子，"零分布std / |观测值|" 探针，阈值 5%）：
          仓位/本金   5%     10%    25%    50%    75%    100%
          比值      0.58%  1.14%  2.77%  5.32%  7.74%  10.10%
          探针触发   5/5    5/5    5/5    1/5    0/5    0/5
      **满仓时探针完全失灵**，而港美股现货 long-only 的常态就是满仓。
      另一个候选判据"重排零分布std / 自助零分布std"也试过，退化组最大 7.6%
      与有功效组最小 5.9% 直接重叠，同样分不开。
      ⇒ 结论：**任何基于幅度的探针都不可靠**。本模块因此改用 `_MULTISET_INVARIANT`
      静态表做主判据 —— 哪些统计量在严格置换下恒等不变是**解析事实**，
      与数据、仓位、种子全都无关，判定结果写进 `PermutationResult.degenerate`
      （机器可读，下游别再去 warnings 里抓关键字）。幅度探针降级为
      自定义统计量的兜底。

      `RETURNS_BLOCK` 因此只允许配路径依赖统计量（最大回撤 / 收益回撤比 / RAR / R³）。
      要用它检验"均值是否为零"，得改用 `RETURNS_BOOTSTRAP_H0`（去均值后平稳自助），
      那是自助检验不是重排检验，且对 long-only 策略仍有下面 C 说的 beta 污染问题。

  方案 B · 重排【价格序列】后重跑策略（Masters MCPT）
      H0：价格序列除了无条件收益分布之外没有可利用的结构。
      优点：把参数寻优整个塞进重跑回调里，就能同时校正选择偏差（见第五节）。
      缺点：贵（B 次完整回测），且它把"这个市场有趋势"和"我的规则抓住了趋势"混在
      一起算作 edge。对 long-only 品种，一个恒定满仓的策略在牛市里也能"显著"。

  方案 C · 重排【持仓/暴露序列】，市场收益原地不动  ←── 本模块默认，也是我的选择
      H0：策略的持仓时点与其后的市场收益无关。
      零假设下保留的：整条市场路径（趋势、波动聚集、暴涨暴跌全部原样）、暴露序列的
      全部取值（在场时间占比、仓位大小分布）、暴露的持续性（块内的持仓段完整保留）。
      被破坏的：只有【对齐关系】—— 什么时候在场。

  为什么选 C：项目实盘是【单标的 long-only 海龟】。这类策略的收益天然含两块，
  一块是"在场就有的市场 beta"，一块是"择时"。方案 B 的零分布是"没有趋势的市场"，
  于是 beta 那块会被整个算进 edge 里 —— 港美股 2024-2026 是上行市，任何长期在场的
  规则都会显著，这正是本项目栽过的跟头（样本外 -19.5% 而事前一切指标都好看）。
  方案 C 的零分布是"同一个市场 + 同样的在场时间 + 随机的择时"，它把 beta 从两边同时
  约掉，剩下的才是择时技术。**对 long-only 策略，这是唯一能把运气和技术分开的重排。**

  代价要说清楚：方案 C 不检验"这个策略能不能赚钱"（在牛市里满仓当然能），它检验
  "这个策略的择时比随机择时好"。这两个问题都重要，但只有后者是可以外推的。
  想回答前者，请把 `POSITIONS_BLOCK` 的结果和买入持有基准并列读。

━━━ 二、为什么必须【分块】 ━━━

日收益有自相关（波动聚集尤其强），持仓序列的自相关更是极端 —— 海龟一拿几周，
暴露序列近似分段常数。iid 重排会把这些结构全打散，造出现实中不可能出现的路径，
零分布方差被系统性低估，p 值随之偏小（假阳性）。分块重排保留块内的依赖结构，
块长 L 越大保留得越多（也越保守）。

块长怎么选：默认走 **Politis & White (2004) 自动选块长**（`optimal_block_length`），
即用 flat-top 核估计谱密度及其导数，令 b_opt = (2·Ĝ²/D̂)^(1/3)·n^(1/3)。
关键实现选择：**块长在【被重排的那条序列】上估计**，不是在收益上。
`POSITIONS_*` 方案重排的是暴露序列，就在暴露序列上估；`RETURNS_*` 方案才在收益上估。
这不是细节 —— 暴露序列的自相关系数常在 0.9 以上，Politis-White 给出的块长会自动
落在"平均持仓周期"这个量级上，正好是它该在的地方；若误用收益序列估出来的 L≈2，
重排会把几周一次的持仓打成两天一换，零分布方差被严重低估。

`min_blocks` 参数强制块数下限（默认 8）：块数太少时重排的自由度不够，
零分布本身就不可信。触发时会在 `warnings` 里留痕并把块长压回去。

━━━ 三、600 日 / B=1000 的功效（模拟实测，不是套公式） ━━━

下表是实跑出来的：每格 200 次独立复现，n=600，POSITIONS_BLOCK + 年化夏普，
long-only 两状态持仓（约 50% 时间在场、平均持仓段 20 天），块长自动选。
复现脚本即 `tests/test_permutation_test.py` 里的 `build_edged_case` + 循环。

    每日 edge   实测夏普   零分布均值   零分布std   择时增量   功效@0.05   功效@0.01
    0.00000     0.04       0.02        0.451      0.02       0.06        0.01
    0.00050     0.41       0.21        0.452      0.20       0.13        0.04
    0.00100     0.77       0.40        0.454      0.38       0.20        0.07
    0.00160     1.20       0.62        0.457      0.59       0.39        0.14
    0.00220     1.63       0.84        0.462      0.79       0.53        0.26
    0.00300     2.18       1.12        0.470      1.06       0.74        0.47
    0.00400     2.83       1.46        0.482      1.37       0.93        0.71

必须读懂的四件事：

  1. **edge=0 那行是有效性证据**：α=0.05 拒绝 6%、α=0.01 拒绝 1%，
     第一类错误率被卡住了。方法本身站得住。

  2. **功效由"择时增量"决定，不是由策略夏普决定**。表里策略夏普 1.20 时功效只有
     0.39，而按夏普渐近公式（SE=√(252/600)=0.648，见 `sharpe_standard_error`）算出来
     是 0.58 —— 公式**系统性高估**，因为它检验的是"夏普 ≠ 0"这个松得多的问题。
     本检验的零分布中心是 0.62（随机择时也能拿到的成绩），策略真正的增量只有 0.59。
     换句话说：**渐近公式给的是乐观上界，别拿它当功效**。
     实测的门槛（增量 ÷ null_std = 2.49）：
         50% 功效需要择时增量 ≈ 1.645 × 0.46 ≈ **0.75 年化夏普**
         80% 功效需要择时增量 ≈ 2.49  × 0.46 ≈ **1.15 年化夏普**
     对照表里 edge=0.0022（增量 0.79）实测功效 0.53、edge=0.0030（增量 1.06）
     实测 0.74，与门槛吻合。`PermutationResult.min_detectable_effect` 就是
     2.49 × 实测 null_std，**该看它，不该看夏普公式**。

  3. **B 几乎不影响功效，只影响 p 值分辨率**。独立复现（n=600，edge=0.0030，
     200 次复现，另一套数据生成过程）：
         B=  99 → 功效 0.795（最小可达 p 0.0100）
         B= 199 → 功效 0.800（0.0050）
         B= 499 → 功效 0.820（0.0020）
         B= 999 → 功效 0.825（0.0010）
         B=1999 → 功效 0.830（0.0005）
     B 翻 20 倍只换来 3.5pp 功效。所以 B=1000 对 α=0.05 绰绰有余；
     真正的约束是分辨率：最小可达 p = 1/(B+1)，想给出 α=0.01 的结论 B 至少要 1000，
     想给 α=0.001 得 10000。

  4. **n 才是硬约束，而且是致命的**。同样的 edge 换成 n=134（对应项目那条 134 日曲线）：
         edge=0.0016（夏普 1.08）→ 功效 0.09
         edge=0.0030（夏普 2.05）→ 功效 0.24
         edge=0.0050（夏普 3.39）→ 功效 0.47
     null_std 从 0.46 涨到 0.99。**134 日样本连夏普 2 的策略都有 76% 概率漏掉**，
     那条 sharpe=-1.68 的曲线（SE≈1.37，连符号都没到 2σ）自然什么都不能证明——
     它既不能证明策略坏，也不能证明策略好。

  ⇒ 结论（务必写进每份报告）：600 日样本大约只能确认"择时增量 ≥ 1.15 年化夏普"
    这一档的 edge。真实 CTA 常见的 0.3-0.8 增量，本检验**看不见**。
    因此 **p > 0.05 不等于策略没用**，多半只是在说"600 天不够"。
    本检验的价值不对称，用对方向才有意义：
      · 它拦不住"漏掉好策略"（第二类错误率高得离谱）
      · 但它拦得住"把网格挑出来的噪音当 edge 送上实盘"——这正是本项目栽过的那次。

  ⚑ 独立复现（换一套数据生成过程：GARCH 波动聚集 + 两状态马尔可夫持仓，
    300 次复现/格，B=1000，块长自动选）。数字与上表略有出入（DGP 不同），
    但门槛结论完全一致：
        n=600   edge  0.0000 → 夏普 0.02 增量 -0.03 null_std 0.458 功效 0.03
                      0.0010 → 夏普 0.77 增量  0.41 null_std 0.456 功效 0.22
                      0.0022 → 夏普 1.77 增量  0.87 null_std 0.465 功效 0.60
                      0.0030 → 夏普 2.41 增量  1.15 null_std 0.480 功效 0.83
        n=134   edge  0.0000 → 功效 0.05（第一类错误仍被卡住）
                      0.0030 → 夏普 2.34 增量 1.08 null_std 0.954 功效 0.32
                      0.0040 → 夏普 2.99 增量 1.33 null_std 0.982 功效 0.36
    两点被独立确认：① edge=0 时拒绝率 0.03-0.05，方法有效；
    ② n=600 的 80% 功效门槛落在择时增量 ≈1.15 年化夏普，与上表一致；
    ③ n=134 时 null_std 翻倍到 ≈0.95，夏普 2.3 的策略仍有 68% 概率漏掉。

━━━ 四、小样本下的合法性 ━━━

重排检验的有效性**不依赖大样本渐近性**：在可交换性零假设下它是精确检验，
600 天也好 100 天也好，第一类错误率都被 α 卡死（这正是它优于 t 检验/夏普渐近
标准误的地方）。小样本伤的是**功效**，不是**水平**。
唯一的小样本前提是"块可交换"，块数太少时该前提本身变弱 —— 由 `min_blocks` 兜底。

━━━ 五、参数寻优里怎么用（重要，用错等于没用） ━━━

**不要**把 p 值直接当寻优目标函数。在网格上挑出 p 最小的那组参数，
挑选偏差原封不动地留在里面：跑 200 组参数，光靠运气就会有约 10 组 p<0.05。
正确做法二选一：

  1. 先按常规目标（sharpe_ratio 等）选出参数，**只对最终这一组**跑本检验
     （`permutation_statistics()`），当作上实盘前的一道闸。
  2. 要给"整个寻优流程"定 p 值，用 `selection_bias_corrected_test()`：
     每次重排价格序列后**把寻优重跑一遍**，比较"重排数据上最优参数的成绩"
     与"真实数据上最优参数的成绩"。这才是 Masters 的选择偏差校正版，
     代价是 B × 全网格次回测。

`gated_statistic()` 提供第 1 种用法的便利封装（p 不达标则把目标值压到 0）。
它在 `POSITIONS_*` 方案下够快（整个零分布是一次矩阵运算，B=1000 约 10ms），
但请注意上面那句话：**它不能替代第 2 种做法**。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from statistics import NormalDist
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

DEFAULT_PERMUTATIONS: int = 1000
DEFAULT_MIN_BLOCKS: int = 8
DEFAULT_ALPHA: float = 0.05
DEFAULT_POWER: float = 0.80

# 零分布标准差 / 效应量尺度 低于此比例 → 怀疑零分布退化。
#
# ⚠ 这只是**给自定义统计量兜底的启发式**，不是主判据。它按 |观测值| 归一，
# 因而随仓位规模漂移：同一个"夏普×RETURNS_BLOCK"组合，实测（本文件配套验证脚本，
# n=500，5 个种子）
#     仓位/本金   5%     10%    25%    50%    75%    100%
#     比值      0.58%  1.14%  2.77%  5.32%  7.74%  10.10%
#     是否触发   5/5    5/5    5/5    1/5    0/5    0/5
# 满仓（港美股现货 long-only 的常态）时它**完全不响**。曾试过换成
# "重排零分布std / 自助零分布std" 做归一，实测退化组最大 7.6% 与有功效组最小 5.9%
# 直接重叠，同样分不开。结论：任何基于幅度的探针都不可靠。
# 真正的判据是下面的 _MULTISET_INVARIANT 静态表——它是解析事实，与数据无关。
_DEGENERATE_NULL_RATIO: float = 0.05

# 在【严格置换】（RETURNS_BLOCK 重排 net_pnl 向量）下只依赖 net_pnl 多重集合、
# 因而恒等不变的统计量。这是解析结论，不是估出来的：
#   total_return   = f(Σ net_pnl)                     —— 严格不变（实测 null_std ≈ 2e-14）
#   annual_return  = total_return / n × annual_days   —— 严格不变
#   sharpe_ratio   = mean/std                          —— 算术口径下严格不变；
#       vnpy 走滚动净值的对数收益，重排只留下一点复利路径残差，
#       该残差衡量的是"记账顺序"而非 edge，同样不可用来做检验。
# 命中即判定本次检验没有功效，**与幅度无关、与仓位无关**。
_MULTISET_INVARIANT: frozenset[str] = frozenset(
    {"sharpe_ratio", "total_return", "annual_return"}
)

# 重构 P&L 与引擎 net_pnl 的【逐日】平均绝对偏差 / 逐日平均绝对盈亏，超过此值时告警。
# 真实引擎实测：日线策略在收盘价成交时约 0.4%（trading_pnl≈0），
# 次日开盘成交时会明显更大，那时应改用 exposure_column='end_pos'。
_RECONSTRUCTION_TOLERANCE: float = 0.10


# 检验关闭时并进 statistics dict 的占位值。语义是"没算"，不是"不显著"：
# p=1.0 / significant=False 让下游目标函数在检验关掉时退化成"没有证据"，
# 而不是 KeyError。engine 侧的 perm_fields 就是从这份拷贝出发的。
PERMUTATION_FIELD_DEFAULTS: dict[str, object] = {
    "perm_statistic": "not_computed",
    "perm_scheme": "not_computed",
    "perm_observed": 0.0,
    "perm_p_value": 1.0,
    "perm_z_score": 0.0,
    "perm_null_mean": 0.0,
    "perm_null_std": 0.0,
    "perm_block_length": 0,
    "perm_n_permutations": 0,
    "perm_min_detectable": 0.0,
    "perm_has_power": False,
    "perm_significant": False,
    "perm_warning_count": 0,
    "perm_error": "",
}

# BacktestingEngine.enable_permutation_test 接受的参数名白名单。
# 与 permutation_statistics 的签名一一对应，减去由 engine 自己填的
# daily_df / capital / annual_days —— 那三个不许外部覆盖。
PERMUTATION_SETTING_KEYS: frozenset[str] = frozenset(
    {
        "statistic",
        "n_permutations",
        "scheme",
        "size",
        "seed",
        "block_length",
        "exposure_column",
        "include_costs",
        "alternative",
        "min_blocks",
        "alpha",
        "power",
        "risk_free",
    }
)


__all__ = [
    "PERMUTATION_FIELD_DEFAULTS",
    "PERMUTATION_SETTING_KEYS",
    "BlockLengthResult",
    "PermutationResult",
    "PermutationScheme",
    "Statistic",
    "attach_permutation_statistics",
    "block_permutation_indices",
    "circular_block_indices",
    "empirical_p_value",
    "gated_statistic",
    "make_statistic",
    "minimum_detectable_sharpe",
    "optimal_block_length",
    "permutation_statistics",
    "permutation_test_bars",
    "permutation_test_positions",
    "permutation_test_returns",
    "permute_ohlc",
    "permute_price_series",
    "random_rotation_indices",
    "selection_bias_corrected_test",
    "sharpe_standard_error",
    "stationary_bootstrap_indices",
]


class PermutationScheme(str, Enum):
    """四种重排方案。含义与取舍见模块文档第一节。"""

    POSITIONS_BLOCK = "positions_block"
    """分块重排持仓/暴露序列，市场收益不动。单标的 CTA 的默认选择。"""

    POSITIONS_ROTATE = "positions_rotate"
    """整体循环平移暴露序列。完整保留持仓动态，但只有 n 种不同重排（p 值分辨率受限于 n）。"""

    RETURNS_BLOCK = "returns_block"
    """分块重排策略日收益。夏普对它恒等不变，只对路径依赖统计量有效。"""

    RETURNS_BOOTSTRAP_H0 = "returns_bootstrap_h0"
    """去均值后平稳自助。检验 H0: E[收益]=0；是自助检验不是重排检验。"""


# ══════════════════════════════════════════════════════════════════════
# 块长选择：Politis & White (2004)
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BlockLengthResult:
    """自动选块长的结果与中间量（中间量用于判断这个块长可不可信）。"""

    stationary: int
    """平稳自助的最优平均块长 b_opt。"""

    circular: int
    """循环分块自助 / 分块重排的最优块长。"""

    mhat: int
    """自相关截断点估计 m̂。"""

    bandwidth: int
    """flat-top 核带宽 M = min(2·m̂, m_max)。"""

    lag1_autocorrelation: float
    """一阶自相关，用于人工核对块长是否合理（越接近 1 块长应越大）。"""

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _flat_top_kernel(t: FloatArray) -> FloatArray:
    """Politis & White 的梯形 flat-top 核 λ(t)。

    |t| ≤ 1/2 时为 1；1/2 < |t| ≤ 1 时线性降到 0；|t| > 1 时为 0。
    """
    abs_t = np.abs(t)
    values = np.zeros_like(abs_t)
    flat = abs_t <= 0.5
    slope = (abs_t > 0.5) & (abs_t <= 1.0)
    values[flat] = 1.0
    values[slope] = 2.0 * (1.0 - abs_t[slope])
    return values


def optimal_block_length(series: FloatArray | list[float]) -> BlockLengthResult:
    """Politis & White (2004) 自动选块长（含 Patton, Politis & White 2009 的更正）。

    做法：先用"连续 k_n 个自相关都落进 ±2√(log₁₀n/n) 带内"定截断点 m̂，
    再以 M = 2·m̂ 为带宽、用 flat-top 核估 ĝ(0)（谱密度在 0 点的值）与
    Ĝ（谱密度导数相关量），最后 b_opt = (2·Ĝ²/D̂)^(1/3)·n^(1/3)。
    平稳自助 D̂ = 2ĝ(0)²，循环分块 D̂ = (4/3)ĝ(0)²。

    序列常数（方差为 0）或长度不足时返回块长 1，此时调用方会因块数检查而收到告警。
    """
    values = np.asarray(series, dtype=float).ravel()
    n = values.size
    if n < 8:
        return BlockLengthResult(1, 1, 0, 0, 0.0)

    centred = values - values.mean()
    variance = float(np.dot(centred, centred) / n)
    if variance <= 0.0:
        return BlockLengthResult(1, 1, 0, 0, 0.0)

    k_n = max(5, int(np.ceil(np.sqrt(np.log10(n)))))
    m_max = min(int(np.ceil(np.sqrt(n))) + k_n, n - 1)

    autocov = np.empty(m_max + 1, dtype=float)
    for lag in range(m_max + 1):
        autocov[lag] = float(np.dot(centred[: n - lag], centred[lag:]) / n)
    autocorr = autocov / autocov[0]

    critical = 2.0 * np.sqrt(np.log10(n) / n)

    m_hat = 0
    for start in range(1, m_max + 1):
        window = autocorr[start : min(start + k_n, m_max + 1)]
        if window.size and bool(np.all(np.abs(window) < critical)):
            m_hat = start
            break
    if m_hat == 0:
        significant = np.nonzero(np.abs(autocorr[1:]) >= critical)[0]
        m_hat = int(significant[-1]) + 1 if significant.size else 1

    bandwidth = int(min(2 * m_hat, m_max))
    bandwidth = max(bandwidth, 1)

    lags = np.arange(bandwidth + 1, dtype=float)
    weights = _flat_top_kernel(lags / bandwidth)
    g_hat = float(autocov[0] + 2.0 * np.sum(weights[1:] * autocov[1 : bandwidth + 1]))
    big_g = float(2.0 * np.sum(weights[1:] * lags[1:] * autocov[1 : bandwidth + 1]))

    b_max = float(np.ceil(min(3.0 * np.sqrt(n), n / 3.0)))

    def _solve(d_hat: float) -> int:
        if d_hat <= 0.0 or big_g == 0.0:
            return 1
        raw = (2.0 * big_g * big_g / d_hat) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
        return int(max(1, min(b_max, np.ceil(raw))))

    return BlockLengthResult(
        stationary=_solve(2.0 * g_hat * g_hat),
        circular=_solve(4.0 / 3.0 * g_hat * g_hat),
        mhat=m_hat,
        bandwidth=bandwidth,
        lag1_autocorrelation=float(autocorr[1]) if m_max >= 1 else 0.0,
    )


# ══════════════════════════════════════════════════════════════════════
# 重排索引生成
# ══════════════════════════════════════════════════════════════════════

def block_permutation_indices(
    n: int, block_length: int, rng: np.random.Generator
) -> IntArray:
    """分块重排：切成若干近等长的块，打乱块的顺序后拼回来。

    这是**严格的置换**——每个原始位置恰好出现一次，因此收益的多重集合完全保留
    （夏普恒等不变正源于此）。块数 = ceil(n / block_length)，
    `np.array_split` 保证各块长度最多相差 1。
    """
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    block_length = max(1, min(block_length, n))
    n_blocks = int(np.ceil(n / block_length))
    blocks = np.array_split(np.arange(n, dtype=np.int64), n_blocks)
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order]).astype(np.int64)


def circular_block_indices(
    n: int, block_length: int, rng: np.random.Generator
) -> IntArray:
    """循环分块自助：等长块、有放回抽取块起点、跨越末尾时绕回开头。

    与分块重排不同，这是**有放回抽样**，同一天可能出现多次或一次都不出现。
    """
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    block_length = max(1, min(block_length, n))
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks, dtype=np.int64)
    offsets = np.arange(block_length, dtype=np.int64)
    grid = (starts[:, None] + offsets[None, :]) % n
    return grid.reshape(-1)[:n].astype(np.int64)


def stationary_bootstrap_indices(
    n: int, mean_block_length: float, rng: np.random.Generator
) -> IntArray:
    """Politis & Romano (1994) 平稳自助：块长服从几何分布，均值为 mean_block_length。

    每一步以概率 p = 1/L 跳到一个新的随机起点，否则沿着上一位置往前走一格（循环）。
    相比固定块长，它的重抽样序列本身是平稳的，这也是它得名的原因。
    """
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    mean_block_length = max(1.0, min(float(mean_block_length), float(n)))
    p = 1.0 / mean_block_length

    starts = rng.integers(0, n, size=n, dtype=np.int64)
    fresh = rng.random(n) < p
    fresh[0] = True

    positions = np.arange(n, dtype=np.int64)
    last_fresh = np.maximum.accumulate(np.where(fresh, positions, -1))
    base = starts[last_fresh]
    offset = positions - last_fresh
    return ((base + offset) % n).astype(np.int64)


def random_rotation_indices(n: int, rng: np.random.Generator) -> IntArray:
    """整体循环平移。完整保留序列的全部依赖结构，只破坏它与另一条序列的对齐。

    只有 n-1 种非平凡平移，所以最小可达 p 值受限于 n，且相邻平移高度相关。
    """
    if n <= 1:
        return np.arange(n, dtype=np.int64)
    shift = int(rng.integers(1, n))
    return ((np.arange(n, dtype=np.int64) + shift) % n).astype(np.int64)


def _index_matrix(
    n: int,
    n_permutations: int,
    block_length: int,
    scheme: PermutationScheme,
    rng: np.random.Generator,
) -> IntArray:
    """按方案生成 (B, n) 的重排索引矩阵。"""
    rows = np.empty((n_permutations, n), dtype=np.int64)
    for b in range(n_permutations):
        if scheme is PermutationScheme.POSITIONS_ROTATE:
            rows[b] = random_rotation_indices(n, rng)
        elif scheme is PermutationScheme.RETURNS_BOOTSTRAP_H0:
            rows[b] = stationary_bootstrap_indices(n, block_length, rng)
        else:
            rows[b] = block_permutation_indices(n, block_length, rng)
    return rows


# ══════════════════════════════════════════════════════════════════════
# 统计量
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Statistic:
    """待检验的统计量。

    matrix_fn 接受 (B, n) 的日盈亏矩阵、一次算出 B 个值，是 B=1000 也能进寻优循环的
    关键。统计量无定义时（例如爆仓）应返回 NaN —— NaN 在 p 值里计为"未超过观测值"，
    并从零分布的均值/标准差里剔除。
    """

    name: str
    matrix_fn: Callable[[FloatArray], FloatArray]

    def evaluate(self, pnl_matrix: FloatArray) -> FloatArray:
        return np.asarray(self.matrix_fn(pnl_matrix), dtype=float)

    def evaluate_one(self, pnl: FloatArray) -> float:
        return float(self.evaluate(np.asarray(pnl, dtype=float)[None, :])[0])


def _balance_and_log_returns(
    pnl_matrix: FloatArray, capital: float
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    """由日盈亏矩阵还原净值与对数收益，口径与 backtesting.calculate_statistics 一致。

    vnpy 的做法：balance = capital + cumsum(net_pnl)，return = log(balance / 前一日 balance)，
    比值 ≤ 0 的那天收益记 0。第三个返回值标出"曾经爆仓"的行。
    """
    balance = capital + np.cumsum(pnl_matrix, axis=1)
    previous = np.concatenate(
        [np.full((pnl_matrix.shape[0], 1), capital, dtype=float), balance[:, :-1]],
        axis=1,
    )
    ratio = np.divide(
        balance,
        previous,
        out=np.zeros_like(balance),
        where=previous > 0,
    )
    log_returns = np.zeros_like(ratio)
    positive = ratio > 0
    log_returns[positive] = np.log(ratio[positive])
    ruined = ~np.all(balance > 0, axis=1)
    return balance, log_returns, ruined


def make_statistic(
    name: str,
    capital: float,
    annual_days: int,
    risk_free: float = 0.0,
) -> Statistic:
    """按名字构造统计量。

    可用名字：
        sharpe_ratio           年化夏普（与 calculate_statistics 同口径）
        total_return           总收益率 %
        annual_return          年化收益率 %
        max_ddpercent          百分比最大回撤（负数；越大越好，故 alternative 仍用 greater）
        return_drawdown_ratio  收益回撤比
    """
    if capital <= 0:
        raise ValueError(f"capital 必须为正，收到 {capital}")
    root = float(np.sqrt(annual_days))

    def sharpe(pnl_matrix: FloatArray) -> FloatArray:
        _, log_returns, ruined = _balance_and_log_returns(pnl_matrix, capital)
        mean = log_returns.mean(axis=1) * 100.0
        std = log_returns.std(axis=1, ddof=1) * 100.0
        daily_risk_free = risk_free / root
        out = np.full(pnl_matrix.shape[0], np.nan, dtype=float)
        usable = (std > 0) & ~ruined
        out[usable] = (mean[usable] - daily_risk_free) / std[usable] * root
        return out

    def total_return(pnl_matrix: FloatArray) -> FloatArray:
        balance, _, ruined = _balance_and_log_returns(pnl_matrix, capital)
        out = (balance[:, -1] / capital - 1.0) * 100.0
        out[ruined] = np.nan
        return out

    def annual_return(pnl_matrix: FloatArray) -> FloatArray:
        scaled: FloatArray = total_return(pnl_matrix) / pnl_matrix.shape[1] * annual_days
        return scaled

    def max_ddpercent(pnl_matrix: FloatArray) -> FloatArray:
        balance, _, ruined = _balance_and_log_returns(pnl_matrix, capital)
        high = np.maximum.accumulate(balance, axis=1)
        drawdown = np.divide(
            balance - high, high, out=np.zeros_like(balance), where=high > 0
        ) * 100.0
        out = drawdown.min(axis=1)
        out[ruined] = np.nan
        return out

    def return_drawdown_ratio(pnl_matrix: FloatArray) -> FloatArray:
        total = total_return(pnl_matrix)
        worst = max_ddpercent(pnl_matrix)
        out = np.full(pnl_matrix.shape[0], np.nan, dtype=float)
        usable = worst < 0
        out[usable] = -total[usable] / worst[usable]
        return out

    table: dict[str, Callable[[FloatArray], FloatArray]] = {
        "sharpe_ratio": sharpe,
        "total_return": total_return,
        "annual_return": annual_return,
        "max_ddpercent": max_ddpercent,
        "return_drawdown_ratio": return_drawdown_ratio,
    }
    if name not in table:
        raise ValueError(f"未知统计量 {name!r}，可选：{sorted(table)}")
    return Statistic(name=name, matrix_fn=table[name])


# ══════════════════════════════════════════════════════════════════════
# p 值与结果容器
# ══════════════════════════════════════════════════════════════════════

def empirical_p_value(
    observed: float,
    null_values: FloatArray,
    alternative: str = "greater",
) -> float:
    """经验 p 值，带 Phipson & Smyth (2010) 的 +1 修正。

    p = (1 + #{零分布 ≥ 观测值}) / (B + 1)

    +1 是必须的：不加的话 p 有可能算出 0，而"B 次重排里没有一次超过观测值"
    根本不能支持 p=0 这个结论，它只支持 p < 1/(B+1)。修正后 p 值恒为正，
    且在零假设下是有效的（第一类错误率 ≤ α）。
    NaN（统计量无定义，例如爆仓）一律计为"未超过观测值"。
    """
    values = np.asarray(null_values, dtype=float)
    total = values.size
    if total == 0:
        return 1.0
    if not np.isfinite(observed):
        return 1.0

    greater = int(np.sum(np.nan_to_num(values, nan=-np.inf) >= observed))
    less = int(np.sum(np.nan_to_num(values, nan=np.inf) <= observed))

    p_greater = (1.0 + greater) / (total + 1.0)
    p_less = (1.0 + less) / (total + 1.0)

    if alternative == "greater":
        return p_greater
    if alternative == "less":
        return p_less
    if alternative == "two-sided":
        return min(1.0, 2.0 * min(p_greater, p_less))
    raise ValueError(f"alternative 只能是 greater/less/two-sided，收到 {alternative!r}")


@dataclass(frozen=True)
class PermutationResult:
    """检验结果。字段命名尽量自解释，避免回头翻文档。"""

    scheme: str
    statistic_name: str
    alternative: str
    observed: float
    p_value: float
    n_permutations: int
    sample_size: int
    block_length: int
    n_blocks: int
    null_mean: float
    null_std: float
    null_p05: float
    null_p50: float
    null_p95: float
    z_score: float
    min_detectable_effect: float
    engine_observed: float
    reconstruction_error: float
    """Σ|重构日盈亏 − 引擎日盈亏| ÷ Σ|引擎日盈亏|。逐日口径，非总额口径。"""

    degenerate: bool
    """零分布退化 = 本次检验没有功效，p_value 无意义。

    机器可读，专门用来替代"在 warnings 里找关键字"这种脆做法。
    下游（gated_statistic / 发布闸）应当直接读它。
    """

    warnings: tuple[str, ...]

    @property
    def significant(self) -> bool:
        """按 α=0.05 判定。想用别的水平直接比 p_value。

        退化时恒为 False —— 没有功效的检验不允许输出"显著"。
        """
        return (not self.degenerate) and self.p_value <= DEFAULT_ALPHA

    @property
    def has_power(self) -> bool:
        """这次检验到底有没有功效。False 时 p_value 不可解读。"""
        return not self.degenerate

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"重排检验 [{self.scheme}] 统计量={self.statistic_name} 备择={self.alternative}",
            f"  观测值        {self.observed:,.4f}（引擎值 {self.engine_observed:,.4f}）",
            f"  零分布        均值 {self.null_mean:,.4f}  标准差 {self.null_std:,.4f}"
            f"  [5% {self.null_p05:,.4f} | 50% {self.null_p50:,.4f} | 95% {self.null_p95:,.4f}]",
            f"  p 值          {self.p_value:.4f}"
            f"（B={self.n_permutations}，最小可达 {1.0 / (self.n_permutations + 1):.4f}）"
            + ("  ← 无功效，勿解读" if self.degenerate else ""),
            f"  z 分数        {self.z_score:,.2f}",
            f"  块长          {self.block_length}（共 {self.n_blocks} 块，n={self.sample_size}）",
            f"  80% 功效门槛  效应量需 ≥ 零分布中心 + {self.min_detectable_effect:,.4f}",
        ]
        lines.extend(f"  ⚠ {w}" for w in self.warnings)
        return "\n".join(lines)


def _summarise(
    observed: float,
    engine_observed: float,
    null_values: FloatArray,
    scheme: PermutationScheme,
    statistic_name: str,
    alternative: str,
    block_length: int,
    n_blocks: int,
    sample_size: int,
    reconstruction_error: float,
    warnings: list[str],
    alpha: float,
    power: float,
    known_invariant: bool = False,
    exposure_column: str = "start_pos",
) -> PermutationResult:
    finite = null_values[np.isfinite(null_values)]
    if finite.size < null_values.size:
        warnings.append(
            f"{null_values.size - finite.size}/{null_values.size} 次重排的统计量无定义"
            "（爆仓或方差为零），已计为未超过观测值"
        )

    null_mean = float(np.mean(finite)) if finite.size else float("nan")
    null_std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    p05, p50, p95 = (
        [float(v) for v in np.percentile(finite, [5, 50, 95])]
        if finite.size
        else [float("nan")] * 3
    )

    p_value = empirical_p_value(observed, null_values, alternative)
    z_score = (observed - null_mean) / null_std if null_std > 0 else 0.0

    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - alpha) if alternative != "two-sided" else normal.inv_cdf(1.0 - alpha / 2.0)
    z_power = normal.inv_cdf(power)
    min_detectable = (z_alpha + z_power) * null_std

    candidates = [abs(v) for v in (observed, null_mean) if np.isfinite(v)]
    effect_scale = max(candidates) if candidates else 0.0
    degenerate = False
    if known_invariant:
        # 解析事实，与幅度/仓位/种子全部无关，最先判、判死。
        degenerate = True
        warnings.append(
            f"统计量 {statistic_name} 只依赖 net_pnl 的多重集合，"
            f"而 {scheme.value} 是严格置换 —— 二者组合恒等不变，"
            "零分布里剩下的只有复利记账顺序的残差。"
            "本次检验没有功效，p 值不可用：请换路径依赖统计量"
            "（max_ddpercent / return_drawdown_ratio）或换 POSITIONS_BLOCK 方案"
        )
    elif null_std == 0.0:
        degenerate = True
        warnings.append(
            "零分布完全退化（标准差为 0）：本次检验没有功效，p 值不可用"
        )
    elif effect_scale > 0.0 and null_std / effect_scale < _DEGENERATE_NULL_RATIO:
        degenerate = True
        warnings.append(
            f"零分布标准差仅为效应量尺度的 {null_std / effect_scale:.2%}"
            f"（阈值 {_DEGENERATE_NULL_RATIO:.0%}）：统计量 {statistic_name} 对 "
            f"{scheme.value} 重排近似恒等不变，本次检验没有功效，p 值不可用"
            "——换统计量或换方案"
        )
    if n_blocks < DEFAULT_MIN_BLOCKS:
        warnings.append(f"只有 {n_blocks} 个块，重排自由度不足，p 值参考价值有限")
    if abs(reconstruction_error) > _RECONSTRUCTION_TOLERANCE:
        other = "end_pos" if exposure_column == "start_pos" else "start_pos"
        warnings.append(
            f"重构日盈亏与引擎 net_pnl 的逐日相对偏差 {reconstruction_error:.1%}，"
            f"超过 {_RECONSTRUCTION_TOLERANCE:.0%}：当前用的是 "
            f"exposure_column={exposure_column!r}，暴露口径多半选错了。"
            f"改用 {other!r} 试试，或改用 BARS 方案。"
            "⚠ 收盘价成交的策略若误用 end_pos，会把当日尚未持有的仓位算成全天持有"
            "＝前视偏差，实测能把夏普从 1.0 抬到 2.0、p 值从 0.08 压到 0.002"
        )

    return PermutationResult(
        scheme=scheme.value,
        statistic_name=statistic_name,
        alternative=alternative,
        observed=float(observed),
        p_value=p_value,
        n_permutations=int(null_values.size),
        sample_size=int(sample_size),
        block_length=int(block_length),
        n_blocks=int(n_blocks),
        null_mean=null_mean,
        null_std=null_std,
        null_p05=p05,
        null_p50=p50,
        null_p95=p95,
        z_score=float(z_score),
        min_detectable_effect=float(min_detectable),
        engine_observed=float(engine_observed),
        reconstruction_error=float(reconstruction_error),
        degenerate=degenerate,
        warnings=tuple(warnings),
    )


def _resolve_block_length(
    series: FloatArray,
    block_length: int | None,
    scheme: PermutationScheme,
    min_blocks: int,
    warnings: list[str],
) -> int:
    """块长：外部指定优先，否则 Politis & White 自动选，最后受块数下限约束。"""
    n = series.size
    if block_length is None:
        chosen = optimal_block_length(series)
        block_length = (
            chosen.stationary
            if scheme is PermutationScheme.RETURNS_BOOTSTRAP_H0
            else chosen.circular
        )
        warnings.append(
            f"自动选块长 L={block_length}（Politis-White，ρ₁={chosen.lag1_autocorrelation:.3f}，"
            f"m̂={chosen.mhat}）"
        )
    block_length = max(1, min(int(block_length), n))

    max_allowed = max(1, n // min_blocks)
    if block_length > max_allowed:
        warnings.append(
            f"块长 {block_length} 会让块数低于 {min_blocks}，压回 {max_allowed}"
        )
        block_length = max_allowed
    return block_length


# ══════════════════════════════════════════════════════════════════════
# 方案 C：重排持仓（默认）
# ══════════════════════════════════════════════════════════════════════

def _extract_exposure(
    daily_df: DataFrame,
    size: float,
    exposure_column: str,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """从 daily_df 取出 (暴露, 每单位持仓盈亏, 收盘价, 实际成本)。

    每单位持仓盈亏 = (close - pre_close) × size，与 DailyResult.calculate_pnl 里
    holding_pnl = start_pos × (close - pre_close) × size 完全同源。
    """
    required = {exposure_column, "close_price", "pre_close", "net_pnl"}
    missing = required - set(daily_df.columns)
    if missing:
        raise ValueError(f"daily_df 缺少列：{sorted(missing)}")

    exposure = daily_df[exposure_column].to_numpy(dtype=float)
    close = daily_df["close_price"].to_numpy(dtype=float)
    pre_close = daily_df["pre_close"].to_numpy(dtype=float)
    unit_pnl = (close - pre_close) * size

    cost = np.zeros_like(exposure)
    for column in ("commission", "slippage"):
        if column in daily_df.columns:
            cost = cost + daily_df[column].to_numpy(dtype=float)
    return exposure, unit_pnl, close, cost


def _turnover_notional(exposure: FloatArray, close: FloatArray, size: float) -> FloatArray:
    """逐日成交名义金额，长度与 exposure 相同。

    第 t 日记 |e_t − e_{t−1}| × close_t × size（e_{−1}=0，即首日建仓计入）；
    末日的清仓折算进最后一天，**刻意不额外多出一个元素** ——
    多出来那一格会让"总换手"与"逐日换手之和"对不上，
    进而让成本率的分母比实际收费的天数多一天。
    """
    changes = np.abs(np.diff(np.concatenate([[0.0], exposure])))
    changes[-1] += abs(exposure[-1])                 # 末日清仓折进最后一天
    notional: FloatArray = changes * close * size
    return notional


def _turnover_notional_matrix(
    exposure_matrix: FloatArray, close: FloatArray, size: float
) -> FloatArray:
    """_turnover_notional 的批量版本，逐行口径完全一致。"""
    rows = exposure_matrix.shape[0]
    zeros = np.zeros((rows, 1), dtype=float)
    changes = np.abs(np.diff(np.concatenate([zeros, exposure_matrix], axis=1), axis=1))
    changes[:, -1] += np.abs(exposure_matrix[:, -1])
    notional: FloatArray = changes * close[None, :] * size
    return notional


def permutation_test_positions(
    daily_df: DataFrame,
    capital: float,
    annual_days: int,
    statistic: str | Statistic = "sharpe_ratio",
    n_permutations: int = DEFAULT_PERMUTATIONS,
    block_length: int | None = None,
    scheme: PermutationScheme = PermutationScheme.POSITIONS_BLOCK,
    size: float = 1.0,
    exposure_column: str = "start_pos",
    include_costs: bool = True,
    alternative: str = "greater",
    seed: int | None = 20260725,
    min_blocks: int = DEFAULT_MIN_BLOCKS,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    risk_free: float = 0.0,
) -> PermutationResult:
    """重排持仓序列、市场收益不动，检验"择时是否强于随机择时"。

    daily_df       BacktestingEngine.calculate_result() 的输出
    size           合约乘数；港美股现货为 1
    exposure_column 用哪一列当日内暴露。默认 start_pos（当日开盘时的持仓），
                   与 vnpy 的 holding_pnl 口径一致。若策略在开盘成交，改用 end_pos
                   重构误差通常更小 —— 结果里的 reconstruction_error 会告诉你。
    include_costs  重排后仓位变动次数会变，手续费/滑点随之改变。开启时按实测的
                   "每单位成交名义金额成本率"重算，关闭则零分布不含成本（偏乐观）。

    观测值刻意用【与零分布同一套重构逻辑】算出来，而不是直接取引擎的 net_pnl —— 否则
    分子分母口径不同，p 值会被重构误差污染。引擎的真值另存 engine_observed 供对照。
    """
    if not isinstance(daily_df, DataFrame) or daily_df.empty:
        raise ValueError("daily_df 为空，先跑 calculate_result()")
    if scheme not in (
        PermutationScheme.POSITIONS_BLOCK,
        PermutationScheme.POSITIONS_ROTATE,
    ):
        raise ValueError(f"{scheme.value} 不是持仓重排方案")

    stat = (
        make_statistic(statistic, capital, annual_days, risk_free)
        if isinstance(statistic, str)
        else statistic
    )
    exposure, unit_pnl, close, actual_cost = _extract_exposure(
        daily_df, size, exposure_column
    )
    n = exposure.size
    warnings: list[str] = []

    total_cost = float(actual_cost.sum())
    notional = _turnover_notional(exposure, close, size)
    total_notional = float(notional.sum())
    cost_rate = (
        total_cost / total_notional if include_costs and total_notional > 0 else 0.0
    )
    if include_costs and total_notional <= 0 and total_cost > 0:
        warnings.append("暴露序列恒为零但存在成本，零分布不含成本")

    observed_pnl = exposure * unit_pnl - cost_rate * notional
    observed = stat.evaluate_one(observed_pnl)

    engine_pnl = daily_df["net_pnl"].to_numpy(dtype=float)
    engine_observed = stat.evaluate_one(engine_pnl)
    # 逐日相对误差，不是总额相对误差：总额分母在策略接近打平时会趋近 0，
    # 让一个健康的重构也报出百分之几十的"误差"（实测踩过：总收益 -0.0% 时报 29%）。
    # 而且重排作用在【逐日】盈亏上，逐日跟得准不准才是这里真正要问的事。
    engine_scale = float(np.abs(engine_pnl).sum())
    reconstruction_error = (
        float(np.abs(observed_pnl - engine_pnl).sum()) / engine_scale
        if engine_scale > 0
        else 0.0
    )

    if scheme is PermutationScheme.POSITIONS_ROTATE:
        block_length = n
        warnings.append(
            f"循环平移只有 {n - 1} 种非平凡重排，最小可达 p 值受限于 n 而非 B"
        )
        n_blocks = 1
    else:
        block_length = _resolve_block_length(
            exposure, block_length, scheme, min_blocks, warnings
        )
        n_blocks = int(np.ceil(n / block_length))

    rng = np.random.default_rng(seed)
    indices = _index_matrix(n, n_permutations, block_length, scheme, rng)
    exposure_matrix = exposure[indices]

    null_pnl = exposure_matrix * unit_pnl[None, :]
    if cost_rate > 0:
        null_pnl = null_pnl - cost_rate * _turnover_notional_matrix(
            exposure_matrix, close, size
        )
    null_values = stat.evaluate(null_pnl)

    return _summarise(
        observed=observed,
        engine_observed=engine_observed,
        null_values=null_values,
        scheme=scheme,
        statistic_name=stat.name,
        alternative=alternative,
        block_length=block_length,
        n_blocks=n_blocks,
        sample_size=n,
        reconstruction_error=reconstruction_error,
        warnings=warnings,
        alpha=alpha,
        power=power,
        exposure_column=exposure_column,
    )


# ══════════════════════════════════════════════════════════════════════
# 方案 A：重排策略收益
# ══════════════════════════════════════════════════════════════════════

def permutation_test_returns(
    daily_df: DataFrame,
    capital: float,
    annual_days: int,
    statistic: str | Statistic = "return_drawdown_ratio",
    n_permutations: int = DEFAULT_PERMUTATIONS,
    block_length: int | None = None,
    scheme: PermutationScheme = PermutationScheme.RETURNS_BLOCK,
    alternative: str = "greater",
    seed: int | None = 20260725,
    min_blocks: int = DEFAULT_MIN_BLOCKS,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    risk_free: float = 0.0,
) -> PermutationResult:
    """只动策略日收益序列。

    RETURNS_BLOCK：严格置换，日收益多重集合不变 → **夏普、总收益、年化收益全部恒等
    不变**，只能配路径依赖统计量（默认 return_drawdown_ratio）。它回答的是
    "这条曲线的回撤形态，相对同一批日收益的随机排列，是不是异常"，
    不回答"有没有 edge"。

    RETURNS_BOOTSTRAP_H0：先把日盈亏减去均值（造出零假设世界），再平稳自助。
    它检验 H0: E[日收益]=0，可以配夏普。但对 long-only 策略，市场 beta 会让它
    轻易显著 —— 想把 beta 约掉请用 POSITIONS_BLOCK。
    """
    if not isinstance(daily_df, DataFrame) or daily_df.empty:
        raise ValueError("daily_df 为空，先跑 calculate_result()")
    if scheme not in (
        PermutationScheme.RETURNS_BLOCK,
        PermutationScheme.RETURNS_BOOTSTRAP_H0,
    ):
        raise ValueError(f"{scheme.value} 不是收益重排方案")

    stat = (
        make_statistic(statistic, capital, annual_days, risk_free)
        if isinstance(statistic, str)
        else statistic
    )
    pnl = daily_df["net_pnl"].to_numpy(dtype=float)
    n = pnl.size
    warnings: list[str] = []

    observed = stat.evaluate_one(pnl)
    block_length = _resolve_block_length(pnl, block_length, scheme, min_blocks, warnings)
    n_blocks = int(np.ceil(n / block_length))

    source = pnl
    if scheme is PermutationScheme.RETURNS_BOOTSTRAP_H0:
        source = pnl - pnl.mean()
        warnings.append("已把日盈亏去均值以构造零假设世界（H0: E[收益]=0）")

    rng = np.random.default_rng(seed)
    indices = _index_matrix(n, n_permutations, block_length, scheme, rng)
    null_values = stat.evaluate(source[indices])

    # RETURNS_BLOCK 是严格置换：多重集合函数在它下面恒等不变。
    # RETURNS_BOOTSTRAP_H0 是有放回重抽，多重集合会变，不适用本判据。
    known_invariant = (
        scheme is PermutationScheme.RETURNS_BLOCK
        and stat.name in _MULTISET_INVARIANT
    )

    return _summarise(
        observed=observed,
        engine_observed=observed,
        null_values=null_values,
        scheme=scheme,
        statistic_name=stat.name,
        alternative=alternative,
        block_length=block_length,
        n_blocks=n_blocks,
        sample_size=n,
        reconstruction_error=0.0,
        warnings=warnings,
        alpha=alpha,
        power=power,
        known_invariant=known_invariant,
    )


# ══════════════════════════════════════════════════════════════════════
# 方案 B：重排价格后重跑策略（Masters MCPT）
# ══════════════════════════════════════════════════════════════════════

def permute_price_series(
    prices: FloatArray, indices: IntArray
) -> FloatArray:
    """按给定索引重排对数收益并重建价格路径，起点价格保持不变。

    重排的是 n-1 个对数收益，所以 indices 的长度必须是 len(prices)-1。
    """
    values = np.asarray(prices, dtype=float)
    if values.size < 2:
        return values.copy()
    if np.any(values <= 0):
        raise ValueError("价格序列必须全为正数才能取对数收益")
    log_returns = np.diff(np.log(values))
    if indices.size != log_returns.size:
        raise ValueError(
            f"indices 长度 {indices.size} 与对数收益长度 {log_returns.size} 不符"
        )
    return np.concatenate(
        [[values[0]], values[0] * np.exp(np.cumsum(log_returns[indices]))]
    )


def permute_ohlc(
    open_: FloatArray,
    high: FloatArray,
    low: FloatArray,
    close: FloatArray,
    indices: IntArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """重排 OHLC 日线，保持每根 K 线自身的几何形状不变。

    做法：把每根 K 线拆成"收盘到收盘的对数收益"与"开/高/低相对收盘的对数偏移"，
    **整组一起重排**，再按新的收盘价重建。这样 high ≥ max(open, close) ≥ min(...) ≥ low
    这些内部关系天然保持成立，不会造出不可能的 K 线。

    与 Masters 把跳空和日内偏移独立重排的做法相比，本实现更保守：它只打散跨日的
    时序结构，不额外破坏日内结构。对只看收盘价的日线 CTA 二者等价。
    """
    arrays = [np.asarray(a, dtype=float) for a in (open_, high, low, close)]
    if len({a.size for a in arrays}) != 1:
        raise ValueError("OHLC 四条序列长度必须一致")
    o, h, low_arr, c = arrays
    if np.any(np.concatenate(arrays) <= 0):
        raise ValueError("价格必须全为正数才能取对数偏移")

    new_close = permute_price_series(c, indices)
    # 第 i+1 根新 K 线取自原第 indices[i]+1 根；第 0 根保持原样
    shift = np.concatenate([np.zeros(1, dtype=np.int64), indices + 1])
    offsets_open = np.log(o / c)[shift]
    offsets_high = np.log(h / c)[shift]
    offsets_low = np.log(low_arr / c)[shift]
    return (
        new_close * np.exp(offsets_open),
        new_close * np.exp(offsets_high),
        new_close * np.exp(offsets_low),
        new_close,
    )


def permutation_test_bars(
    prices: FloatArray,
    rerun: Callable[[FloatArray], float],
    n_permutations: int = DEFAULT_PERMUTATIONS,
    block_length: int | None = None,
    alternative: str = "greater",
    seed: int | None = 20260725,
    min_blocks: int = DEFAULT_MIN_BLOCKS,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    statistic_name: str = "sharpe_ratio",
) -> PermutationResult:
    """Masters MCPT：分块重排价格序列，每次重跑策略。

    rerun 接收一条与输入等长的价格序列，返回该序列上的统计量（一个标量）。
    整个回测（含数据准备、下单、成本）都应封装在 rerun 里 —— 只要 rerun 里包含
    参数寻优，得到的就是**已校正选择偏差**的 p 值，见 selection_bias_corrected_test。

    代价：B 次完整回测。600 日 × B=1000，若单次回测 0.2 秒即约 3.3 分钟；
    含寻优的话乘以网格大小，通常要下降到 B=200-500 才现实（p 值分辨率随之降到 1/201）。
    """
    values = np.asarray(prices, dtype=float)
    n = values.size
    if n < 3:
        raise ValueError(f"价格序列太短：{n}")

    warnings: list[str] = []
    log_returns = np.diff(np.log(values))
    block_length = _resolve_block_length(
        log_returns, block_length, PermutationScheme.RETURNS_BLOCK, min_blocks, warnings
    )
    n_blocks = int(np.ceil(log_returns.size / block_length))

    observed = float(rerun(values))
    rng = np.random.default_rng(seed)
    null_values = np.empty(n_permutations, dtype=float)
    for b in range(n_permutations):
        indices = block_permutation_indices(log_returns.size, block_length, rng)
        null_values[b] = float(rerun(permute_price_series(values, indices)))

    return _summarise(
        observed=observed,
        engine_observed=observed,
        null_values=null_values,
        scheme=PermutationScheme.RETURNS_BLOCK,
        statistic_name=statistic_name,
        alternative=alternative,
        block_length=block_length,
        n_blocks=n_blocks,
        sample_size=n,
        reconstruction_error=0.0,
        warnings=warnings,
        alpha=alpha,
        power=power,
    )


def selection_bias_corrected_test(
    prices: FloatArray,
    optimise_and_score: Callable[[FloatArray], float],
    n_permutations: int = 200,
    block_length: int | None = None,
    alternative: str = "greater",
    seed: int | None = 20260725,
    min_blocks: int = DEFAULT_MIN_BLOCKS,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    statistic_name: str = "sharpe_ratio",
) -> PermutationResult:
    """选择偏差校正版：把【整个参数寻优流程】当成被检验的对象。

    optimise_and_score 拿到一条价格序列后，要在这条序列上把参数网格重新跑一遍，
    返回**最优那组参数的成绩**。于是零分布是"在没有结构的数据上，让寻优器尽情挑最好的
    一组，能挑出多好"，实测值与之相比才是干净的。

    这是唯一能同时回答"策略有没有 edge"和"这个 edge 是不是网格挑出来的"的做法，
    也是唯一贵到需要预算的做法：B × 网格大小 次回测。默认把 B 降到 200
    （最小可达 p = 1/201 ≈ 0.005，够判 α=0.05）。

    与 permutation_test_bars 的唯一区别是【调用者的责任】而非实现：
    传进来的回调必须含寻优步骤。函数签名相同，故此处只做透传与默认值调整。
    """
    return permutation_test_bars(
        prices,
        optimise_and_score,
        n_permutations=n_permutations,
        block_length=block_length,
        alternative=alternative,
        seed=seed,
        min_blocks=min_blocks,
        alpha=alpha,
        power=power,
        statistic_name=statistic_name,
    )


# ══════════════════════════════════════════════════════════════════════
# 功效
# ══════════════════════════════════════════════════════════════════════

def sharpe_standard_error(
    n_days: int, annual_days: int, annual_sharpe: float = 0.0
) -> float:
    """年化夏普估计量的标准误（Lo 2002，iid 情形）。

    SE(年化夏普) = √((1 + SR_日²/2) / n) × √annual_days
    SR_日 = 年化夏普 / √annual_days，日频下平方项可忽略，所以近似 √(annual_days/n)。

    n=600, annual_days=252 → 0.648；n=134 → 1.371（对上"134 日曲线 SE≈1.36"）。
    序列相关会进一步放大它，本函数不含该修正，故是**乐观下界**。
    """
    if n_days <= 1 or annual_days <= 0:
        return float("inf")
    daily_sharpe = annual_sharpe / np.sqrt(annual_days)
    return float(
        np.sqrt((1.0 + 0.5 * daily_sharpe**2) / n_days) * np.sqrt(annual_days)
    )


def minimum_detectable_sharpe(
    n_days: int,
    annual_days: int = 252,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    two_sided: bool = False,
) -> float:
    """给定样本长度，能以指定功效检出的最小真实年化夏普。

    (z_α + z_power) × SE。n=600/annual_days=252/α=0.05 单边/功效 0.8 → 约 1.61。
    """
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - (alpha / 2.0 if two_sided else alpha))
    z_power = normal.inv_cdf(power)
    return (z_alpha + z_power) * sharpe_standard_error(n_days, annual_days)


# ══════════════════════════════════════════════════════════════════════
# 与 calculate_statistics / 参数寻优对接
# ══════════════════════════════════════════════════════════════════════

def permutation_statistics(
    daily_df: DataFrame,
    capital: float,
    annual_days: int,
    statistic: str = "sharpe_ratio",
    n_permutations: int = DEFAULT_PERMUTATIONS,
    scheme: PermutationScheme = PermutationScheme.POSITIONS_BLOCK,
    size: float = 1.0,
    seed: int | None = 20260725,
    block_length: int | None = None,
    exposure_column: str = "start_pos",
    include_costs: bool = True,
    alternative: str = "greater",
    min_blocks: int = DEFAULT_MIN_BLOCKS,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    risk_free: float = 0.0,
) -> dict[str, object]:
    """跑一次检验并压成可直接并进 statistics dict 的扁平字典。

    键一律带 perm_ 前缀，不与既有键冲突。值全部是 float / int / str，
    能安全通过 calculate_statistics 末尾那圈 np.nan_to_num 过滤。
    """
    result = permutation_test_positions(
        daily_df,
        capital=capital,
        annual_days=annual_days,
        statistic=statistic,
        n_permutations=n_permutations,
        scheme=scheme,
        size=size,
        seed=seed,
        block_length=block_length,
        exposure_column=exposure_column,
        include_costs=include_costs,
        alternative=alternative,
        min_blocks=min_blocks,
        alpha=alpha,
        power=power,
        risk_free=risk_free,
    )
    return {
        "perm_statistic": result.statistic_name,
        "perm_scheme": result.scheme,
        "perm_observed": result.observed,
        "perm_p_value": result.p_value,
        "perm_z_score": result.z_score,
        "perm_null_mean": result.null_mean,
        "perm_null_std": result.null_std,
        "perm_block_length": result.block_length,
        "perm_n_permutations": result.n_permutations,
        "perm_min_detectable": result.min_detectable_effect,
        "perm_has_power": result.has_power,
        "perm_significant": result.significant,
        "perm_warning_count": len(result.warnings),
    }


def attach_permutation_statistics(
    statistics: dict[str, object],
    daily_df: DataFrame,
    capital: float,
    annual_days: int,
    **kwargs: Any,
) -> dict[str, object]:
    """把检验结果并进已有 statistics dict（原 dict 不改，返回新 dict）。

    kwargs 原样透传给 permutation_statistics（参数表见那边）。

    daily_df 为空 / 列不全 / 任何异常 → 原样返回并写入 perm_error，
    绝不让一个诊断指标把整条回测流程带崩（fail-open）。这是刻意的：
    显著性检验是诊断，不是回测的前置条件，它失败时该退场而不是拖着回测一起死。
    """
    merged = dict(statistics)
    try:
        merged.update(
            permutation_statistics(
                daily_df, capital=capital, annual_days=annual_days, **kwargs
            )
        )
    except (ValueError, KeyError, TypeError) as exc:
        merged["perm_error"] = f"{type(exc).__name__}: {exc}"
    return merged


def gated_statistic(
    statistics: dict[str, object],
    target_key: str = "sharpe_ratio",
    alpha: float = DEFAULT_ALPHA,
    failed_value: float = 0.0,
) -> float:
    """寻优目标的显著性闸：p 值不达标就把目标值压到 failed_value。

    用法（配 run_bf_optimization / run_ga_optimization）：先在自定义的
    OptimizationSetting 目标函数里调 attach_permutation_statistics 拿到 perm_p_value，
    再用本函数产出最终目标值。

    ⚠ 再强调一次（见模块文档第五节）：**这不校正网格挑选带来的选择偏差**。
    它只是把"单看一组参数就不显著"的组合先淘汰掉，缩小网格；
    真要给整个寻优流程一个 p 值，必须用 selection_bias_corrected_test。
    """
    p_value = statistics.get("perm_p_value")
    target = statistics.get(target_key)
    if not isinstance(target, int | float):
        return failed_value
    # 没有功效的检验，其 p 值再小也不算数（例如 sharpe×RETURNS_BLOCK 的复利残差
    # 能随机给出 p=0.01）。缺该键时按"有功效"处理，保持对旧字典的向后兼容。
    if statistics.get("perm_has_power", True) is False:
        return failed_value
    if not isinstance(p_value, int | float) or p_value > alpha:
        return failed_value
    return float(target)
