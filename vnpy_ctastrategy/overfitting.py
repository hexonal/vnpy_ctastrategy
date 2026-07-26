"""样本外验证：Walk-Forward 分析 与 CSCV / PBO（回测过拟合概率）。

本模块回答的问题与 `robust_metrics.py` 正交：

    robust_metrics  ——  这条曲线【形状】好不好？（样本内描述统计）
    overfitting     ——  这条曲线是不是【挑出来的运气】？（选择过程的样本外检验）

两块内容：

1. Walk-Forward（滚动样本内选参 → 相邻样本外验证）
   把样本切成若干 (训练窗, 测试窗) 对，在训练窗上跑完整参数网格选出最优参数，
   把该参数原封不动地拿到【紧接其后、从未参与选参】的测试窗上回测，
   最后把所有测试窗的逐日盈亏首尾相接成一条【样本外净值曲线】。
   这条曲线才是"这套流程如果当年真的照做，会得到什么"的近似。

   衰减度量：
     - walk_forward_efficiency = 样本外年化 / 样本内年化（Pardo）。<0.5 视为严重衰减。
     - parameter_stability     = 各折选出的参数有多一致。每折都换一套参数 = 在拟合噪声。
     - 样本外 Sharpe 的标准误 / t 值 / block-bootstrap p 值 —— 直接回答"这是不是运气"。

2. CSCV → PBO（Bailey, Borwein, López de Prado, Zhu 2014,
   "The Probability of Backtest Overfitting", Journal of Computational Finance）
   给定 N 组参数各自在全样本上的逐期收益矩阵 M (T×N)：
     - 把 T 期按时序切成 S 个等长块；
     - 枚举全部 C(S, S/2) 种"取一半块当样本内、剩下一半当样本外"的组合（组合对称）；
     - 每个组合里选出样本内最优的那一列 n*，看它在样本外的排名 ω ∈ (0,1)；
     - λ = ln(ω/(1−ω))；PBO = P(λ ≤ 0)，即"样本内最优在样本外掉到后半段"的概率。

   PBO 的零假设是 0.5：若选参过程完全不携带信息，样本外排名均匀分布，PBO = 0.5。
   PBO 显著低于 0.5 才说明"样本内最优"这件事有信息量；PBO > 0.5 说明选参【反向有害】。

━━━ 使用前必读：本模块的两个诚实边界 ━━━

* **PBO 与 Walk-Forward 都不能把没有 alpha 的策略变成有 alpha。** 它们是否决工具：
  只能告诉你"这个结果站不住"，不能告诉你"这个结果站得住"。
  PBO 低 + Walk-Forward 显著，也只是【没被这两关拦下】，不等于样本外一定赚钱。

* **600 日单标的日线是低功效样本。** 见 `sample_size_diagnosis()`：
  Sharpe 在 300 日子样本上的标准误约 0.9（年化单位），单标的海龟 600 日通常只有
  十几到几十笔完整交易。在这种样本上 PBO 会被噪声拉向 0.5 ——
  **PBO ≈ 0.5 在低功效下同时兼容"确实过拟合"与"信息量不足"两种解释，不可单独定罪。**
  必须与 `sample_size_diagnosis()` 的诊断字段一起读。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from typing import Any

import numpy as np
from pandas import DataFrame, Series, concat
from scipy.stats import norm, rankdata
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData
from vnpy.trader.optimize import OptimizationSetting
from vnpy_gatewaykit.query_window import localize_bound, query_tz

from .backtesting import BacktestingEngine, load_bar_data
from .base import BacktestingMode
from .template import CtaTemplate

# ══════════════════════════════════════════════════════════════════════
# 0. 基础统计量 —— 口径必须与 backtesting.calculate_statistics 完全一致
# ══════════════════════════════════════════════════════════════════════

def daily_log_returns(net_pnl: Sequence[float] | np.ndarray, capital: float) -> np.ndarray:
    """逐日净盈亏 → 逐日对数收益，口径与 `calculate_statistics` 的 df["return"] 逐字一致。

    vnpy 的做法（backtesting.py）：
        balance = net_pnl.cumsum() + capital
        pre_balance = balance.shift(1); pre_balance[0] = capital
        x = balance / pre_balance;  x[x <= 0] = nan
        return = log(x).fillna(0)

    这里必须复刻而不是另起炉灶：一旦口径不同，PBO 里的 Sharpe 和面板上的
    sharpe_ratio 就不是同一个数，判据无法互相印证。测试 `test_daily_log_returns_
    matches_engine` 会直接拿引擎的输出对齐本函数。
    """
    if capital <= 0:
        raise ValueError(f"capital 必须为正，收到 {capital}")

    pnl = np.asarray(net_pnl, dtype=float)
    if pnl.size == 0:
        return np.zeros(0, dtype=float)

    balance = np.cumsum(pnl) + capital
    pre_balance = np.empty_like(balance)
    pre_balance[0] = capital
    pre_balance[1:] = balance[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        x = balance / pre_balance
        x = np.where(x <= 0, np.nan, x)
        out = np.log(x)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def annualised_sharpe(returns: np.ndarray, annual_days: int) -> float:
    """年化 Sharpe。std 为 0（例如整段没有交易）时返回 0.0。

    返回 0.0 而不是 nan，是为了和 vnpy 的 `if return_std: ... else: sharpe_ratio = 0`
    保持一致 —— 一个从不交易的参数组在排名里应当是"平庸"，不是"缺失"。
    """
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    std = float(np.std(r, ddof=1))
    if std == 0.0 or not math.isfinite(std):
        return 0.0
    return float(np.mean(r) / std * math.sqrt(annual_days))


def _sharpe_from_moments(
    count: np.ndarray, s1: np.ndarray, s2: np.ndarray, annual_days: int
) -> np.ndarray:
    """由 (期数, Σr, Σr²) 直接算年化 Sharpe，用于 CSCV 的向量化快路径。

    CSCV 要在 C(16,8)=12870 个组合 × N 组参数上反复算 Sharpe。逐组合拼接子矩阵是
    O(C·T·N)；改用块矩组合则是 O(C·S·N) 的一次矩阵乘法。两条路径在
    `test_cscv_fast_path_matches_generic_path` 里被强制对齐，防止快路径悄悄算错。
    """
    n = count.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s1 / n
        var = (s2 - n * mean * mean) / np.maximum(n - 1.0, 1.0)
        var = np.where(var < 0.0, 0.0, var)      # 浮点误差可能给出 -1e-18
        std = np.sqrt(var)
        sharpe = np.where(std > 0.0, mean / std * math.sqrt(annual_days), 0.0)
    return np.nan_to_num(sharpe, nan=0.0, posinf=0.0, neginf=0.0)


# ══════════════════════════════════════════════════════════════════════
# 1. 显著性：Sharpe 的标准误、t 值、block-bootstrap p 值
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SignificanceResult:
    """一条收益序列"是不是运气"的检验结果。"""

    n_obs: int
    sharpe: float                # 年化
    sharpe_se: float             # 年化标准误（Lo 2002 的 iid 近似）
    t_stat: float
    p_normal: float              # 正态近似双侧 p
    p_block_bootstrap: float     # 循环块 bootstrap 双侧 p（对自相关/厚尾稳健）
    block_size: int
    n_bootstrap: int
    significant: bool            # p_block_bootstrap < alpha
    alpha: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_obs": self.n_obs,
            "sharpe": self.sharpe,
            "sharpe_se": self.sharpe_se,
            "t_stat": self.t_stat,
            "p_normal": self.p_normal,
            "p_block_bootstrap": self.p_block_bootstrap,
            "block_size": self.block_size,
            "n_bootstrap": self.n_bootstrap,
            "significant": self.significant,
            "alpha": self.alpha,
        }


def sharpe_standard_error(sharpe: float, n_obs: int, annual_days: int) -> float:
    """年化 Sharpe 的标准误，Lo (2002) 的 iid 近似。

        SE(SR_daily) = sqrt((1 + 0.5·SR_daily²) / T)
        SE(SR_annual) = SE(SR_daily) · sqrt(annual_days)

    对本项目那条 134 日、Sharpe = −1.68 的曲线（annual_days=240）：
    SR_daily = −1.68/√240 = −0.1084，SE_daily = √(1.00588/134) = 0.0866，
    年化 SE = 0.0866·√240 = **1.34** —— 与实测的 ~1.36 吻合。
    |t| = 1.68/1.34 = 1.25 < 1.96：连符号都没到 2σ。这正是现有面板缺的那一行。
    """
    if n_obs < 2 or annual_days <= 0:
        return float("nan")
    sr_daily = sharpe / math.sqrt(annual_days)
    return math.sqrt((1.0 + 0.5 * sr_daily * sr_daily) / n_obs) * math.sqrt(annual_days)


def _rowwise_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """(C, N) 两个矩阵逐行求 Pearson 相关，返回均值。传入秩即得 Spearman。

    逐行做而不是把所有组合摊平成一个大样本 —— 摊平会把"组合之间的水平差异"
    也算进相关里，那不是我们要问的问题。我们问的是：**在同一个组合内部**，
    样本内排名靠前的参数，样本外是不是也排名靠前。
    """
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        rows = np.where(den > 0, num / den, 0.0)
    return float(np.mean(np.nan_to_num(rows, nan=0.0)))


def _sharpe_rows(samples: np.ndarray, annual_days: int) -> np.ndarray:
    """对 (B, T) 的每一行算年化 Sharpe，向量化。"""
    mean = samples.mean(axis=1)
    std = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(std > 0.0, mean / std * math.sqrt(annual_days), 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def assess_significance(
    returns: Sequence[float] | np.ndarray,
    annual_days: int = 252,
    n_bootstrap: int = 4000,
    block_size: int | None = None,
    alpha: float = 0.05,
    seed: int = 20260725,
) -> SignificanceResult:
    """检验一条日收益序列的 Sharpe 是否显著异于 0。

    零假设 H0：日收益均值为 0（Sharpe = 0）。
    检验统计量：年化 Sharpe。

    给两个 p 值，因为两者的失效模式不同：
      * `p_normal`      —— Lo (2002) iid 近似，快，但收益自相关（趋势策略持仓跨日、
                           波动率聚集）时会【低估】标准误，从而高估显著性。
      * `p_block_bootstrap` —— 把序列去均值后按【循环块】重采样，块长默认 T^(1/3)，
                           保留块内的自相关与厚尾结构。这是判据采用的那个 p 值。

    块长默认 T^(1/3)（T=300 → 7 日）是块 bootstrap 的常规经验取值；持仓周期明显更长的
    策略（海龟日线常见 20-40 日）应显式传入 `block_size ≈ 中位持仓天数`，否则块内
    结构切断、p 值仍偏乐观。
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = int(r.size)

    if n < 10:
        # 样本太短，任何检验都没有意义；显式返回 nan 而不是给一个假的 p 值
        return SignificanceResult(
            n_obs=n, sharpe=annualised_sharpe(r, annual_days), sharpe_se=float("nan"),
            t_stat=float("nan"), p_normal=float("nan"), p_block_bootstrap=float("nan"),
            block_size=0, n_bootstrap=0, significant=False, alpha=alpha,
        )

    sharpe = annualised_sharpe(r, annual_days)
    se = sharpe_standard_error(sharpe, n, annual_days)
    t_stat = sharpe / se if se and math.isfinite(se) and se > 0 else float("nan")
    p_normal = (
        float(2.0 * (1.0 - norm.cdf(abs(t_stat)))) if math.isfinite(t_stat) else float("nan")
    )

    if block_size is None:
        block_size = max(1, int(round(n ** (1.0 / 3.0))))
    block_size = max(1, min(block_size, n))

    centred = r - r.mean()          # 强制 H0 成立
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block_size))
    starts = rng.integers(0, n, size=(n_bootstrap, n_blocks))
    offsets = np.arange(block_size)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_bootstrap, -1)[:, :n] % n
    boot = _sharpe_rows(centred[idx], annual_days)

    p_block = float((1 + np.count_nonzero(np.abs(boot) >= abs(sharpe))) / (n_bootstrap + 1))

    return SignificanceResult(
        n_obs=n, sharpe=sharpe, sharpe_se=se, t_stat=t_stat, p_normal=p_normal,
        p_block_bootstrap=p_block, block_size=block_size, n_bootstrap=n_bootstrap,
        significant=bool(p_block < alpha and sharpe > 0), alpha=alpha,
    )


# ══════════════════════════════════════════════════════════════════════
# 2. CSCV / PBO
# ══════════════════════════════════════════════════════════════════════

MAX_COMBINATIONS: int = 200_000


@dataclass(frozen=True)
class PBOResult:
    """一次 CSCV 的完整结果。字段全部保留，便于事后复查而不必重跑。"""

    pbo: float
    n_configs: int
    n_blocks: int
    block_size: int
    n_obs_used: int
    n_combinations: int
    offset: int

    logits: np.ndarray                    # 每个组合的 λ
    relative_ranks: np.ndarray            # 每个组合的 ω ∈ (0,1)
    is_perf_selected: np.ndarray          # 被选中列的样本内表现
    oos_perf_selected: np.ndarray         # 被选中列的样本外表现
    selected_counts: np.ndarray           # 每列被选为样本内最优的次数

    rank_ic: float                        # 每个组合内 IS/OOS 绩效的截面 Spearman 相关，取均值
    degradation_slope: float              # OOS ~ a + b·IS 的 b（见下方注意）
    degradation_intercept: float
    prob_oos_loss: float                  # 被选中列样本外表现 < 0 的组合占比
    median_oos_selected: float
    median_oos_all: float
    tie_fraction: float                   # λ 恰为 0（排名并列在中位）的组合占比

    warnings: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return pbo_verdict(self.pbo)

    def as_dict(self) -> dict[str, Any]:
        """只含标量，便于落 CSV / JSON。数组字段请直接访问属性。"""
        return {
            "pbo": self.pbo,
            "verdict": self.verdict,
            "n_configs": self.n_configs,
            "n_blocks": self.n_blocks,
            "block_size": self.block_size,
            "n_obs_used": self.n_obs_used,
            "n_combinations": self.n_combinations,
            "offset": self.offset,
            "rank_ic": self.rank_ic,
            "degradation_slope": self.degradation_slope,
            "degradation_intercept": self.degradation_intercept,
            "prob_oos_loss": self.prob_oos_loss,
            "median_oos_selected": self.median_oos_selected,
            "median_oos_all": self.median_oos_all,
            "tie_fraction": self.tie_fraction,
            "warnings": list(self.warnings),
        }


def pbo_verdict(pbo: float) -> str:
    """PBO 判据。

    Bailey et al. 原文只给一条线：PBO > 0.5 = 明确过拟合（选参比抛硬币还差）。
    本项目按"能不能拿这套参数去下真钱"收紧成四档 —— 因为 PBO 是概率不是显著性：
    PBO = 0.4 意味着"有 40% 的可能你选出的参数样本外落在中位以下"，
    这个赔率不足以支撑把真钱押在【优化出来的】参数上。

        ≤ 0.10   选参携带信息，可用（仍需 Walk-Forward 复核）
        ≤ 0.25   边缘，减半仓 + 必须有参数高原佐证
        ≤ 0.50   不可采信优化结果 —— 改用高原中心参数或教科书默认值
        > 0.50   选参反向有害，该参数空间整体拒绝

    低功效警告：600 日单标的样本下 PBO 天然向 0.5 靠拢（见 sample_size_diagnosis）。
    此时"≤0.50 不可采信"这一档的正确读法是**"数据不足以支持优化"**，
    而不是"策略一定过拟合"。两种情况的处置相同（不用优化出来的参数），结论不同。
    """
    if not math.isfinite(pbo):
        return "无法判定"
    if pbo <= 0.10:
        return "稳健：选参携带信息"
    if pbo <= 0.25:
        return "边缘：减半仓，需高原佐证"
    if pbo <= 0.50:
        return "不可采信：勿用优化参数"
    return "反向有害：拒绝该参数空间"


def recommend_n_blocks(
    n_obs: int, min_block_obs: int = 20, max_blocks: int = 16
) -> tuple[int, list[str]]:
    """给定样本长度，推荐 CSCV 的块数 S（偶数）。

    约束：
      1. S 必须是偶数（要对半分 IS/OOS）。
      2. 每块长度 floor(n_obs/S) ≥ min_block_obs。
         `min_block_obs` 应取【约 1 倍中位持仓天数】—— 块比持仓还短，
         同一笔交易会被切到两个块里，块与块之间不再近似独立。
      3. S ≤ max_blocks（16 是 Bailey 的推荐值，C(16,8)=12870，再大组合数爆炸且边际收益极小）。

    返回 (S, 警告列表)。凑不满 min_block_obs 时返回能给的最大偶数 S 并附警告，
    而不是抛异常 —— 让调用方看到"我降级了"，比直接失败更有用。
    """
    notes: list[str] = []
    if n_obs < 40:
        return 0, [f"样本仅 {n_obs} 期，无法做 CSCV（至少需要 40 期）"]

    max_by_block = n_obs // max(1, min_block_obs)
    s = min(max_blocks, max_by_block)
    s -= s % 2                                  # 向下取到偶数

    if s < 4:
        s = 4
        notes.append(
            f"样本 {n_obs} 期 / 最小块长 {min_block_obs} 期只够 {max_by_block} 块，"
            f"已强制 S=4（C(4,2)=6 个组合），PBO 分辨率极低，仅作参考"
        )
    elif s < max_blocks:
        notes.append(
            f"因最小块长 {min_block_obs} 期的约束，S 从 {max_blocks} 降到 {s}"
            f"（块长 {n_obs // s} 期，组合数 {math.comb(s, s // 2)}）"
        )
    return s, notes


def cscv_pbo(
    matrix: np.ndarray,
    n_blocks: int = 16,
    annual_days: int = 252,
    offset: int = 0,
    performance: Callable[[np.ndarray], np.ndarray] | None = None,
    max_combinations: int = MAX_COMBINATIONS,
) -> PBOResult:
    """CSCV（组合对称交叉验证）→ PBO。

    matrix : (T, N)，第 n 列是第 n 组参数的逐期收益（建议用对数收益，与面板同口径）
    n_blocks : S，必须是 ≥2 的偶数
    offset : 块网格的起始偏移。块边界位置是任意的，改变 offset 会给出不同的 PBO 估计；
             用 `cscv_pbo_stability()` 扫多个 offset 看估计本身有多稳。
    performance : 自定义绩效函数，输入 (rows, N) 子矩阵返回 (N,) 绩效。
                  留空则用年化 Sharpe（走块矩快路径，快约 30 倍）。

    并列（tie）的处理 —— 与原文的一处显式偏离：
        原文定义 PBO = P(λ ≤ 0)。当多列样本外表现完全相同（例如全部参数都没交易，
        或参数网格里有重复配置），平均秩恰为 (N+1)/2 → ω = 0.5 → λ = 0，
        按 "≤" 会被【全部记为过拟合】，给出 PBO = 1.0 —— 这是明显的误判：
        完全无区分度应当对应"无信息" = 0.5。
        本实现用 PBO = P(λ<0) + 0.5·P(λ=0)。连续情形下 P(λ=0)=0，与原文完全等价；
        只在退化情形下给出更诚实的答案。`tie_fraction` 字段暴露这一项的占比。
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"matrix 必须是二维 (T, N)，收到 shape={m.shape}")
    if not np.isfinite(m).all():
        raise ValueError("matrix 含 nan/inf —— 请在传入前处理缺失（无交易日应填 0，不是 nan）")

    n_obs_total, n_configs = m.shape
    if n_configs < 2:
        raise ValueError(f"至少需要 2 组参数才能排名，收到 {n_configs}")
    if n_blocks < 2 or n_blocks % 2 != 0:
        raise ValueError(f"n_blocks 必须是 ≥2 的偶数，收到 {n_blocks}")

    warnings: list[str] = []
    if n_configs < 10:
        warnings.append(
            f"仅 {n_configs} 组参数：秩的分辨率为 1/{n_configs + 1}，PBO 只能取少数几个值，"
            f"建议 ≥ 20 组"
        )

    n_combinations = math.comb(n_blocks, n_blocks // 2)
    if n_combinations > max_combinations:
        raise ValueError(
            f"C({n_blocks},{n_blocks // 2}) = {n_combinations} 超过上限 {max_combinations}，"
            f"请减小 n_blocks"
        )

    block_size = (n_obs_total - offset) // n_blocks
    if block_size < 1:
        raise ValueError(f"offset={offset} 下块长为 0（样本 {n_obs_total} 期 / {n_blocks} 块）")
    n_used = block_size * n_blocks
    if offset < 0 or offset + n_used > n_obs_total:
        raise ValueError(f"offset={offset} 越界（样本 {n_obs_total} 期，需要 {n_used} 期）")

    used = m[offset: offset + n_used]
    blocks = used.reshape(n_blocks, block_size, n_configs)

    if block_size < 10:
        warnings.append(f"块长仅 {block_size} 期，块内绩效估计几乎全是噪声")

    # ── 枚举 C(S, S/2) 个组合，做成 (C, S) 的 0/1 掩码 ──
    half = n_blocks // 2
    mask = np.zeros((n_combinations, n_blocks), dtype=bool)
    for i, combo in enumerate(combinations(range(n_blocks), half)):
        mask[i, list(combo)] = True

    if performance is None:
        # 快路径：块矩 → 组合矩 → Sharpe，一次矩阵乘法搞定所有组合
        s1 = blocks.sum(axis=1)                                   # (S, N)
        s2 = (blocks * blocks).sum(axis=1)                        # (S, N)
        cnt = np.full(n_blocks, block_size, dtype=float)

        fm = mask.astype(float)
        is_cnt = (fm @ cnt)[:, None]                              # (C, 1)
        oos_cnt = ((~mask).astype(float) @ cnt)[:, None]
        is_perf = _sharpe_from_moments(is_cnt, fm @ s1, fm @ s2, annual_days)
        oos_perf = _sharpe_from_moments(
            oos_cnt, (~mask).astype(float) @ s1, (~mask).astype(float) @ s2, annual_days
        )
    else:
        # 通用路径：逐组合拼接子矩阵。慢，但支持任意绩效定义（如 R³、Calmar）
        is_perf = np.empty((n_combinations, n_configs), dtype=float)
        oos_perf = np.empty((n_combinations, n_configs), dtype=float)
        for i in range(n_combinations):
            sel = mask[i]
            is_perf[i] = performance(blocks[sel].reshape(-1, n_configs))
            oos_perf[i] = performance(blocks[~sel].reshape(-1, n_configs))
        is_perf = np.nan_to_num(is_perf, nan=0.0, posinf=0.0, neginf=0.0)
        oos_perf = np.nan_to_num(oos_perf, nan=0.0, posinf=0.0, neginf=0.0)

    # ── 样本内最优 → 样本外相对秩 → logit ──
    best = np.argmax(is_perf, axis=1)                             # (C,)
    rows = np.arange(n_combinations)

    ranks = rankdata(oos_perf, method="average", axis=1)          # 1..N，并列取平均
    rank_selected = ranks[rows, best]
    omega = rank_selected / (n_configs + 1.0)
    logits = np.log(omega / (1.0 - omega))

    n_below = int(np.count_nonzero(logits < 0.0))
    n_tie = int(np.count_nonzero(logits == 0.0))
    pbo = (n_below + 0.5 * n_tie) / n_combinations

    is_sel = is_perf[rows, best]
    oos_sel = oos_perf[rows, best]

    # ── 截面 rank IC：每个组合内，把 N 组参数的 IS 绩效与 OOS 绩效做 Spearman 相关 ──
    # 这是比退化回归斜率【可靠得多】的诊断：它问的是"样本内的排名能不能预测样本外的排名"，
    # 完全在截面上做，不受下面那个复杂度的影响。
    #   > 0  样本内排名有预测力（与低 PBO 同向）
    #   ≈ 0  排名纯噪声
    #   < 0  排名系统性反转 = 教科书式过拟合
    is_ranks = rankdata(is_perf, method="average", axis=1)
    ric = _rowwise_correlation(is_ranks, ranks)

    # ── 退化回归：OOS = a + b·IS，跨组合拟合。
    # 注意（一个容易被误读的陷阱）：同一列的 IS 与 OOS 是【互补的两半】，
    # 二者之和恒等于该列的全样本表现，因此即便某组参数确有真实优势，
    # 这条回归的斜率也会被这个恒等式压成负数（实测：单列真实占优时斜率仍为 −1.0）。
    # 所以 **负斜率本身不构成过拟合的证据**，只作为分布形状的描述量保留；
    # 判定请用 PBO 与 rank_ic。
    if np.std(is_sel) > 0:
        slope, intercept = np.polyfit(is_sel, oos_sel, 1)
    else:
        slope, intercept = 0.0, float(np.mean(oos_sel))

    counts = np.bincount(best, minlength=n_configs)

    return PBOResult(
        pbo=float(pbo),
        n_configs=n_configs,
        n_blocks=n_blocks,
        block_size=block_size,
        n_obs_used=n_used,
        n_combinations=n_combinations,
        offset=offset,
        logits=logits,
        relative_ranks=omega,
        is_perf_selected=is_sel,
        oos_perf_selected=oos_sel,
        selected_counts=counts,
        rank_ic=ric,
        degradation_slope=float(slope),
        degradation_intercept=float(intercept),
        prob_oos_loss=float(np.count_nonzero(oos_sel < 0.0) / n_combinations),
        median_oos_selected=float(np.median(oos_sel)),
        median_oos_all=float(np.median(oos_perf)),
        tie_fraction=float(n_tie / n_combinations),
        warnings=warnings,
    )


def cscv_pbo_stability(
    matrix: np.ndarray,
    n_blocks: int = 16,
    annual_days: int = 252,
    n_offsets: int = 8,
    **kwargs: Any,
) -> tuple[list[PBOResult], dict[str, float]]:
    """在多个块网格相位上重复 CSCV，看 PBO 估计本身有多稳。

    块边界落在哪里是【任意的】。600 日 / 16 块时块长 38 天，把网格整体平移几天
    就会把不同的交易切进不同的块 —— 如果 PBO 因此从 0.3 跳到 0.6，
    那么"PBO=0.3"这个数字本身就不可报。这是小样本下最容易被忽略的一层不确定性。

    返回 (每个相位的结果列表, {min/median/max/spread})。
    """
    m = np.asarray(matrix, dtype=float)
    n_obs = m.shape[0]

    n_offsets = max(1, n_offsets)
    block_size = (n_obs - (n_offsets - 1)) // n_blocks
    if block_size < 1:
        n_offsets = 1
        block_size = n_obs // n_blocks
    if block_size < 1:
        raise ValueError(f"样本 {n_obs} 期切不出 {n_blocks} 块")

    results: list[PBOResult] = []
    for offset in range(n_offsets):
        trimmed = m[offset: offset + block_size * n_blocks]
        results.append(cscv_pbo(trimmed, n_blocks, annual_days, offset=0, **kwargs))

    values = np.array([r.pbo for r in results], dtype=float)
    summary = {
        "pbo_min": float(values.min()),
        "pbo_median": float(np.median(values)),
        "pbo_max": float(values.max()),
        "pbo_spread": float(values.max() - values.min()),
        "n_offsets": float(len(results)),
    }
    return results, summary


def average_config_correlation(matrix: np.ndarray) -> float:
    """参数组之间收益序列的平均两两相关系数。

    参数网格上的相邻配置往往高度相关（同一套突破系统、窗口只差几天），
    ρ 常在 0.8-0.99。这个数字本身是有用的诊断：ρ→1 说明网格里其实只有一个策略，
    "N 组参数"的多样性是假的，PBO 的排名分辨率也随之下降。
    """
    m = np.asarray(matrix, dtype=float)
    if m.shape[1] < 2:
        return float("nan")
    std = m.std(axis=0)
    keep = std > 0
    if keep.sum() < 2:
        return float("nan")
    corr = np.corrcoef(m[:, keep], rowvar=False)
    iu = np.triu_indices_from(corr, k=1)
    return float(np.nanmean(corr[iu]))


@dataclass(frozen=True)
class PBONull:
    """PBO 在"选参完全不携带信息"这一零假设下的模拟分布。"""

    n_sims: int
    n_obs: int
    n_configs: int
    n_blocks: int
    correlation: float
    mean: float
    sd: float
    q05: float
    q50: float
    q95: float
    values: np.ndarray

    def p_value(self, pbo: float) -> float:
        """单侧 p 值：零分布中有多大比例的 PBO ≤ 观测值。

        PBO 越低越好，所以"低得足以排除运气"= p 小。
        p ≥ 0.05 意味着：**这个 PBO 完全可能是没有任何选参能力时随机产生的**。
        """
        if not math.isfinite(pbo) or self.values.size == 0:
            return float("nan")
        return float((1 + np.count_nonzero(self.values <= pbo)) / (self.values.size + 1))

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_sims": self.n_sims, "n_obs": self.n_obs, "n_configs": self.n_configs,
            "n_blocks": self.n_blocks, "correlation": self.correlation,
            "mean": self.mean, "sd": self.sd, "q05": self.q05, "q50": self.q50, "q95": self.q95,
        }


def pbo_null_distribution(
    n_obs: int,
    n_configs: int,
    n_blocks: int = 16,
    annual_days: int = 252,
    correlation: float = 0.0,
    n_sims: int = 200,
    seed: int = 20260725,
) -> PBONull:
    """模拟 PBO 的零分布 —— 本模块最重要的一个函数。

    **为什么必须做这一步**：教科书说"PBO 的零假设是 0.5"，那是【期望值】。
    单次估计的离散度在小样本上极大。实测（T=611, N=50, S=16, 纯噪声）：

        PBO 零分布 ≈ 均值 0.45，标准差 0.157，5% 分位 0.21

    也就是说，一个 **PBO = 0.30 的回测，在完全没有任何选参能力的情况下也有约 20%
    的概率出现**。把 0.30 当成"通过"是把噪声当成了证据。
    没有这条零分布，PBO 就只是"一个数字对一条民间阈值"，谈不上检验。

    零假设的构造：N 列收益同分布、无任何真实优劣差异，用等相关高斯因子模型
    r[t,n] = √ρ·f[t] + √(1−ρ)·e[t,n] 生成。实测 ρ 从 0 到 0.98 对零分布几乎无影响
    （共同因子对所有列同向移动，不改变排名），所以 ρ 只影响诊断解读、不影响临界值；
    保留该参数是为了让调用方能用实测 ρ 复核这一点，而不是让人相信一句断言。

    尺度不影响结果（Sharpe 是尺度无关的），故直接用单位方差生成。
    """
    if n_obs < n_blocks or n_configs < 2:
        raise ValueError(f"n_obs={n_obs} / n_configs={n_configs} 不足以模拟零分布")
    rho = min(max(float(correlation), 0.0), 0.999)
    rng = np.random.default_rng(seed)

    values = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        factor = rng.standard_normal((n_obs, 1))
        idio = rng.standard_normal((n_obs, n_configs))
        sim = math.sqrt(rho) * factor + math.sqrt(1.0 - rho) * idio
        values[i] = cscv_pbo(sim, n_blocks=n_blocks, annual_days=annual_days).pbo

    return PBONull(
        n_sims=n_sims, n_obs=n_obs, n_configs=n_configs, n_blocks=n_blocks, correlation=rho,
        mean=float(values.mean()), sd=float(values.std(ddof=1)),
        q05=float(np.quantile(values, 0.05)), q50=float(np.quantile(values, 0.50)),
        q95=float(np.quantile(values, 0.95)), values=values,
    )


def pbo_verdict_calibrated(pbo: float, null: PBONull, alpha: float = 0.05) -> str:
    """用模拟零分布判定，取代对 0.5 的直觉比较。

    这是本模块推荐的判据（`pbo_verdict` 的固定阈值只在没跑零分布时兜底）：

        p ≥ alpha              → 不通过：PBO 低得不够，无法与"纯运气"区分
        p < alpha 且 PBO ≤ 0.25 → 通过：选参携带信息
        p < alpha 且 PBO > 0.25 → 边缘：统计上不同于零分布，但绝对水平仍高，减半仓
        PBO > 零分布 95% 分位   → 反向有害：比抛硬币还差
    """
    p = null.p_value(pbo)
    if not math.isfinite(p):
        return "无法判定"
    if pbo > null.q95:
        return f"反向有害：PBO={pbo:.3f} 高于零分布 95% 分位 {null.q95:.3f}"
    if p >= alpha and pbo > null.q50:
        return (
            f"不通过：PBO={pbo:.3f} 高于零分布中位 {null.q50:.3f}（p={p:.3f}）—— "
            f"选参无正贡献"
        )
    if p >= alpha:
        return (
            f"不通过：PBO={pbo:.3f}，零分布下 p={p:.3f} ≥ {alpha} —— "
            f"与「选参毫无信息」不可区分"
        )
    if pbo <= 0.25:
        return f"通过：PBO={pbo:.3f}，p={p:.3f}，显著优于零分布"
    return f"边缘：PBO={pbo:.3f} 显著低于零分布(p={p:.3f})但绝对水平仍高，减半仓"


# ══════════════════════════════════════════════════════════════════════
# 3. 小样本诊断 —— 回答"600 日到底够不够"
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SampleDiagnosis:
    """样本是否支撑得起这次检验。所有字段都是可核对的数字，不是形容词。"""

    n_obs: int
    n_blocks: int
    block_size: int
    n_combinations: int
    obs_per_side: int                    # IS / OOS 各自的期数
    sharpe_se_per_side: float            # 单侧子样本上 Sharpe 的标准误（年化）
    min_detectable_sharpe: float         # 该样本能以 α=5% / 功效 80% 检出的最小 Sharpe
    trades_total: int | None
    trades_per_side: float | None
    block_vs_holding: float | None       # 块长 / 中位持仓天数
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_obs": self.n_obs,
            "n_blocks": self.n_blocks,
            "block_size": self.block_size,
            "n_combinations": self.n_combinations,
            "obs_per_side": self.obs_per_side,
            "sharpe_se_per_side": self.sharpe_se_per_side,
            "min_detectable_sharpe": self.min_detectable_sharpe,
            "trades_total": self.trades_total,
            "trades_per_side": self.trades_per_side,
            "block_vs_holding": self.block_vs_holding,
            "notes": list(self.notes),
        }


def sample_size_diagnosis(
    n_obs: int,
    n_blocks: int = 16,
    annual_days: int = 252,
    trades_total: int | None = None,
    median_holding_days: float | None = None,
) -> SampleDiagnosis:
    """回答"600 日样本切 S 块够不够"，用数字而不是印象。

    关键区分（很多实现在这里含糊）：
      * **块长不是估计误差的来源** —— IS/OOS 各拿 S/2 块，各占样本的一半。
        600 日 / S=16 → 块长 38 天，但 IS 和 OOS 各是 8×38 = 304 天，不是 38 天。
        所以"每块只有 37 天够不够"的正确答法是：估计精度由 **304 天** 决定，
        块长只决定 (a) 序列结构是否被切碎 (b) 组合数。
      * **真正的约束是独立交易笔数，不是天数。** 单标的日线海龟 600 天通常
        只有 15-30 笔完整交易；对半分后每侧 8-15 笔。Sharpe 建立在 ~10 笔上，
        排名基本由噪声决定 —— 这才是 600 日样本的硬墙。

    `min_detectable_sharpe` = (z_{0.975} + z_{0.80}) × SE ≈ 2.80 × SE：
    该子样本长度下，要在 5% 显著水平上有 80% 把握检出，真实年化 Sharpe 至少得多大。
    304 天 / 252 年化日 → SE ≈ 0.91 → 最小可检出 Sharpe ≈ 2.5。
    **真实 Sharpe 低于 2.5 的策略，在这个样本上根本无法被可靠区分。**
    """
    notes: list[str] = []
    block_size = n_obs // n_blocks if n_blocks else 0
    obs_per_side = block_size * (n_blocks // 2) if n_blocks else 0
    n_comb = math.comb(n_blocks, n_blocks // 2) if n_blocks >= 2 else 0

    se = sharpe_standard_error(0.0, obs_per_side, annual_days) if obs_per_side >= 2 else float("nan")
    mds = 2.802 * se if math.isfinite(se) else float("nan")

    if math.isfinite(mds):
        notes.append(
            f"IS/OOS 各 {obs_per_side} 期 → Sharpe 标准误 {se:.2f}（年化）；"
            f"能以 80% 功效检出的最小真实 Sharpe ≈ {mds:.2f}"
        )
        if mds > 1.5:
            notes.append(
                f"最小可检出 Sharpe {mds:.2f} 高于绝大多数真实 CTA 的水平（0.5-1.0）——"
                f"本样本处于低功效区，PBO 会被噪声拉向 0.5，"
                f"**PBO≈0.5 不能单独定罪为过拟合，也可能只是信息量不足**"
            )

    trades_per_side: float | None = None
    if trades_total is not None:
        trades_per_side = trades_total / 2.0
        notes.append(f"全样本约 {trades_total} 笔完整交易 → IS/OOS 各约 {trades_per_side:.0f} 笔")
        if trades_per_side < 20:
            notes.append(
                f"每侧仅约 {trades_per_side:.0f} 笔交易（经验下限 20 笔）："
                f"排名主要由少数几笔的运气决定，CSCV 的分辨率受限于此，与天数无关"
            )

    ratio: float | None = None
    if median_holding_days:
        ratio = block_size / median_holding_days
        notes.append(f"块长 {block_size} 天 / 中位持仓 {median_holding_days:.0f} 天 = {ratio:.1f}×")
        if ratio < 2.0:
            notes.append(
                f"块长仅为持仓周期的 {ratio:.1f} 倍（要求 ≥2×）：同一笔交易会被块边界切断，"
                f"块间不再近似独立，应减小 S"
            )

    return SampleDiagnosis(
        n_obs=n_obs, n_blocks=n_blocks, block_size=block_size, n_combinations=n_comb,
        obs_per_side=obs_per_side, sharpe_se_per_side=se, min_detectable_sharpe=mds,
        trades_total=trades_total, trades_per_side=trades_per_side,
        block_vs_holding=ratio, notes=notes,
    )


# ══════════════════════════════════════════════════════════════════════
# 4. 回测执行器 —— 把 (参数, 起止) 变成 daily_df
# ══════════════════════════════════════════════════════════════════════

def _naive(moment: datetime) -> datetime:
    """去掉时区信息。

    **只在"两个时刻本来就出自同一个时钟"时才可用**（例如同为调用方手打的裸
    边界）。跨时钟比较请走 `EngineRunner._instant` —— 剥时区不是"忽略时区"，
    而是"把对方的时钟当成自己的墙钟读"，这正是窗口整体错位一个交易日的病根。
    """
    return moment.replace(tzinfo=None) if moment.tzinfo is not None else moment


class _QuietBacktestingEngine(BacktestingEngine):
    """静音版引擎。Walk-Forward 一次要跑上百次回测，逐条打印进度条毫无用处。"""

    def output(self, msg: str) -> None:
        pass


BacktestRunner = Callable[[dict, datetime, datetime], DataFrame]
"""(参数字典, 起始, 结束) -> daily_df。测试用假 runner 注入，生产用 EngineRunner。"""


class EngineRunner:
    """默认执行器：真的跑一次 `BacktestingEngine`，返回 `calculate_result()` 的 daily_df。

    做了一处内部优化：整段历史 K 线只从数据库取一次，之后按窗口切片直接塞进
    `engine.history_data`，跳过 `engine.load_data()`。Walk-Forward 要跑几百次回测，
    每次都重查数据库会让整个流程慢一个数量级。`cache_bars=False` 可关掉回到官方路径。

    **warmup_bars（短窗口回测必设）**：把回测起点向前多推 N 根真实 K 线，
    跑完后再把预热段从 daily_df 里剔除。两个作用：

      1. 指标预热。CtaTemplate.load_bar 的默认 interval 是 `Interval.MINUTE`，
         且 days 参数按【自然日】算 —— `LongOnlyTurtleStrategy.on_init` 里的
         `self.load_bar(self.breakout_window + 10)` 实际是在查 65 自然日的【分钟线】。
         本项目的 700.SEHK 在该区间没有分钟数据，返回 0 根，于是 ArrayManager(size=65)
         在一个 63 根的样本外窗口里【永远 inited 不了，策略一笔都不会交易】。
         全样本回测（627 根）看不出这个问题，Walk-Forward 一切短窗口就暴露了。
      2. 消除折边界的"强制空仓"失真：预热段跑完时若已持仓，该仓位会自然带进测试窗，
         首日只计 (close − pre_close) × start_pos 的持仓盈亏 —— 这比每折从空仓重来
         更接近"真实滚动运行"的情形。

    建议取值 ≥ 策略最长窗口的 2 倍（海龟 55 日 → 120）。
    """

    def __init__(
        self,
        strategy_class: type[CtaTemplate],
        vt_symbol: str,
        interval: Interval,
        rate: float,
        slippage: float,
        size: float,
        pricetick: float,
        capital: int,
        start: datetime,
        end: datetime,
        annual_days: int = 252,
        risk_free: float = 0.0,
        half_life: int = 120,
        mode: BacktestingMode = BacktestingMode.BAR,
        cache_bars: bool = True,
        warmup_bars: int = 0,
    ) -> None:
        self.strategy_class = strategy_class
        self.vt_symbol = vt_symbol
        self.interval = interval
        self.rate = rate
        self.slippage = slippage
        self.size = size
        self.pricetick = pricetick
        self.capital = capital
        self.start = start
        self.end = end
        self.annual_days = annual_days
        self.risk_free = risk_free
        self.half_life = half_life
        self.mode = mode
        self.cache_bars = cache_bars
        self.warmup_bars = warmup_bars
        self._bars: list[BarData] | None = None

    # 缓存的 BarData 列表体积可观且可重建，进程间传递时丢弃
    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_bars"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    @property
    def exchange(self) -> Exchange:
        """标的所属交易所 —— 窗口边界的裸 datetime 按它的墙钟读。"""
        return Exchange(self.vt_symbol.rsplit(".", 1)[1])

    def _instant(self, moment: datetime) -> datetime:
        """把一个时刻钉成"哪一个瞬间"，好和别的时刻比较。

        数据库交回的 K 线带 DB_TZ，调用方传进来的窗口边界通常是裸 datetime。
        裸边界的含义是**交易所墙钟**（`query_window` 那套口径，`load_bar_data`
        查询时用的也是它），所以这里给它贴上交易所时区，而不是把 K 线那一侧
        的时区剥掉。两种做法在 `database.timezone` 恰好等于交易所时区时同解；
        本项目 `database.timezone = UTC` 而港股在 UTC+8，剥时区会把港股
        2024-01-26 那根读成 2024-01-25，整个窗口后移一个交易日 ——
        低端丢掉首日、高端吃进 `end` 之后那一根（Walk-Forward 的前视泄漏）。

        已经带时区的时刻原样返回：它本来就指名了一个瞬间。
        """
        return localize_bound(moment, self.exchange)

    def bars(self) -> list[BarData]:
        """整段历史 K 线（惰性加载，进程内只查一次数据库）。"""
        if self._bars is None:
            symbol = self.vt_symbol.rsplit(".", 1)[0]
            self._bars = load_bar_data(
                symbol, self.exchange, self.interval, self.start, self.end
            )
        return self._bars

    def bar_datetimes(self) -> list[datetime]:
        """整段历史的 K 线时间戳，用于按【真实交易日】而不是自然日切分窗口。"""
        return [bar.datetime for bar in self.bars()]

    def new_engine(self, start: datetime, end: datetime) -> BacktestingEngine:
        engine = _QuietBacktestingEngine()
        engine.set_parameters(
            vt_symbol=self.vt_symbol, interval=self.interval, start=start, end=end,
            rate=self.rate, slippage=self.slippage, size=self.size,
            pricetick=self.pricetick, capital=self.capital, mode=self.mode,
            risk_free=self.risk_free, annual_days=self.annual_days, half_life=self.half_life,
        )
        return engine

    def warmup_start(self, start: datetime) -> datetime:
        """把起点向前推 warmup_bars 根真实 K 线（不足则退到样本最开始）。"""
        if self.warmup_bars <= 0:
            return start
        lo = self._instant(start)
        before = [b.datetime for b in self.bars() if self._instant(b.datetime) < lo]
        if not before:
            return start
        return before[-min(self.warmup_bars, len(before))]

    def __call__(self, setting: dict, start: datetime, end: datetime) -> DataFrame:
        run_start = self.warmup_start(start)
        engine = self.new_engine(run_start, end)
        engine.add_strategy(self.strategy_class, dict(setting))

        # K 线时间戳与窗口边界可能出自两个时钟，先钉到瞬间再比（见 _instant）。
        lo = self._instant(run_start)
        hi = self.window_end(end)

        if self.cache_bars:
            engine.history_data = [
                b for b in self.bars() if lo <= self._instant(b.datetime) <= hi
            ]
        if not engine.history_data:
            engine.load_data()
            # load_data 走的是 engine.end，而 set_parameters 会把它撑到当日
            # 23:59:59。日线下那仍是同一根，日内下就是另一个窗口了 —— 两条
            # 取数路径必须交出同一段，否则 cache_bars 开关会改变绩效。
            engine.history_data = [
                b for b in engine.history_data if lo <= self._instant(b.datetime) <= hi
            ]
        if not engine.history_data:
            return DataFrame()

        # 预热段的剔除锚点：窗口自己第一根 K 线的【日期键】。daily_df 的索引是
        # `bar.datetime.date()`，即 K 线自己那个时钟的日期；拿调用方墙钟的
        # `start.date()` 去裁会把窗口首日一起裁掉（港股首日的键是前一日 UTC）。
        window_lo = self._instant(start)
        first_in_window = next(
            (b.datetime for b in engine.history_data if self._instant(b.datetime) >= window_lo),
            None,
        )

        engine.run_backtesting()
        df = engine.calculate_result()
        if df is None or df.empty:
            return DataFrame()
        df = df.copy()
        if lo != window_lo:
            # 预热段只用来把指标和仓位喂热，不计入绩效
            if first_in_window is None:
                return DataFrame()
            cut = first_in_window.date()
            df = df[[d >= cut for d in df.index]]
        return df

    def window_end(self, end: datetime) -> datetime:
        """窗口上界（含）。日线撑到当日收盘，日内按原值取。

        日线的 `end` 通常是一个日期（00:00），意思是"含这一天"，所以要撑到
        23:59:59 才能把当天那根收进来。**"当天"按交易所墙钟算**：`end` 若是
        `bar_datetimes()` 交回的真实 K 线时间戳，它带的是 DB_TZ，直接
        `replace(hour=23)` 撑的是 DB 那个时钟的当天，与查询口径对不上。

        日内周期下 `end` 是一根真实 K 线的时间戳，再撑到 23:59:59 就会把当天
        剩下的 K 线一并收进来 —— `ThreeWaySplit` 的相邻两段若落在同一自然日
        （15m 线下几乎必然），前一段就会吃进后一段的数据，VALID 的绩效里混进
        TEST 段。所以日内一律按原值取。
        """
        moment = self._instant(end)
        if self.interval in (Interval.MINUTE, Interval.HOUR):
            return moment
        return moment.astimezone(query_tz(self.exchange)).replace(
            hour=23, minute=59, second=59
        )

    def statistics(self, df: DataFrame) -> dict:
        """对任意 daily_df（含拼接出来的样本外曲线）算标准 statistics 面板。

        直接复用 `calculate_statistics` —— 拼接曲线因此和普通回测走【同一套指标口径】，
        包括本 fork 加的 RAR / R³ / Robust Sharpe。
        """
        if df is None or df.empty:
            return {}
        engine = self.new_engine(self.start, self.end)
        return engine.calculate_statistics(df=df.copy(), output=False)


def run_settings(
    runner: BacktestRunner,
    settings: Sequence[dict],
    start: datetime,
    end: datetime,
) -> list[DataFrame]:
    """把一组参数在同一窗口上依次回测。串行 —— 日线单标的一次回测只有毫秒级，
    并行带来的进程启动 + pickle 开销通常比收益大。"""
    return [runner(setting, start, end) for setting in settings]


def returns_matrix(
    frames: Sequence[DataFrame], capital: float
) -> tuple[np.ndarray, list[int]]:
    """一组 daily_df → CSCV 需要的 (T, N) 收益矩阵。

    按日期【并集】对齐：某组参数当天没有记录时填 0 收益（当天没交易 = 净值不变），
    这与 vnpy 的逐日盯市口径一致。返回 (矩阵, 被丢弃的空 df 下标)。
    """
    kept: list[tuple[int, DataFrame]] = [
        (i, f) for i, f in enumerate(frames) if f is not None and not f.empty
    ]
    dropped = [i for i, f in enumerate(frames) if f is None or f.empty]
    if not kept:
        return np.zeros((0, 0), dtype=float), dropped

    index = sorted({d for _, f in kept for d in f.index})
    columns: list[np.ndarray] = []
    for _, f in kept:
        pnl = Series(f["net_pnl"].to_numpy(), index=list(f.index)).reindex(index).fillna(0.0)
        columns.append(daily_log_returns(pnl.to_numpy(), capital))
    return np.column_stack(columns), dropped


# ══════════════════════════════════════════════════════════════════════
# 5. Walk-Forward
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class WalkForwardSplit:
    """一折的窗口边界。边界取自【真实 K 线时间戳】，所以 train_bars/test_bars 是交易日数。"""

    index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_bars: int
    test_bars: int


def make_walk_forward_splits(
    bar_datetimes: Sequence[datetime],
    train_bars: int,
    test_bars: int,
    anchored: bool = False,
    step_bars: int | None = None,
) -> list[WalkForwardSplit]:
    """按真实交易日切出 (训练窗, 测试窗) 序列。

    anchored=False（默认，滚动窗）：训练窗长度固定，整体向前滚。
    anchored=True（锚定/扩张窗）：训练窗起点固定在样本最开始，越往后训练数据越多。

    两者都保证 **test_start 严格晚于 train_end，且各测试窗互不重叠** ——
    这是"没有前视泄漏"的全部含义，由 `test_no_leakage_between_train_and_test` 钉住。
    step_bars 默认 = test_bars，即测试窗首尾相接、完整覆盖样本外区间。
    """
    dts = list(bar_datetimes)
    n = len(dts)
    if train_bars < 2 or test_bars < 1:
        raise ValueError(f"train_bars≥2 且 test_bars≥1，收到 {train_bars}/{test_bars}")
    step = test_bars if step_bars is None else step_bars
    if step < 1:
        raise ValueError(f"step_bars 必须 ≥1，收到 {step}")

    splits: list[WalkForwardSplit] = []
    i = 0
    while train_bars + i + test_bars <= n:
        train_lo = 0 if anchored else i
        train_hi = i + train_bars                    # 开区间上界
        test_hi = train_hi + test_bars
        splits.append(
            WalkForwardSplit(
                index=len(splits),
                train_start=dts[train_lo], train_end=dts[train_hi - 1],
                test_start=dts[train_hi], test_end=dts[test_hi - 1],
                train_bars=train_hi - train_lo, test_bars=test_bars,
            )
        )
        i += step
    return splits


Selector = Callable[[Sequence[dict], np.ndarray], int]
"""(参数列表, 各自的样本内目标值) -> 选中的下标。"""


def argmax_selector(settings: Sequence[dict], scores: np.ndarray) -> int:
    """默认选参：取样本内目标值最高者 —— 也就是常规参数寻优在做的事。"""
    del settings
    return int(np.argmax(scores))


def plateau_selector(settings: Sequence[dict], scores: np.ndarray) -> int:
    """高原选参：取【邻域平均目标值】最高者，而不是单点最高。

    动机：参数网格上的单点最高值往往是噪声尖峰 —— 邻居一换就崩，正是过拟合的形状。
    高原中心（自己不算最高、但周围一圈都不差）在样本外远更容易复现。
    当 PBO 判定"不可采信优化参数"时，这是比"退回默认值"更进取的替代方案，
    且可以直接作为 `run_walk_forward(selector=...)` 传入，让 WF 去实测它是否真的衰减更小。

    邻域定义：所有数值型参数上，取值排序后位置相差 ≤1 格；非数值参数必须完全相同。

    **边界惩罚**：网格边缘的参数邻居更少，直接取邻域均值会让边缘点占便宜
    （少几个拖后腿的邻居，均值反而更高）。这里按【完整邻域大小】做分母，
    缺失的邻居按全网格最低分补齐 —— 位于网格边缘的参数本来就无法被证实为高原中心，
    这个惩罚是对的，不是权宜之计。
    """
    n = len(settings)
    if n == 0:
        raise ValueError("settings 为空")
    keys = sorted({k for s in settings for k in s})

    # 每个参数的取值刻度（数值型才有序）
    ladders: dict[str, list[Any]] = {}
    numeric: dict[str, bool] = {}
    for k in keys:
        vals = {s.get(k) for s in settings}
        is_num = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals)
        numeric[k] = is_num
        ladders[k] = sorted(vals) if is_num else sorted(vals, key=repr)

    pos = [
        {k: ladders[k].index(s.get(k)) for k in keys}
        for s in settings
    ]

    full_size = 1
    for k in keys:
        full_size *= min(3, len(ladders[k])) if numeric[k] else 1
    floor = float(np.min(scores))

    smoothed = np.empty(n, dtype=float)
    for i in range(n):
        neigh = [
            j for j in range(n)
            if all(
                (abs(pos[j][k] - pos[i][k]) <= 1) if numeric[k] else (pos[j][k] == pos[i][k])
                for k in keys
            )
        ]
        missing = max(0, full_size - len(neigh))
        smoothed[i] = (float(np.sum(scores[neigh])) + missing * floor) / max(1, full_size)
    return int(np.argmax(smoothed))


@dataclass
class WalkForwardFold:
    """一折的结果。"""

    split: WalkForwardSplit
    chosen_setting: dict
    chosen_index: int
    is_target: float
    oos_target: float
    is_annual_return: float
    oos_annual_return: float
    efficiency: float                 # oos_annual / is_annual，分母近 0 时为 nan
    n_candidates: int
    oos_daily_df: DataFrame
    oos_statistics: dict


@dataclass
class WalkForwardReport:
    """整套 Walk-Forward 的结论。"""

    target_name: str
    folds: list[WalkForwardFold]
    oos_daily_df: DataFrame
    oos_statistics: dict
    significance: SignificanceResult
    efficiency_median: float
    efficiency_mean: float
    parameter_stability: float             # 0-1，各参数"取众数的折数占比"的平均
    parameter_detail: dict[str, float]
    n_unique_settings: int
    baseline_statistics: dict | None
    baseline_significance: SignificanceResult | None
    warnings: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return walk_forward_verdict(self)

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_name": self.target_name,
            "n_folds": len(self.folds),
            "oos_total_return": self.oos_statistics.get("total_return"),
            "oos_annual_return": self.oos_statistics.get("annual_return"),
            "oos_sharpe_ratio": self.oos_statistics.get("sharpe_ratio"),
            "oos_max_ddpercent": self.oos_statistics.get("max_ddpercent"),
            "efficiency_median": self.efficiency_median,
            "efficiency_mean": self.efficiency_mean,
            "parameter_stability": self.parameter_stability,
            "n_unique_settings": self.n_unique_settings,
            "significance": self.significance.as_dict(),
            "baseline_sharpe": (
                self.baseline_statistics.get("sharpe_ratio") if self.baseline_statistics else None
            ),
            "verdict": self.verdict,
            "warnings": list(self.warnings),
        }


def walk_forward_verdict(report: WalkForwardReport) -> str:
    """Walk-Forward 判据。

    通过（可考虑上真钱，仍须配合 PBO）需要【同时】满足：
      1. 样本外 Sharpe > 0 且 block-bootstrap p < 0.05；
      2. walk_forward_efficiency 中位数 ≥ 0.5（样本外至少保住样本内一半）；
      3. parameter_stability ≥ 0.5（各折选出的参数不是每折一换）；
      4. 若给了 baseline，优化参数的样本外 Sharpe 必须 ≥ baseline —— 否则整套优化是负贡献。

    任一不满足即不通过。第 4 条最常被忽略：如果优化打不过教科书默认参数，
    那么"优化"这个动作本身就是在烧样本。
    """
    s = report.significance
    fails: list[str] = []
    if not (s.sharpe > 0 and math.isfinite(s.p_block_bootstrap) and s.p_block_bootstrap < s.alpha):
        fails.append(f"样本外 Sharpe={s.sharpe:.2f} / p={s.p_block_bootstrap:.3f} 不显著")
    if not math.isfinite(report.efficiency_median):
        # 与"效率低"区分开：没有一折的样本内年化为正，比值根本没有定义。
        # 仍然算不通过（证明不了没衰减），但理由必须诚实，不能印成 "nan < 0.5"。
        fails.append("WF 效率中位数不可计算：无一折样本内年化为正，比值无定义")
    elif report.efficiency_median < 0.5:
        fails.append(f"WF 效率中位数={report.efficiency_median:.2f} < 0.5")
    if report.parameter_stability < 0.5:
        fails.append(f"参数稳定性={report.parameter_stability:.2f} < 0.5")
    if report.baseline_statistics is not None:
        opt = float(report.oos_statistics.get("sharpe_ratio", 0.0) or 0.0)
        base = float(report.baseline_statistics.get("sharpe_ratio", 0.0) or 0.0)
        if opt < base:
            fails.append(f"优化样本外 Sharpe {opt:.2f} < 默认参数 {base:.2f}（优化是负贡献）")
    return "通过" if not fails else "不通过：" + "；".join(fails)


def _annual_return_of(stats: dict) -> float:
    value = stats.get("annual_return", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def walk_forward_efficiency(
    is_annual_return: float, oos_annual_return: float, floor: float = 1e-9
) -> float:
    """Pardo 的 Walk-Forward Efficiency = 样本外年化 / 样本内年化。

    **只有在样本内年化为正时这个比值才有定义。** 这不是洁癖，是符号陷阱：
    某折样本内 −10%、样本外 −20%（即样本外亏得更狠），裸除得到 +2.0 ——
    落进"效率 ≥ 0.5 = 没有衰减"这一档，把一个灾难折记成优秀折，
    而且它会把中位数往上抬，足以让 `walk_forward_verdict` 从"不通过"翻成"通过"。
    同理样本内 −10%、样本外 +5% 会得到 −0.5，把一个变好的折记成负效率。

    因此分母 ≤ 0 时返回 **nan（无定义）而不是 0（衰减到底）**：
    下游 `effective_efficiency` / `np.median` 一律只统计有限值，
    "这折没资格进效率统计"与"这折效率是 0"是两件事，不能混。

    分母为正但极小的折（IS 年化 ≈ 0）同样返回 nan：比值会被放大到任意数值，
    这是数值噪声不是衰减信号。
    """
    if not (math.isfinite(is_annual_return) and math.isfinite(oos_annual_return)):
        return float("nan")
    if is_annual_return <= floor:
        return float("nan")
    return oos_annual_return / is_annual_return


def stitch_daily_frames(frames: Sequence[DataFrame]) -> DataFrame:
    """把各折的样本外 daily_df 首尾相接成一条连续曲线。

    拼的是 **net_pnl**，不是净值 —— `calculate_statistics` 会用
    `balance = net_pnl.cumsum() + capital` 重算净值，所以拼 net_pnl 得到的正是
    "一路按固定名义本金做下来"的曲线。

    这里有一个必须讲清的假设：**各折之间不复利**。每折回测都以同一初始资金起步，
    策略的 `trading_capital` 也是固定参数，因此每折的 net_pnl 建立在同一名义本金上，
    相加才有意义。若要看复利效果，需改成按折缩放仓位，本函数不做这件事。
    """
    valid = [f for f in frames if f is not None and not f.empty]
    if not valid:
        return DataFrame()
    stitched = concat(valid).sort_index()
    dup = stitched.index.duplicated()
    if dup.any():
        raise ValueError(f"样本外窗口发生重叠：{int(dup.sum())} 个重复日期 —— 切分逻辑有误")
    return stitched


def run_walk_forward(
    runner: BacktestRunner,
    settings: Sequence[dict],
    splits: Sequence[WalkForwardSplit],
    target_name: str = "sharpe_ratio",
    statistics_func: Callable[[DataFrame], dict] | None = None,
    selector: Selector = argmax_selector,
    baseline_setting: dict | None = None,
    annual_days: int = 252,
    alpha: float = 0.05,
    capital: float | None = None,
    n_bootstrap: int = 4000,
    block_size: int | None = None,
    seed: int = 20260725,
) -> WalkForwardReport:
    """跑完整的 Walk-Forward。

    runner          (setting, start, end) -> daily_df。生产用 `EngineRunner`。
    settings        候选参数网格，通常来自 `OptimizationSetting.generate_settings()`。
    splits          `make_walk_forward_splits()` 的输出。
    target_name     选参目标，取自 statistics 面板的键（sharpe_ratio / r_cubed / ...）。
    statistics_func daily_df -> statistics dict。留空则用 `EngineRunner.statistics`
                    （要求 runner 是 EngineRunner）。
    selector        选参规则，默认 argmax；传 `plateau_selector` 可实测高原选参是否更抗衰减。
    baseline_setting 对照组参数（如教科书默认值）。给了就一并跑一条样本外曲线，
                    用来回答"优化到底有没有正贡献"。

    折边界的处理：`EngineRunner(warmup_bars=0)` 时每折样本外从【空仓 + 冷指标】开始，
    这会切断一笔正在持有的趋势单，对海龟这类长持仓策略是系统性低估；更糟的是
    指标可能根本预热不完（见 EngineRunner 的 warmup_bars 说明，本项目实测就踩到了）。
    **生产用法一律设 warmup_bars ≥ 策略最长窗口的 2 倍。**
    """
    if not settings:
        raise ValueError("settings 为空")
    if not splits:
        raise ValueError("splits 为空 —— 样本长度不足以切出任何一折")

    if statistics_func is None:
        if not isinstance(runner, EngineRunner):
            raise ValueError("runner 不是 EngineRunner 时必须显式传入 statistics_func")
        statistics_func = runner.statistics
    if capital is None:
        capital = float(runner.capital) if isinstance(runner, EngineRunner) else 1.0

    warnings: list[str] = []
    if len(splits) < 4:
        warnings.append(f"仅 {len(splits)} 折：WF 效率的中位数几乎没有统计意义")

    folds: list[WalkForwardFold] = []
    n_nonpositive_is: int = 0
    for split in splits:
        is_frames = run_settings(runner, settings, split.train_start, split.train_end)
        is_stats = [statistics_func(f) for f in is_frames]
        scores = np.array(
            [float(s.get(target_name, 0.0) or 0.0) for s in is_stats], dtype=float
        )
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        if not np.isfinite(scores).any():
            warnings.append(f"第 {split.index} 折样本内全部无效，跳过")
            continue

        chosen = selector(settings, scores)
        oos_df = runner(settings[chosen], split.test_start, split.test_end)
        oos_stats = statistics_func(oos_df)

        is_annual = _annual_return_of(is_stats[chosen])
        oos_annual = _annual_return_of(oos_stats)
        eff = walk_forward_efficiency(is_annual, oos_annual)
        if is_annual <= 0:
            n_nonpositive_is += 1

        folds.append(
            WalkForwardFold(
                split=split, chosen_setting=dict(settings[chosen]), chosen_index=chosen,
                is_target=float(scores[chosen]),
                oos_target=float(oos_stats.get(target_name, 0.0) or 0.0),
                is_annual_return=is_annual, oos_annual_return=oos_annual, efficiency=eff,
                n_candidates=len(settings), oos_daily_df=oos_df, oos_statistics=oos_stats,
            )
        )

    if not folds:
        raise ValueError("所有折都失败，无法生成报告")

    stitched = stitch_daily_frames([f.oos_daily_df for f in folds])
    oos_stats_all = statistics_func(stitched)

    # 零成交守卫：样本外一笔没成交，几乎一定是【指标预热不足】而不是"策略选择不交易"。
    # 短窗口回测里 ArrayManager 填不满 → am.inited 恒为 False → 一笔不发。
    # 这种情形下所有绩效都是 0，会被误读成"样本外持平"，必须显式喊出来。
    if "trade_count" in stitched.columns and float(stitched["trade_count"].sum()) == 0.0:
        warnings.append(
            "样本外零成交：测试窗长度很可能不足以让策略的指标预热完成"
            "（检查 on_init 里 load_bar 的天数与 interval，或给 EngineRunner 设 warmup_bars）"
        )
    oos_returns = daily_log_returns(stitched["net_pnl"].to_numpy(), capital)
    sig = assess_significance(
        oos_returns, annual_days=annual_days, alpha=alpha, seed=seed,
        n_bootstrap=n_bootstrap, block_size=block_size,
    )

    effs = np.array([f.efficiency for f in folds], dtype=float)
    finite = effs[np.isfinite(effs)]
    eff_median = float(np.median(finite)) if finite.size else float("nan")
    eff_mean = float(np.mean(finite)) if finite.size else float("nan")
    if n_nonpositive_is:
        # 不是"提醒一下"：这些折被排除在效率统计之外，中位数的分母因此变小。
        # 分母小到 0 时 eff_median = nan，判据会走"不可计算"分支而不是"通过"。
        warnings.append(
            f"{n_nonpositive_is}/{len(folds)} 折的样本内年化 ≤ 0，"
            f"WF 效率对这些折无定义（比值符号会反转），已排除出效率统计；"
            f"参与效率统计的折数 = {int(finite.size)}"
        )

    # 参数稳定性：逐参数看"取众数的折数占比"，再平均
    detail: dict[str, float] = {}
    keys = sorted({k for f in folds for k in f.chosen_setting})
    for k in keys:
        vals = [repr(f.chosen_setting.get(k)) for f in folds]
        detail[k] = max(vals.count(v) for v in set(vals)) / len(vals)
    stability = float(np.mean(list(detail.values()))) if detail else 1.0
    n_unique = len({repr(sorted(f.chosen_setting.items())) for f in folds})

    baseline_stats: dict | None = None
    baseline_sig: SignificanceResult | None = None
    if baseline_setting is not None:
        base_frames = [
            runner(baseline_setting, s.test_start, s.test_end) for s in [f.split for f in folds]
        ]
        base_stitched = stitch_daily_frames(base_frames)
        if not base_stitched.empty:
            baseline_stats = statistics_func(base_stitched)
            baseline_sig = assess_significance(
                daily_log_returns(base_stitched["net_pnl"].to_numpy(), capital),
                annual_days=annual_days, alpha=alpha, seed=seed,
                n_bootstrap=n_bootstrap, block_size=block_size,
            )

    return WalkForwardReport(
        target_name=target_name, folds=folds, oos_daily_df=stitched,
        oos_statistics=oos_stats_all, significance=sig,
        efficiency_median=eff_median, efficiency_mean=eff_mean,
        parameter_stability=stability, parameter_detail=detail,
        n_unique_settings=n_unique, baseline_statistics=baseline_stats,
        baseline_significance=baseline_sig, warnings=warnings,
    )


# ══════════════════════════════════════════════════════════════════════
# 6. 与参数寻优的接口 + 文本报告
# ══════════════════════════════════════════════════════════════════════

def settings_from_optimization(optimization_setting: OptimizationSetting) -> list[dict]:
    """把 `OptimizationSetting` 展开成参数列表 —— 与 run_bf_optimization 用的是同一套网格，
    保证"寻优时评估的参数集合"和"PBO 里被排名的参数集合"逐一对应。"""
    return optimization_setting.generate_settings()


@dataclass
class PBOStudy:
    """一次完整的 PBO 研究：点估计 + 相位稳定性 + 零分布 + 样本诊断。

    单独一个 PBO 数字不可报 —— 必须四件一起看，缺一件都会得出过度自信的结论。
    """

    result: PBOResult
    stability: dict[str, float]
    null: PBONull | None
    diagnosis: SampleDiagnosis
    correlation: float
    matrix_shape: tuple[int, int]
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.null is None:
            return self.result.verdict
        return pbo_verdict_calibrated(self.result.pbo, self.null)

    @property
    def p_value(self) -> float:
        return self.null.p_value(self.result.pbo) if self.null else float("nan")

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = self.result.as_dict()
        out.update(self.stability)
        out["p_value_vs_null"] = self.p_value
        out["calibrated_verdict"] = self.verdict
        out["avg_config_correlation"] = self.correlation
        out["n_obs"], out["n_columns"] = self.matrix_shape
        out["notes"] = list(self.notes)
        return out

    def text(self) -> str:
        return format_pbo_report(self)


def pbo_from_settings(
    runner: BacktestRunner,
    settings: Sequence[dict],
    start: datetime,
    end: datetime,
    capital: float,
    n_blocks: int | None = None,
    annual_days: int = 252,
    min_block_obs: int = 20,
    n_offsets: int = 8,
    n_null_sims: int = 200,
    seed: int = 20260725,
    round_trip_counter: Callable[[DataFrame], int] | None = None,
) -> PBOStudy:
    """一步到位：跑完参数网格 → 收益矩阵 → CSCV → PBO + 相位稳定性 + 零分布 + 样本诊断。

    每组参数在【全样本】上只回测一次；CSCV 之后只是对这条已算好的收益序列做时间切分。
    这一点必须说清：**块边界不会打断任何一笔交易** —— 仓位序列始终连续，
    切分只决定哪些天被记作样本内、哪些记作样本外。所以 CSCV 不需要重跑回测，
    也不会因为切块而人为制造"折边界平仓"的失真（这一点与 Walk-Forward 不同）。

    `n_null_sims=0` 可跳过零分布模拟（约 0.09 秒/次 × n_sims），但那样只能拿到
    未校准的固定阈值判据，不推荐。

    `round_trip_counter`：daily_df -> 该组参数的**完整回合数**。不传则退回
    `trade_count / 2`（成交笔数的一半）——对金字塔加仓策略会高估一倍以上
    （海龟 4 买 1 卖 = 5 笔成交 → 估 2.5 个回合，实际 1 个），
    于是本报告的"样本诊断"会比 `overfitting_audit` 用仓位序列数出来的那份宽松，
    同一份报告里出现两个互相矛盾的回合数。生产调用请传
    `lambda df: count_round_trips(df)[1]`（`overfitting_audit.run_audit` 已经这么做）。
    """
    frames = run_settings(runner, settings, start, end)
    matrix, dropped = returns_matrix(frames, capital)
    notes: list[str] = []
    if dropped:
        notes.append(f"{len(dropped)} 组参数回测为空，已剔除（下标 {dropped[:10]}）")
    if matrix.size == 0:
        raise ValueError("所有参数组回测均为空，无法计算 PBO")

    extra_notes: list[str] = []
    non_empty = [f for f in frames if f is not None and not f.empty]
    if round_trip_counter is not None:
        # 精确路径：由调用方按仓位序列数完整回合（金字塔加仓不会被数成多个回合）
        avg_round_trips = (
            sum(float(round_trip_counter(f)) for f in non_empty) / len(non_empty)
            if non_empty else 0.0
        )
    else:
        # 退化路径：trade_count 是「成交笔数」，一个回合至少一买一卖，除以 2 近似
        avg_round_trips = (
            sum(float(f["trade_count"].sum()) for f in non_empty) / len(non_empty) / 2.0
            if non_empty else 0.0
        )
        extra_notes.append(
            "回合数按 trade_count/2 估算（未传 round_trip_counter）："
            "金字塔加仓策略会被高估，样本诊断偏宽松"
        )

    return pbo_from_matrix(
        matrix,
        annual_days=annual_days,
        n_blocks=n_blocks,
        min_block_obs=min_block_obs,
        n_offsets=n_offsets,
        n_null_sims=n_null_sims,
        seed=seed,
        avg_round_trips=avg_round_trips,
        notes=notes,
        extra_notes=extra_notes,
    )


def pbo_from_matrix(
    matrix: np.ndarray,
    annual_days: int = 252,
    n_blocks: int | None = None,
    min_block_obs: int = 20,
    n_offsets: int = 8,
    n_null_sims: int = 200,
    seed: int = 20260725,
    avg_round_trips: float | None = None,
    notes: Sequence[str] | None = None,
    extra_notes: Sequence[str] | None = None,
) -> PBOStudy:
    """已有 (T, N) 收益矩阵时的 PBO 研究入口 —— 不重跑任何回测。

    `pbo_from_settings` 是"先跑网格再算"，本函数是"矩阵已经有了再算"。参数寻优
    （`BacktestingEngine.run_bf_optimization`）本身就把每组参数在全样本上跑了一遍，
    收益矩阵是那趟回放的副产品；再走 `pbo_from_settings` 等于把整个网格重跑第二遍，
    而且用的是另一条取数路径（`EngineRunner` 单次查询 vs 寻优子进程的分块查询），
    两条路径的 K 线根数实测不一致。所以寻优后的 PBO 必须吃寻优自己产出的矩阵。

    matrix          (T, N)，第 n 列是第 n 组参数的逐期【对数收益】（口径见
                    `daily_log_returns`）。列的顺序由调用方负责与参数组对齐。
    avg_round_trips 每组参数的平均完整回合数，喂给 `sample_size_diagnosis`。
                    传 None 表示"数不出来"，样本诊断里的交易笔数一栏留空
                    —— 比塞一个猜出来的数诚实。
    notes           先于块数建议插入的备注（例如"某几组回测为空已剔除"）。
    extra_notes     紧跟在块数建议之后插入的备注（例如回合数的口径退化说明）。
                    两个参数分开是为了让最终备注顺序与 `pbo_from_settings` 一致。

    `n_null_sims=0` 可跳过零分布模拟（约 0.09 秒/次 × n_sims），但那样只能拿到
    未校准的固定阈值判据，不推荐。
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"matrix 必须是二维 (T, N)，收到 shape={m.shape}")
    if m.size == 0:
        raise ValueError("收益矩阵为空，无法计算 PBO")

    out_notes: list[str] = list(notes) if notes else []
    n_obs, n_cols = m.shape
    if n_blocks is None:
        n_blocks, block_notes = recommend_n_blocks(n_obs, min_block_obs=min_block_obs)
        out_notes.extend(block_notes)
    if n_blocks < 2:
        raise ValueError(f"样本 {n_obs} 期不足以做 CSCV")
    if extra_notes:
        out_notes.extend(extra_notes)

    results, summary = cscv_pbo_stability(
        m, n_blocks=n_blocks, annual_days=annual_days, n_offsets=n_offsets
    )
    representative = min(results, key=lambda r: abs(r.pbo - summary["pbo_median"]))
    rho = average_config_correlation(m)

    null: PBONull | None = None
    if n_null_sims > 0:
        null = pbo_null_distribution(
            n_obs=representative.n_obs_used, n_configs=n_cols, n_blocks=n_blocks,
            annual_days=annual_days, correlation=0.0 if not math.isfinite(rho) else max(rho, 0.0),
            n_sims=n_null_sims, seed=seed,
        )

    diagnosis = sample_size_diagnosis(
        n_obs=n_obs, n_blocks=n_blocks, annual_days=annual_days,
        # 不垫地板：`None` 是"数不出来"，`0` 是**数出来的 0**（整个网格一个回合
        # 都没成交）。把后者垫成 1 等于在一份专治自欺的报告里造一个没发生过的
        # 回合，而零成交恰恰是这套判据最该喊出来的情形。
        trades_total=(
            int(round(avg_round_trips)) if avg_round_trips is not None else None
        ),
    )
    if math.isfinite(rho) and rho > 0.95:
        out_notes.append(
            f"参数组间平均相关 {rho:.3f} > 0.95：网格里实际上只有一个策略，"
            f"「N 组参数」的多样性是假的"
        )
    return PBOStudy(
        result=representative, stability=summary, null=null, diagnosis=diagnosis,
        correlation=rho, matrix_shape=(n_obs, n_cols), notes=out_notes,
    )


def format_pbo_report(study: PBOStudy) -> str:
    """人读的 PBO 报告。"""
    result, summary, null = study.result, study.stability, study.null
    lines = [
        "─" * 66,
        "CSCV / PBO（回测过拟合概率）",
        f"  参数组数 N         : {result.n_configs}"
        f"（平均两两相关 {study.correlation:.3f}）",
        f"  块数 S / 块长       : {result.n_blocks} / {result.block_size} 期",
        f"  组合数 C(S,S/2)     : {result.n_combinations}",
        f"  参与计算期数        : {result.n_obs_used}",
        "",
        f"  PBO                 : {result.pbo:.3f}",
        f"  截面 rank IC        : {result.rank_ic:+.3f}"
        f"（样本内排名对样本外排名的预测力；≈0 = 排名纯噪声，<0 = 系统性反转）",
        f"  样本外亏损概率      : {result.prob_oos_loss:.3f}"
        f"（被选参数样本外 Sharpe<0 的组合占比）",
        f"  退化回归            : OOS ≈ {result.degradation_intercept:+.2f} "
        f"{result.degradation_slope:+.2f}×IS   （描述量；IS/OOS 互补导致其天然为负，勿单独解读）",
        f"  被选参数 OOS 中位    : {result.median_oos_selected:+.3f}  vs  "
        f"全体中位 {result.median_oos_all:+.3f}",
        f"  并列占比            : {result.tie_fraction:.3f}",
    ]
    if summary:
        lines += [
            "",
            f"  相位稳定性（{int(summary['n_offsets'])} 个块网格相位）: "
            f"PBO ∈ [{summary['pbo_min']:.3f}, {summary['pbo_max']:.3f}]，"
            f"中位 {summary['pbo_median']:.3f}，跨度 {summary['pbo_spread']:.3f}",
        ]
        if summary["pbo_spread"] > 0.15:
            lines.append("  ⚠ 跨度 >0.15：PBO 估计本身随块边界漂移，不可只报单一数字")
    if null is not None:
        lines += [
            "",
            f"  零分布（{null.n_sims} 次模拟，同 T/N/S，无任何真实优劣差异）:",
            f"     均值 {null.mean:.3f}  标准差 {null.sd:.3f}  "
            f"5%分位 {null.q05:.3f}  中位 {null.q50:.3f}  95%分位 {null.q95:.3f}",
            f"     观测 PBO 的单侧 p = {study.p_value:.3f}",
            f"     ⇒ 纯噪声也有 {null.p_value(result.pbo) * 100:.0f}% 的概率给出不高于本次的 PBO",
        ]
    for w in result.warnings:
        lines.append(f"  ⚠ {w}")
    for note in study.notes:
        lines.append(f"  · {note}")
    lines.append("")
    lines.append("样本诊断:")
    for note in study.diagnosis.notes:
        lines.append(f"  · {note}")
    lines += ["", f"判定：{study.verdict}", "─" * 66]
    return "\n".join(lines)


def format_walk_forward_report(report: WalkForwardReport) -> str:
    """人读的 Walk-Forward 报告。"""
    s = report.significance
    lines = [
        "─" * 62,
        f"Walk-Forward（选参目标 = {report.target_name}，共 {len(report.folds)} 折）",
        "",
        f"{'折':>3} {'训练窗':>21} {'测试窗':>21} {'IS年化':>9} {'OOS年化':>9} {'效率':>7}  参数",
    ]
    for f in report.folds:
        sp = f.split
        lines.append(
            f"{sp.index:>3} "
            f"{sp.train_start:%Y-%m-%d}~{sp.train_end:%Y-%m-%d} "
            f"{sp.test_start:%Y-%m-%d}~{sp.test_end:%Y-%m-%d} "
            f"{f.is_annual_return:>8.1f}% {f.oos_annual_return:>8.1f}% "
            f"{f.efficiency:>7.2f}  {f.chosen_setting}"
        )
    st = report.oos_statistics
    lines += [
        "",
        "拼接后的样本外曲线（这才是「当年照做会得到什么」）：",
        f"  总收益 {st.get('total_return', 0):.2f}%   年化 {st.get('annual_return', 0):.2f}%   "
        f"最大回撤 {st.get('max_ddpercent', 0):.2f}%",
        f"  Sharpe {s.sharpe:+.2f} ± {s.sharpe_se:.2f}(SE)   t={s.t_stat:+.2f}   "
        f"p_normal={s.p_normal:.3f}   p_block={s.p_block_bootstrap:.3f}   n={s.n_obs}",
        "",
        f"  WF 效率  中位 {report.efficiency_median:.2f} / 均值 {report.efficiency_mean:.2f}"
        f"（<0.5 = 严重衰减）",
        f"  参数稳定性 {report.parameter_stability:.2f}"
        f"（{report.n_unique_settings} 套不同参数 / {len(report.folds)} 折）",
        f"  逐参数: {report.parameter_detail}",
    ]
    if report.baseline_statistics is not None:
        lines.append(
            f"  对照(默认参数) 样本外 Sharpe {report.baseline_statistics.get('sharpe_ratio', 0):+.2f}"
            f" / 年化 {report.baseline_statistics.get('annual_return', 0):.2f}%"
        )
    for w in report.warnings:
        lines.append(f"  ⚠ {w}")
    lines += ["", f"判定：{report.verdict}", "─" * 62]
    return "\n".join(lines)
