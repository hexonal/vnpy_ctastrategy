"""Deflated Sharpe Ratio（DSR）与其组件：PSR、夏普标准误、期望最大夏普。

来源：
    Bailey & López de Prado (2014), "The Deflated Sharpe Ratio: Correcting for
    Selection Bias, Backtest Overfitting and Non-Normality", Journal of
    Portfolio Management 40(5), 94-107.
    其中的 PSR 来自 Bailey & López de Prado (2012), "The Sharpe Ratio Efficient
    Frontier", Journal of Risk 15(2), 3-44.

━━━ 这个模块回答的问题，和 robust_metrics.py 回答的问题不是同一个 ━━━

`calculate_statistics()` 现有的全部指标 —— 含本 fork 新加的 RAR / R³ / Robust
Sharpe —— 都是【样本内描述统计】：它们描述这条曲线长什么样，不回答"这个结果
是不是运气"。DSR 是【统计检验】，回答的是后者，而且是在【挑过最好的那组参数
之后】仍然成立的那种回答。

两层偏差，本模块分别处理：

  第一层 · 单条曲线的抽样噪声 → PSR / 夏普标准误
      夏普本身是估计量，有标准误。日线 600 根、正态假设下，年化夏普的标准误
      约 √(247/599) ≈ 0.64；样本只有 134 根时约 1.36。也就是说一条 134 日、
      年化夏普 −1.68 的曲线，连"真实夏普是负的"都没到 2σ —— 它和纯随机不可
      区分。`sharpe_significance()` 把这件事直接印在面板上。

  第二层 · 多重检验（选择偏差）→ DSR
      跑 N 组参数取最好的一组，那一组的夏普天然被高估：即便所有参数都毫无
      技能（真实夏普 = 0），N 组试验里的最大值也会显著大于 0。期望值为
          E[max SR] ≈ √V · [(1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))]
      γ 为 Euler–Mascheroni 常数，V 为试验间夏普的方差。DSR 就是把这个
      E[max SR] 当作零假设基准去算 PSR：
          DSR = PSR(SR*) = Φ[ (SR − SR*)·√(T−1) / √(1 − γ₃·SR + (γ₄−1)/4·SR²) ]
      DSR ≥ 0.95 才算"在多重检验之后仍然显著"。

━━━ 单位约定（最容易出错的地方，务必先读）━━━

Bailey 的公式里 SR、SR*、γ₃、γ₄、T 必须是【同一频率】的：SR 是每期（本项目
= 每日）夏普，γ₃/γ₄ 是每期收益的偏度/峰度，T 是期数。而 vnpy 的
`statistics["sharpe_ratio"]` 是【年化】的。本模块所有对外函数默认
`annualised=True`，即输入按 vnpy 口径给年化值，内部自行除以 √annual_days；
输出同时给出年化与每日两套数字。传 `annualised=False` 则全程按每期口径。

γ₄ 一律是【非超额】峰度（正态 = 3）。pandas 的 `Series.kurt()` 返回的是超额
峰度（Fisher，正态 = 0），直接喂进来会把分母算错 —— 用 `return_moments()`
取矩，不要用 pandas 的 skew/kurt。

━━━ 已知局限（用之前必须知道，不要拿 DSR 当免死金牌）━━━

1. V（试验间夏普方差）是从实际跑出来的 N 组结果里估的，而这 N 组【不独立】
   （相邻参数的收益曲线高度相关）。Bailey 的 E[max] 公式假设 N 个试验 iid，
   用相关试验去套会高估 E[max] → SR* 偏高 → DSR 偏保守。这个方向是安全的。
2. 反过来，若策略确实有技能，V 里混进了真实的收益离散度，同样会抬高 SR*。
   DSR 因此是一个偏保守的检验，通不过不等于策略一定无效。
3. 遗传算法（`run_ga_optimization`）的试验分布【不是随机采样】：它主动往好的
   区域收敛，试验夏普被截尾，V 被系统性低估 → SR* 偏低 → DSR 偏乐观。这个
   方向不安全。GA 的 V 应当另取一次随机/穷举采样来估，见
   `trial_sharpes_from_optimization()` 的 docstring。
4. DSR 只处理"挑参数"这一种多重检验。数据窥探（换标的、换时间段、改策略逻辑
   之后重跑）不计入 N，而它们同样是多重检验。N 只是【记录在案的】试验次数。
5. DSR 不检验样本外表现。它说的是"样本内这个夏普在扣掉选择偏差后仍不像运气"，
   不是"样本外还会这样"。样本外要靠 walk-forward / PBO。
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

# Euler–Mascheroni 常数，Bailey (2014) 式 (5) 中的 γ
EULER_MASCHERONI: float = 0.5772156649015329

# 标准正态。用标准库 statistics.NormalDist 而不是 scipy.stats.norm：
# 本包 pyproject 的依赖只有 vnpy/pandas/plotly，不含 scipy，加一个重依赖只为
# 两个函数不划算。实测两者的 cdf/ppf 在 [1e-12, 1-1e-9] 上差 <2e-15
# （见 tests/test_deflated_sharpe.py::test_normal_dist_matches_scipy）。
_NORMAL: NormalDist = NormalDist()

DEFAULT_CONFIDENCE: float = 0.95
NORMAL_KURTOSIS: float = 3.0


# ── 基础：正态分布 ─────────────────────────────────────────────────────

def normal_cdf(x: float) -> float:
    """标准正态 CDF Φ(x)。"""
    return _NORMAL.cdf(x)


def normal_ppf(p: float) -> float:
    """标准正态分位数 Φ⁻¹(p)，p 必须严格落在 (0, 1)。"""
    if not 0.0 < p < 1.0:
        raise ValueError(f"概率必须落在开区间 (0, 1)，收到 {p}")
    return _NORMAL.inv_cdf(p)


# ── 收益矩 ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReturnMoments:
    """每期收益的前四阶矩。峰度为【非超额】口径（正态 = 3）。"""

    n_obs: int
    mean: float
    std: float                   # ddof=1，与 vnpy 的 return_std 同口径
    skew: float                  # γ₃，总体（plug-in）估计
    kurtosis: float              # γ₄，总体（plug-in）估计，非超额

    @property
    def excess_kurtosis(self) -> float:
        """超额峰度（Fisher 口径，正态 = 0）。pandas `.kurt()` 返回的就是这个。"""
        return self.kurtosis - NORMAL_KURTOSIS

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def return_moments(daily_returns: Sequence[float] | np.ndarray) -> ReturnMoments:
    """从每期收益序列取前四阶矩。

    偏度/峰度用【总体（plug-in / 有偏）】估计量 m₃/m₂^1.5 与 m₄/m₂²，而不是
    pandas 默认的无偏修正版本。这不是图省事 —— 是 PSR 分母正定性的要求：

        分母² = 1 − γ₃·SR + (γ₄−1)/4·SR²

    把它看作 SR 的二次式，判别式 = γ₃² − (γ₄ − 1)。对任何【经验分布】，
    Cauchy–Schwarz 保证 γ₄ ≥ γ₃² + 1，故判别式 ≤ 0，二次式恒非负，分母恒有定义。
    而 pandas 的无偏修正版本会破坏这个恒等式（小样本下可以出现 γ₄ < γ₃² + 1），
    进而让分母开根号开出 NaN。用总体估计量则永远不会。

    std 仍用 ddof=1，因为它要和 vnpy `calculate_statistics()` 里的
    `df["return"].std()` 对齐（pandas 默认 ddof=1），否则算出来的夏普对不上。
    """
    values = np.asarray(daily_returns, dtype=float).ravel()
    n = int(values.size)
    if n < 2:
        raise ValueError(f"至少需要 2 个观测才能计算矩，收到 {n}")

    mean = float(np.mean(values))
    deviations = values - mean
    m2 = float(np.mean(deviations ** 2))
    std = float(np.std(values, ddof=1))

    if m2 <= 0.0:
        # 常数序列：偏度/峰度无定义，退回正态值，让下游公式退化成正态情形
        return ReturnMoments(n_obs=n, mean=mean, std=std, skew=0.0, kurtosis=NORMAL_KURTOSIS)

    m3 = float(np.mean(deviations ** 3))
    m4 = float(np.mean(deviations ** 4))
    skew = m3 / m2 ** 1.5
    kurtosis = m4 / m2 ** 2
    return ReturnMoments(n_obs=n, mean=mean, std=std, skew=skew, kurtosis=kurtosis)


# ── 第一层：单条曲线的抽样噪声 ─────────────────────────────────────────

def sharpe_variance_factor(sharpe: float, skew: float, kurtosis: float) -> float:
    """PSR 分母的平方：1 − γ₃·SR + (γ₄−1)/4·SR²（每期口径）。

    它就是夏普估计量渐近方差乘以 (T−1) 后的部分。负偏度（γ₃ < 0）配正夏普时
    这一项变大 → 标准误变大 → 显著性变差，符合直觉：靠"平时小赚、偶尔巨亏"
    堆出来的夏普不可信。
    """
    factor = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe * sharpe
    if factor <= 0.0:
        # 数学上（对经验分布）不可能，见 return_moments 的说明。这里兜底是防止
        # 调用方手工传入不自洽的 (skew, kurtosis) 组合导致 sqrt 出 NaN。
        raise ValueError(
            f"夏普方差因子非正（{factor}），(skew={skew}, kurtosis={kurtosis}) 不是"
            f"任何真实分布的矩组合：任何分布都满足 γ₄ ≥ γ₃² + 1"
        )
    return factor


def sharpe_standard_error(
    sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = NORMAL_KURTOSIS,
) -> float:
    """夏普估计量的标准误（与传入 sharpe 同频率口径）。

    SE(SR) = √[ (1 − γ₃·SR + (γ₄−1)/4·SR²) / (T − 1) ]

    正态、SR≈0 时退化为经典的 √(1/(T−1))。年化标准误 = 本值 × √annual_days。
    """
    if n_obs < 2:
        raise ValueError(f"至少需要 2 个观测，收到 {n_obs}")
    return math.sqrt(sharpe_variance_factor(sharpe, skew, kurtosis) / (n_obs - 1))


def probabilistic_sharpe_ratio(
    sharpe: float,
    benchmark_sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = NORMAL_KURTOSIS,
) -> float:
    """PSR：真实夏普超过 benchmark_sharpe 的概率（Bailey & LdP 2012）。

        PSR(SR*) = Φ[ (SR − SR*)·√(T−1) / √(1 − γ₃·SR + (γ₄−1)/4·SR²) ]

    sharpe / benchmark_sharpe 必须同频率（本模块内部一律先折算成每期口径）。
    benchmark_sharpe = 0 时它就是"真实夏普为正的概率"；benchmark 换成期望最大
    夏普 SR* 时，它就是 DSR。
    """
    if n_obs < 2:
        raise ValueError(f"至少需要 2 个观测，收到 {n_obs}")
    denominator = math.sqrt(sharpe_variance_factor(sharpe, skew, kurtosis))
    statistic = (sharpe - benchmark_sharpe) * math.sqrt(n_obs - 1) / denominator
    return normal_cdf(statistic)


@dataclass(frozen=True)
class SharpeSignificance:
    """单条回测曲线的夏普显著性 —— 不涉及多重检验，N=1 的情形。"""

    sharpe_annual: float
    sharpe_daily: float
    std_error_annual: float
    std_error_daily: float
    t_stat: float                       # (SR − 0) / SE，频率无关
    psr_zero: float                     # 真实夏普 > 0 的概率
    significant_at_95: bool             # psr_zero ≥ 0.95（单边）
    n_obs: int
    skew: float
    kurtosis: float
    annual_days: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        verdict = "显著" if self.significant_at_95 else "不显著（与随机不可区分）"
        return (
            f"年化夏普 {self.sharpe_annual:+.2f} ± {self.std_error_annual:.2f}"
            f"（1σ, n={self.n_obs}）  t={self.t_stat:+.2f}  "
            f"PSR(0)={self.psr_zero:.3f}  → {verdict}"
        )


def sharpe_significance(
    daily_returns: Sequence[float] | np.ndarray,
    annual_days: int,
    sharpe_annual: float | None = None,
) -> SharpeSignificance:
    """给一条收益序列算夏普标准误与 PSR(0)。

    daily_returns   每期收益率（小数）。与 vnpy `daily_df["return"]` 同口径 ——
                    注意那一列是【对数收益】，本函数不做任何转换，直接取矩。
    annual_days     年化交易日数（港股 247、美股 252，由调用方按市场传入）。
    sharpe_annual   年化夏普。默认 None 时按 mean/std(ddof=1)×√annual_days 自算；
                    传入则用传入值（用于和 vnpy 已经算好的 sharpe_ratio 对齐，
                    比如 vnpy 会扣 risk_free，自算不扣）。
    """
    moments = return_moments(daily_returns)
    root = math.sqrt(annual_days)

    if sharpe_annual is None:
        # std == 0 表示净值全程不动（例如整段没成交），夏普定义为 0 而非除零
        sharpe_annual = 0.0 if moments.std <= 0.0 else moments.mean / moments.std * root

    sharpe_daily = sharpe_annual / root
    se_daily = sharpe_standard_error(
        sharpe_daily, moments.n_obs, moments.skew, moments.kurtosis
    )
    psr0 = probabilistic_sharpe_ratio(
        sharpe_daily, 0.0, moments.n_obs, moments.skew, moments.kurtosis
    )
    return SharpeSignificance(
        sharpe_annual=sharpe_annual,
        sharpe_daily=sharpe_daily,
        std_error_annual=se_daily * root,
        std_error_daily=se_daily,
        t_stat=sharpe_daily / se_daily if se_daily else 0.0,
        psr_zero=psr0,
        significant_at_95=psr0 >= DEFAULT_CONFIDENCE,
        n_obs=moments.n_obs,
        skew=moments.skew,
        kurtosis=moments.kurtosis,
        annual_days=annual_days,
    )


# ── 第二层：多重检验 ──────────────────────────────────────────────────

def expected_max_sharpe(trial_sharpe_std: float, n_trials: int) -> float:
    """N 组【零技能】试验中最大夏普的期望值 SR*（Bailey 2014 式 5）。

        SR* = √V · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]

    输入输出同频率：传每日试验夏普标准差就得每日 SR*，传年化就得年化 SR*。

    N = 1 时无多重检验，返回 0.0（此时 Φ⁻¹(1 − 1/N) = Φ⁻¹(0) = −∞，原式不适用）。
    """
    if n_trials < 1:
        raise ValueError(f"试验次数至少为 1，收到 {n_trials}")
    if trial_sharpe_std < 0.0:
        raise ValueError(f"试验夏普标准差不能为负，收到 {trial_sharpe_std}")
    if n_trials == 1 or trial_sharpe_std == 0.0:
        return 0.0

    gamma = EULER_MASCHERONI
    first = normal_ppf(1.0 - 1.0 / n_trials)
    second = normal_ppf(1.0 - 1.0 / (n_trials * math.e))
    return trial_sharpe_std * ((1.0 - gamma) * first + gamma * second)


def minimum_sharpe_for_confidence(
    benchmark_sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = NORMAL_KURTOSIS,
    confidence: float = DEFAULT_CONFIDENCE,
) -> float:
    """反解 PSR：使 PSR(benchmark_sharpe) = confidence 所需的最小夏普（每期口径）。

    解析解而非迭代。令 a = z/√(T−1)、c = (γ₄−1)/4，PSR = confidence 等价于
        SR − SR* = a·√(1 − γ₃·SR + c·SR²)     （要求 SR > SR*）
    两边平方整理成二次方程
        (1 − a²c)·SR² + (a²γ₃ − 2SR*)·SR + (SR*² − a²) = 0
    取较大根并回代验证（平方会引入 SR < SR* 的伪根）。

    若 1 − a²c ≤ 0，说明检验统计量在 SR → ∞ 时上界仍低于 z（极端厚尾 + 极短
    样本才会发生），此时任何夏普都达不到该置信度，返回 math.inf。
    """
    if n_obs < 2:
        raise ValueError(f"至少需要 2 个观测，收到 {n_obs}")
    z = normal_ppf(confidence)
    a = z / math.sqrt(n_obs - 1)
    a2 = a * a
    c = (kurtosis - 1.0) / 4.0

    quad_a = 1.0 - a2 * c
    if quad_a <= 0.0:
        return math.inf

    quad_b = a2 * skew - 2.0 * benchmark_sharpe
    quad_c = benchmark_sharpe * benchmark_sharpe - a2

    discriminant = quad_b * quad_b - 4.0 * quad_a * quad_c
    if discriminant < 0.0:
        return math.inf

    root = math.sqrt(discriminant)
    for candidate in ((-quad_b + root) / (2.0 * quad_a), (-quad_b - root) / (2.0 * quad_a)):
        if candidate <= benchmark_sharpe:
            continue                     # 平方引入的伪根
        try:
            residual = candidate - benchmark_sharpe - a * math.sqrt(
                sharpe_variance_factor(candidate, skew, kurtosis)
            )
        except ValueError:
            continue
        if abs(residual) < 1e-9 * max(1.0, abs(candidate)):
            return candidate
    return math.inf


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """DSR 计算结果 + 全部中间量（便于核对，不必反查）。"""

    deflated_sharpe_ratio: float        # 扣掉选择偏差后，真实夏普 > 0 的概率
    probabilistic_sharpe_ratio: float   # 不扣选择偏差的 PSR(0)，用于对照
    significant: bool                   # DSR ≥ confidence

    observed_sharpe_annual: float
    observed_sharpe_daily: float
    required_sharpe_annual: float       # 达到 confidence 所需的最小年化夏普
    required_sharpe_daily: float

    expected_max_sharpe_annual: float   # SR*，零技能下 N 组试验的最大夏普期望
    expected_max_sharpe_daily: float
    trial_sharpe_std_annual: float
    n_trials: int

    n_obs: int
    skew: float
    kurtosis: float
    annual_days: int
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        verdict = "显著" if self.significant else "不显著（挑参数挑出来的）"
        return (
            f"DSR={self.deflated_sharpe_ratio:.3f} vs PSR(0)={self.probabilistic_sharpe_ratio:.3f}  "
            f"→ {verdict}\n"
            f"  观测年化夏普 {self.observed_sharpe_annual:+.2f}；"
            f"N={self.n_trials} 组试验的零技能最大夏普期望 SR*={self.expected_max_sharpe_annual:.2f}"
            f"（试验间夏普 std={self.trial_sharpe_std_annual:.2f}）\n"
            f"  在 n={self.n_obs}、skew={self.skew:+.2f}、kurt={self.kurtosis:.1f} 下，"
            f"达到 {self.confidence:.0%} 置信需要年化夏普 ≥ {self.required_sharpe_annual:.2f}"
        )


def deflated_sharpe_ratio(
    observed_sharpe: float,
    trial_sharpe_std: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = NORMAL_KURTOSIS,
    annual_days: int = 252,
    confidence: float = DEFAULT_CONFIDENCE,
    annualised: bool = True,
) -> DeflatedSharpeResult:
    """Deflated Sharpe Ratio。

    observed_sharpe     被选中那组参数的夏普。
    trial_sharpe_std    N 组试验的夏普【标准差】（不是标准误），与 observed 同口径。
    n_trials            试验次数 N。见模块 docstring 关于 GA 的警告。
    n_obs               收益观测数 T（本项目 ≈ 600 根日线）。
    skew, kurtosis      每期收益的 γ₃ / γ₄（非超额，正态 = 3）。用 return_moments() 取。
    annual_days         年化交易日数，仅在 annualised=True 时用于折算。
    confidence          判定显著的阈值，默认 0.95。
    annualised          True（默认，vnpy 口径）表示上面两个夏普是年化值。

    返回 DeflatedSharpeResult。判定看 `.significant`。
    """
    if n_obs < 2:
        raise ValueError(f"至少需要 2 个观测，收到 {n_obs}")
    if annual_days < 1:
        raise ValueError(f"annual_days 必须为正整数，收到 {annual_days}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence 必须落在 (0, 1)，收到 {confidence}")

    root = math.sqrt(annual_days) if annualised else 1.0
    sr_daily = observed_sharpe / root
    trial_std_daily = trial_sharpe_std / root

    sr_star_daily = expected_max_sharpe(trial_std_daily, n_trials)
    dsr = probabilistic_sharpe_ratio(sr_daily, sr_star_daily, n_obs, skew, kurtosis)
    psr0 = probabilistic_sharpe_ratio(sr_daily, 0.0, n_obs, skew, kurtosis)
    required_daily = minimum_sharpe_for_confidence(
        sr_star_daily, n_obs, skew, kurtosis, confidence
    )

    return DeflatedSharpeResult(
        deflated_sharpe_ratio=dsr,
        probabilistic_sharpe_ratio=psr0,
        significant=dsr >= confidence,
        observed_sharpe_annual=sr_daily * root,
        observed_sharpe_daily=sr_daily,
        required_sharpe_annual=required_daily * root,
        required_sharpe_daily=required_daily,
        expected_max_sharpe_annual=sr_star_daily * root,
        expected_max_sharpe_daily=sr_star_daily,
        trial_sharpe_std_annual=trial_std_daily * root,
        n_trials=n_trials,
        n_obs=n_obs,
        skew=skew,
        kurtosis=kurtosis,
        annual_days=annual_days,
        confidence=confidence,
    )


# ── 接 vnpy 寻优流程 ───────────────────────────────────────────────────

@dataclass(frozen=True)
class TrialSharpes:
    """从寻优结果里抽出来的试验夏普分布。"""

    values: np.ndarray                  # 可用的试验夏普（年化，vnpy 口径）
    n_total: int                        # 试验总数（含算不出夏普的）
    n_missing: int                      # statistics 里没有该键的试验数（爆仓等）

    @property
    def n_usable(self) -> int:
        return int(self.values.size)

    def std(self, ddof: int = 1) -> float:
        """试验间夏普标准差 √V。可用样本 < 2 时返回 0.0（等价于不做 deflation）。"""
        if self.values.size < 2:
            return 0.0
        return float(np.std(self.values, ddof=ddof))

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_total": self.n_total,
            "n_usable": self.n_usable,
            "n_missing": self.n_missing,
            "std": self.std(),
            "mean": float(np.mean(self.values)) if self.values.size else 0.0,
            "max": float(np.max(self.values)) if self.values.size else 0.0,
        }


def trial_sharpes_from_optimization(
    results: Sequence[Any],
    sharpe_key: str = "sharpe_ratio",
) -> TrialSharpes:
    """从 `BacktestingEngine.run_bf_optimization()` / `run_ga_optimization()` 的
    返回值里抽出全部试验的夏普。

    ━━━ 为什么这件事做得到（读过 vnpy/trader/optimize.py 之后的事实）━━━

    `vnpy_ctastrategy.backtesting.evaluate()` 的返回值是
        (setting, target_value, statistics)
    第三个元素是【那一组参数的完整 statistics dict】，不是只有目标值。

    穷举：`run_bf_optimization()` 里
        results = list(tqdm(executor.map(evaluate_func, settings)))
        results.sort(reverse=True, key=key_func)
        return results
    —— 返回的是【全部 len(settings) 组】结果，只是排了序，没有截断成 top N。
    所以 N 和试验夏普分布都是现成的，不需要重跑回测。

    遗传：`run_ga_optimization()` 里
        results = list(cache.values());  results.sort(...);  return results
    cache 是 `ga_evaluate()` 以参数元组为键的去重缓存，所以返回的是
    【所有被求值过的不重复参数组】。len(results) 就是唯一试验数（重复求值同一
    组参数不构成独立试验，不该计入 N，去重恰好是对的）。

    ⚠️ 但 GA 的【夏普分布】不能直接当 V 用：GA 主动向高适应度区域收敛，
    试验夏普被截尾，std 被系统性低估 → SR* 偏低 → DSR 偏乐观。GA 场景的正确
    做法是用 `run_bf_optimization` 在同一参数空间随机/穷举采一次样来估 V，
    再把这个 V 和 GA 的 n_trials 一起传给 `deflated_sharpe_ratio()`。

    算不出夏普的试验（回测中爆仓 → `calculate_statistics()` 返回 `{}`）计入
    n_total 但不进 values：它们确实是被搜索过的试验（影响 N），但没有夏普可以
    进入方差估计。两个计数都返回，调用方可自行决定口径。
    """
    values: list[float] = []
    n_total = 0
    n_missing = 0

    for item in results:
        n_total += 1
        stats: Any = None
        if isinstance(item, dict):
            stats = item
        elif isinstance(item, Sequence) and not isinstance(item, str | bytes) and len(item) >= 3:
            stats = item[2]

        if not isinstance(stats, dict) or sharpe_key not in stats:
            n_missing += 1
            continue

        raw = stats[sharpe_key]
        try:
            value = float(raw)
        except (TypeError, ValueError):
            n_missing += 1
            continue
        if not math.isfinite(value):
            n_missing += 1
            continue
        values.append(value)

    return TrialSharpes(
        values=np.asarray(values, dtype=float),
        n_total=n_total,
        n_missing=n_missing,
    )


# statistics dict 里收益矩的候选键名，按优先级排列。
# "sharpe_skew"/"sharpe_kurtosis" 由 sharpe_inference.py 写入（它已接进
# calculate_statistics），"return_skew_pop"/"return_kurtosis_pop" 是本模块
# 早期约定的键名，保留以兼容手工补丁过的分支。
SKEW_KEYS: tuple[str, ...] = ("sharpe_skew", "return_skew_pop")
KURTOSIS_KEYS: tuple[str, ...] = ("sharpe_kurtosis", "return_kurtosis_pop")


def _read_moments(statistics: dict[str, Any]) -> tuple[float, float]:
    """从 statistics dict 里读 (γ₃, γ₄)，读不到或不可信则退回正态 (0, 3)。

    必须做有效性校验，不能见到键就用：`sharpe_inference.statistics_fields()`
    在"未计算"时把 sharpe_skew / sharpe_kurtosis 填成 0.0，而 γ₄ = 0 不是任何
    分布的峰度 —— 任何分布都满足 γ₄ ≥ γ₃² + 1 ≥ 1。把这个哨兵值当真峰度用会
    让 PSR 分母偏小、显著性被高估。故以 γ₄ < 1 判定"没算出来"，退回正态。
    """
    kurtosis = NORMAL_KURTOSIS
    for key in KURTOSIS_KEYS:
        if key in statistics:
            try:
                candidate = float(statistics[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(candidate) and candidate >= 1.0:
                kurtosis = candidate
                break
            return 0.0, NORMAL_KURTOSIS          # 哨兵/无效值：整组矩都不采信

    skew = 0.0
    for key in SKEW_KEYS:
        if key in statistics:
            try:
                candidate = float(statistics[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(candidate):
                skew = candidate
                break

    # 保证 (γ₃, γ₄) 自洽，否则 sharpe_variance_factor 会抛错
    if kurtosis < skew * skew + 1.0:
        return 0.0, NORMAL_KURTOSIS
    return skew, kurtosis


def deflate_optimization(
    results: Sequence[Any],
    annual_days: int,
    selected_index: int = 0,
    confidence: float = DEFAULT_CONFIDENCE,
    sharpe_key: str = "sharpe_ratio",
    n_obs: int | None = None,
    skew: float | None = None,
    kurtosis: float | None = None,
    n_trials: int | None = None,
    trial_sharpe_std: float | None = None,
) -> DeflatedSharpeResult:
    """一步到位：把寻优结果直接转成 DSR。

    results         `run_bf_optimization()` / `run_ga_optimization()` 的返回值。
    annual_days     年化交易日数（港股 247、美股 252）。
    selected_index  被选中的那组参数在结果里的下标。results 已按目标值降序排好，
                    所以默认 0 = "最好的那组"，也就是真正会被拿去实盘的那组。
    n_obs           收益观测数。默认从该组的 statistics["total_days"] 读。
    skew/kurtosis   默认按 SKEW_KEYS / KURTOSIS_KEYS 从 statistics 里读 —— 也就是
                    sharpe_inference.py 已经写进面板的 "sharpe_skew" /
                    "sharpe_kurtosis"（同为总体估计、非超额口径），所以寻优流程
                    不需要为 DSR 再改一次 calculate_statistics。读不到或读到哨兵值
                    则退回正态 (0, 3)；注意正态假设对负偏厚尾的真实收益会【高估】
                    显著性，是不安全的方向，能读到真实矩就一定要读。
    n_trials        默认 = len(results)。GA 场景应显式传参（见
                    `trial_sharpes_from_optimization` 的警告）。
    trial_sharpe_std 默认从全部试验夏普算。GA 场景应传入独立采样估出的值。
    """
    if not results:
        raise ValueError("寻优结果为空，无法计算 DSR")
    if not -len(results) <= selected_index < len(results):
        raise IndexError(f"selected_index={selected_index} 越界，结果共 {len(results)} 组")

    selected = results[selected_index]
    if isinstance(selected, dict):
        selected_stats: dict[str, Any] = selected
    elif isinstance(selected, Sequence) and not isinstance(selected, str | bytes) and len(selected) >= 3:
        selected_stats = selected[2]
    else:
        raise TypeError(
            "寻优结果的元素必须是 (setting, target, statistics) 三元组或 statistics dict，"
            f"收到 {type(selected).__name__}"
        )
    if not isinstance(selected_stats, dict):
        raise TypeError(f"第 {selected_index} 组的 statistics 不是 dict：{type(selected_stats).__name__}")
    if sharpe_key not in selected_stats:
        raise KeyError(
            f"第 {selected_index} 组的 statistics 里没有 {sharpe_key!r} —— "
            "该组回测可能已爆仓（calculate_statistics 返回空 dict）"
        )

    trials = trial_sharpes_from_optimization(results, sharpe_key=sharpe_key)

    read_skew, read_kurt = _read_moments(selected_stats)
    if n_obs is None:
        if "total_days" not in selected_stats:
            raise KeyError(
                f"第 {selected_index} 组的 statistics 里没有 'total_days'，无法确定样本长度 T。"
                " 这不是 vnpy `calculate_statistics()` 的正常输出（它一定带这个键）；"
                " 若 results 是手工构造的，请显式传 n_obs="
            )
        resolved_n_obs = int(selected_stats["total_days"])
    else:
        resolved_n_obs = n_obs
    resolved_skew = read_skew if skew is None else skew
    resolved_kurt = read_kurt if kurtosis is None else kurtosis
    resolved_n_trials = trials.n_total if n_trials is None else n_trials
    resolved_std = trials.std() if trial_sharpe_std is None else trial_sharpe_std

    return deflated_sharpe_ratio(
        observed_sharpe=float(selected_stats[sharpe_key]),
        trial_sharpe_std=resolved_std,
        n_trials=resolved_n_trials,
        n_obs=resolved_n_obs,
        skew=resolved_skew,
        kurtosis=resolved_kurt,
        annual_days=annual_days,
        confidence=confidence,
        annualised=True,
    )


INTEGRATION_NOTE: str = (
    "DSR 不进 calculate_statistics()：单次回测里没有 N，也没有试验夏普分布，\n"
    "算不出 DSR。正确的接入点是寻优【之后】：\n"
    "    results = engine.run_bf_optimization(setting)\n"
    "    print(deflate_optimization(results, annual_days=247).summary())\n"
    "所需的 (γ₃, γ₄) 直接读 sharpe_inference.py 已写进面板的 sharpe_skew /\n"
    "sharpe_kurtosis，无需再改 calculate_statistics()。\n"
    "单条曲线那一层（标准误 / PSR）由 sharpe_inference.py 负责，本模块的\n"
    "sharpe_significance() 只是不依赖它时的轻量替代。"
)


# ── 门槛表（回答"原始夏普要多高才在 DSR 下仍显著"）────────────────────

def required_sharpe_table(
    n_trials_list: Sequence[int],
    trial_std_list: Sequence[float],
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = NORMAL_KURTOSIS,
    annual_days: int = 247,
    confidence: float = DEFAULT_CONFIDENCE,
) -> list[dict[str, float | int]]:
    """给定试验次数与试验夏普离散度，反解所需的最小【年化】夏普。

    所有输入输出都是年化口径。返回的每一行含 SR*（选择偏差本身的量级）与门槛值，
    两者之差就是"单条曲线抽样噪声"贡献的那部分。
    """
    root = math.sqrt(annual_days)
    rows: list[dict[str, float | int]] = []
    for trial_std in trial_std_list:
        for n_trials in n_trials_list:
            sr_star_daily = expected_max_sharpe(trial_std / root, n_trials)
            required_daily = minimum_sharpe_for_confidence(
                sr_star_daily, n_obs, skew, kurtosis, confidence
            )
            rows.append({
                "n_trials": n_trials,
                "trial_sharpe_std": trial_std,
                "n_obs": n_obs,
                "expected_max_sharpe": sr_star_daily * root,
                "required_sharpe": required_daily * root,
            })
    return rows


def format_required_sharpe_table(rows: Sequence[dict[str, float | int]]) -> str:
    """把 required_sharpe_table() 的输出排成等宽文本表。"""
    header = (
        f"{'N(试验)':>8} {'试验SR std':>11} {'样本T':>7} "
        f"{'SR*(选择偏差)':>14} {'门槛年化SR':>11}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        required = float(row["required_sharpe"])
        required_text = "  不可达" if math.isinf(required) else f"{required:11.2f}"
        lines.append(
            f"{int(row['n_trials']):>8} {float(row['trial_sharpe_std']):>11.2f} "
            f"{int(row['n_obs']):>7} {float(row['expected_max_sharpe']):>14.2f} {required_text}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI：打印门槛表。`python -m vnpy_ctastrategy.deflated_sharpe`"""
    parser = argparse.ArgumentParser(
        prog="deflated_sharpe",
        description="打印 DSR 显著性门槛表：跑 N 组参数后，原始年化夏普要多高才仍显著。",
    )
    parser.add_argument("--n-obs", type=int, default=600, help="收益观测数 T，默认 600（本项目典型日线样本）")
    parser.add_argument("--annual-days", type=int, default=247, help="年化交易日数，默认 247（港股）")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE, help="置信度，默认 0.95")
    parser.add_argument("--skew", type=float, default=0.0, help="每期收益偏度 γ₃，默认 0")
    parser.add_argument("--kurtosis", type=float, default=NORMAL_KURTOSIS, help="每期收益峰度 γ₄（非超额），默认 3")
    parser.add_argument("--n-trials", type=int, nargs="+", default=[1, 10, 100, 1000, 10000], help="试验次数列表")
    parser.add_argument("--trial-std", type=float, nargs="+", default=[0.25, 0.5, 1.0], help="试验间年化夏普标准差列表")
    args = parser.parse_args(argv)

    rows = required_sharpe_table(
        n_trials_list=args.n_trials,
        trial_std_list=args.trial_std,
        n_obs=args.n_obs,
        skew=args.skew,
        kurtosis=args.kurtosis,
        annual_days=args.annual_days,
        confidence=args.confidence,
    )
    print(
        f"DSR 门槛表  T={args.n_obs}  annual_days={args.annual_days}  "
        f"置信度={args.confidence:.0%}  skew={args.skew:+.2f}  kurt={args.kurtosis:.1f}"
    )
    print(format_required_sharpe_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
