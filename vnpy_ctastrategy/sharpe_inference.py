"""Sharpe 比率的标准误、t 统计量与显著性检验。

现有指标（含 RAR / R³ / Robust Sharpe）都是样本内描述统计，没有一个回答"这是不是运气"。
本模块产出 ``sharpe = 0.92, SE = 0.41, t = 2.24, p = 0.025``，而不是孤零零的 0.92。

Lo (2002, FAJ 58(4)) 给出 √n(SR̂ − SR) → N(0, V)。三档 V 全部实现：
  1. iid 正态（Lo 式9）      V = 1 + SR²/2
  2. iid 非正态（Mertens 2002） V = 1 − γ₃·SR + (γ₄−1)/4·SR²   γ₄ 非超额，正态=3
  3. 序列相关（GMM+Newey-West HAC） V = ∇'Ω∇，∇ = [1/σ, −SR/(2σ²)]
     Ω = u_t = [r_t−μ, (r_t−μ)²−σ²]' 的 Bartlett 长期协方差
第 3 档在 lags=0 时严格退化为第 2 档，第 2 档在正态下严格退化为第 1 档（有测试钉住）。

用哪个修正：所有修正项都乘着【单期】SR，日线 SR 很小（年化 1.0 / 247 日 → 日 SR≈0.064），
故高阶矩修正对 SE 的影响仅 0.1%~3%，几乎不动。真正会动 SE 的是序列相关 —— CTA 持仓跨多日，
日 P&L 天然自相关，AR(1) 系数 ρ 把 SE 放大 √((1+ρ)/(1−ρ))（ρ=0.3 → ×1.36）。
故默认 method="hac"，并输出 hac_inflation 让人看见修正有多大；≈1.0 即说明确实近似 iid。

⚠️ 把【年化】SR 代进公式（SR=1.0 → SR²/2=0.5）会让 SE 虚增 22%，是本主题最常见的实现错误。
本模块一律吃单期 SR，年化只在最后乘 √annual_days；strict=True 时 |SR|>0.5 直接报错。

600 日样本下 HAC 修正项自身有估计误差（小样本低估 SE → 过度拒绝），故并行提供
stationary bootstrap（Politis & Romano 1994）作独立第二意见，两者不一致时 warnings 会点出。

本模块不回答：① 寻优挑最好那组的 p 值无效（多重比较）→ 见 deflate_optimization_results；
② 样本外是否成立（需 walk-forward / PBO）；③ p<0.05 ≠ 能赚钱，扣成本后仍是样本内结论。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:                           # pragma: no cover - 仅供类型检查
    from pandas import DataFrame


# 标成 Any 而不是给 None 赋值加 type: ignore —— 那条 ignore 只在**装了** scipy
# 的环境里是必要的；CI 不装 scipy(workflow 只装 vnpy ruff mypy uv),配合
# ignore_missing_imports 这个模块解析成 Any,给 Any 赋 None 无需忽略注释,于是
# warn_unused_ignores 反过来把它判成错。改成先 import 到别名、再赋给一个带
# Any 标注的名字:两种环境都成立。不能写成先裸标注 `_scipy_stats: Any` 再
# import as 同名 —— 裸标注本身算一次定义,import 会被判 no-redef(CI 实测)。
try:                                        # scipy 只用于 t 分布与卡方分布
    from scipy import stats as _stats_mod
    _scipy_stats: Any = _stats_mod
    _HAS_SCIPY: bool = True
except ImportError:                         # pragma: no cover - 环境相关
    _scipy_stats = None
    _HAS_SCIPY = False


EULER_MASCHERONI: float = 0.5772156649015329

# |单期 SR| 超过这个值几乎肯定是误传了年化 SR（日 SR = 0.5 → 年化约 7.9）
IMPLAUSIBLE_PERIOD_SHARPE: float = 0.5


# 分布函数：正态用纯 math 实现（不引入 scipy 硬依赖），t / χ² 才可选 scipy

def norm_cdf(x: float) -> float:
    """标准正态 CDF。math.erfc 精度到机器精度，无需 scipy。"""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def norm_sf(x: float) -> float:
    """标准正态生存函数 1 − Φ(x)。大 x 时比 1 − norm_cdf(x) 精度高。"""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


_ACKLAM_A: tuple[float, ...] = (
    -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
    1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00,
)
_ACKLAM_B: tuple[float, ...] = (
    -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
    6.680131188771972e+01, -1.328068155288572e+01,
)
_ACKLAM_C: tuple[float, ...] = (
    -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
    -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00,
)
_ACKLAM_D: tuple[float, ...] = (
    7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
    3.754408661907416e+00,
)


def norm_ppf(p: float) -> float:
    """标准正态分位数 Φ⁻¹(p)。Acklam 有理逼近 + 一步 Halley 修正。"""
    if not 0.0 < p < 1.0:
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError(f"p 必须落在 (0, 1)，收到 {p}")

    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_ACKLAM_C[0] * q + _ACKLAM_C[1]) * q + _ACKLAM_C[2]) * q + _ACKLAM_C[3]) * q + _ACKLAM_C[4]) * q + _ACKLAM_C[5]) / \
            ((((_ACKLAM_D[0] * q + _ACKLAM_D[1]) * q + _ACKLAM_D[2]) * q + _ACKLAM_D[3]) * q + 1.0)
    elif p <= 1.0 - p_low:
        q = p - 0.5
        r = q * q
        x = (((((_ACKLAM_A[0] * r + _ACKLAM_A[1]) * r + _ACKLAM_A[2]) * r + _ACKLAM_A[3]) * r + _ACKLAM_A[4]) * r + _ACKLAM_A[5]) * q / \
            (((((_ACKLAM_B[0] * r + _ACKLAM_B[1]) * r + _ACKLAM_B[2]) * r + _ACKLAM_B[3]) * r + _ACKLAM_B[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((_ACKLAM_C[0] * q + _ACKLAM_C[1]) * q + _ACKLAM_C[2]) * q + _ACKLAM_C[3]) * q + _ACKLAM_C[4]) * q + _ACKLAM_C[5]) / \
            ((((_ACKLAM_D[0] * q + _ACKLAM_D[1]) * q + _ACKLAM_D[2]) * q + _ACKLAM_D[3]) * q + 1.0)

    # Halley 修正：把有理逼近的 ~1e-9 提到 ~1e-15
    err = ((1.0 - p) - norm_sf(x)) if x > 0.0 else (norm_cdf(x) - p)
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    if pdf > 0.0:
        u = err / pdf
        x = x - u / (1.0 + 0.5 * x * u)
    return x


def _two_sided_p(t_stat: float, dof: int, use_t_dist: bool) -> tuple[float, str]:
    """双尾 p 值 + 实际使用的分布名（fallback 必须留痕，见 CLAUDE.md §12）。"""
    if not math.isfinite(t_stat):
        return float("nan"), "undefined"
    if use_t_dist and _HAS_SCIPY and dof > 0:
        p = float(2.0 * _scipy_stats.t.sf(abs(t_stat), dof))
        return p, f"t(df={dof})"
    label = "normal"
    if use_t_dist and not _HAS_SCIPY:
        label = "normal (fallback: scipy 不可用)"
    return 2.0 * norm_sf(abs(t_stat)), label


def _critical_value(alpha: float, one_sided: bool, dof: int, use_t_dist: bool) -> float:
    """给定显著性水平的临界值。"""
    tail = alpha if one_sided else alpha / 2.0
    if use_t_dist and _HAS_SCIPY and dof > 0:
        return float(_scipy_stats.t.isf(tail, dof))
    return norm_ppf(1.0 - tail)


# 核心：Sharpe 的渐近方差因子 V

def newey_west_lags(n: int) -> int:
    """Newey-West 经验带宽 ⌊4·(n/100)^(2/9)⌋。n=611 → 5。"""
    if n <= 1:
        return 0
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def andrews_lags(u: np.ndarray, max_lags: int | None = None) -> int:
    """Andrews (1991) AR(1) plug-in 带宽（Bartlett 核）。"""
    arr = np.atleast_2d(np.asarray(u, dtype=float))
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    n = arr.shape[0]
    if n < 3:
        return 0
    if max_lags is None:
        max_lags = max(1, n - 2)

    numerator = 0.0
    denominator = 0.0
    for col in range(arr.shape[1]):
        series = arr[:, col]
        lagged = series[:-1]
        current = series[1:]
        denom_ar = float(lagged @ lagged)
        if denom_ar <= 0.0:
            continue
        rho = float(lagged @ current) / denom_ar
        rho = float(np.clip(rho, -0.97, 0.97))          # 防 (1−ρ) → 0 爆炸
        resid = current - rho * lagged
        sigma_sq = float(resid @ resid) / len(resid)
        numerator += 4.0 * rho * rho * sigma_sq ** 2 / ((1.0 - rho) ** 6 * (1.0 + rho) ** 2)
        denominator += sigma_sq ** 2 / (1.0 - rho) ** 4

    if denominator <= 0.0:
        return newey_west_lags(n)
    alpha1 = numerator / denominator
    if alpha1 <= 0.0:
        return 0
    bandwidth = 1.1447 * (alpha1 * n) ** (1.0 / 3.0)
    return int(min(max_lags, max(0, math.floor(bandwidth))))


def long_run_covariance(u: np.ndarray, lags: int) -> np.ndarray:
    """矩条件序列的 Bartlett 加权长期协方差 Ω（Newey-West 1987）。"""
    arr = np.atleast_2d(np.asarray(u, dtype=float))
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    n = arr.shape[0]
    omega: np.ndarray = arr.T @ arr / n
    for j in range(1, min(lags, n - 1) + 1):
        gamma: np.ndarray = arr[j:].T @ arr[:-j] / n
        weight = 1.0 - j / (lags + 1.0)
        omega = omega + weight * (gamma + gamma.T)
    return omega


def sharpe_variance_factor(
    returns: np.ndarray,
    sharpe_period: float,
    lags: int,
    strict: bool = True,
) -> float:
    """Sharpe 的渐近方差因子 V，使得 Var(SR̂) = V / n。"""
    r = np.asarray(returns, dtype=float).ravel()
    n = r.size
    if n < 2:
        return float("nan")
    if strict and abs(sharpe_period) > IMPLAUSIBLE_PERIOD_SHARPE:
        raise ValueError(
            f"sharpe_period={sharpe_period:.4f} 超过 {IMPLAUSIBLE_PERIOD_SHARPE}，"
            "几乎必然是误传了年化 Sharpe。本函数只接受单期 Sharpe。"
        )

    deviations = r - r.mean()
    m2 = float((deviations ** 2).mean())
    if m2 <= 0.0:
        return float("nan")
    sigma_pop = math.sqrt(m2)

    u = np.column_stack([deviations, deviations ** 2 - m2])
    omega = long_run_covariance(u, lags)
    grad = np.array([1.0 / sigma_pop, -sharpe_period / (2.0 * m2)])
    return float(grad @ omega @ grad)


def iid_normal_variance_factor(sharpe_period: float) -> float:
    """Lo (2002) iid 正态：V = 1 + SR²/2。"""
    return 1.0 + 0.5 * sharpe_period * sharpe_period


def iid_nonnormal_variance_factor(sharpe_period: float, skew: float, kurtosis: float) -> float:
    """Mertens (2002) iid 非正态：V = 1 − γ₃·SR + (γ₄−1)/4·SR²。"""
    return 1.0 - skew * sharpe_period + 0.25 * (kurtosis - 1.0) * sharpe_period ** 2


# Stationary bootstrap（独立第二意见）

def stationary_bootstrap_indices(
    n: int, mean_block: float, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Politis & Romano (1994) stationary bootstrap 的重抽样下标矩阵 (n_boot, n)。"""
    if n <= 0:
        raise ValueError("n 必须为正")
    if mean_block < 1.0:
        mean_block = 1.0
    p = 1.0 / mean_block
    fresh = rng.integers(0, n, size=(n_boot, n))
    jump = rng.random((n_boot, n)) < p
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = fresh[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(jump[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def default_mean_block(n: int) -> float:
    """stationary bootstrap 的默认平均块长 n^(1/3)（n=611 → 8.5）。"""
    return float(max(1.0, float(n) ** (1.0 / 3.0)))


def bootstrap_sharpe_test(
    returns: np.ndarray,
    risk_free_period: float = 0.0,
    n_boot: int = 999,
    mean_block: float | None = None,
    seed: int = 20260725,
) -> tuple[float, float]:
    """零假设 SR=0 下的 stationary bootstrap 双尾 p 值 + bootstrap SE。"""
    r = np.asarray(returns, dtype=float).ravel()
    n = r.size
    if n < 8:
        return float("nan"), float("nan")
    if mean_block is None:
        mean_block = default_mean_block(n)

    observed_sigma = float(r.std(ddof=1))
    if observed_sigma <= 0.0:
        return float("nan"), float("nan")
    observed_sr = (float(r.mean()) - risk_free_period) / observed_sigma

    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, mean_block, n_boot, rng)

    # SE：直接重抽样原序列
    raw = r[idx]
    raw_sigma = raw.std(axis=1, ddof=1)
    valid = raw_sigma > 0.0
    raw_sr = np.full(n_boot, np.nan)
    raw_sr[valid] = (raw[valid].mean(axis=1) - risk_free_period) / raw_sigma[valid]
    bootstrap_se = float(np.nanstd(raw_sr, ddof=1))

    # p 值：强加 SR=0（等价于均值 = rf）
    centered = r - r.mean() + risk_free_period
    null = centered[idx]
    null_sigma = null.std(axis=1, ddof=1)
    valid_null = null_sigma > 0.0
    null_sr = np.full(n_boot, np.nan)
    null_sr[valid_null] = (null[valid_null].mean(axis=1) - risk_free_period) / null_sigma[valid_null]

    exceed = int(np.sum(np.abs(null_sr[valid_null]) >= abs(observed_sr)))
    p_value = (1.0 + exceed) / (1.0 + int(valid_null.sum()))
    return p_value, bootstrap_se


# 序列相关诊断

def autocorrelations(returns: np.ndarray, max_lag: int) -> np.ndarray:
    """样本自相关 ρ₁..ρ_max_lag。"""
    r = np.asarray(returns, dtype=float).ravel()
    n = r.size
    dev = r - r.mean()
    denom = float(dev @ dev)
    if denom <= 0.0 or n <= 1:
        return np.zeros(max_lag, dtype=float)
    out = np.empty(min(max_lag, n - 1), dtype=float)
    for k in range(1, out.size + 1):
        out[k - 1] = float(dev[k:] @ dev[:-k]) / denom
    if out.size < max_lag:
        out = np.concatenate([out, np.zeros(max_lag - out.size)])
    return out


def ljung_box(returns: np.ndarray, lags: int = 10) -> tuple[float, float]:
    """Ljung-Box Q 检验：序列相关是否显著。返回 (Q, p)。"""
    r = np.asarray(returns, dtype=float).ravel()
    n = r.size
    if n <= lags + 1:
        return float("nan"), float("nan")
    rho = autocorrelations(r, lags)
    q = float(n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, lags + 1))))
    if not _HAS_SCIPY:
        return q, float("nan")
    return q, float(_scipy_stats.chi2.sf(q, lags))


# 主结果结构

@dataclass(frozen=True)
class SharpeInference:
    """Sharpe 的点估计 + 不确定性 + 判定，以及全部中间量（便于核对）。"""

    # 点估计
    sharpe_period: float                # 单期（日）Sharpe
    sharpe_annual: float                # 年化 Sharpe，= sharpe_period × √annual_days

    # 选定方法下的不确定性
    method: str                         # "iid_normal" | "iid_nonnormal" | "hac"
    standard_error_period: float
    standard_error_annual: float
    t_stat: float
    p_value: float                      # 双尾
    p_value_one_sided: float            # 单尾（H1: SR > 0），SR<0 时 = 1 − 单尾
    ci_low_annual: float
    ci_high_annual: float
    confidence: float                   # 区间置信度，如 0.95
    significant: bool                   # 双尾 p < 1 − confidence
    dist_used: str

    # 三档 SE 并列（年化口径，便于直接比较）
    se_iid_normal_annual: float
    se_iid_nonnormal_annual: float
    se_hac_annual: float
    hac_inflation: float                # se_hac / se_iid_normal，>1 表示自相关在放大不确定性

    # bootstrap 第二意见
    bootstrap_p_value: float
    bootstrap_se_annual: float
    bootstrap_draws: int

    # 诊断量
    n_periods: int
    annual_days: int
    risk_free_period: float
    return_mean_period: float
    return_std_period: float
    skew: float
    kurtosis: float                     # 非超额峰度，正态 = 3
    autocorr_lag1: float
    ljung_box_stat: float
    ljung_box_p: float
    hac_lags: int

    # 这个样本量下"要多大 Sharpe 才显著"
    required_sharpe_annual: float

    warnings: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def summary_line(self) -> str:
        """一行结论，形如 ``sharpe = 0.92, SE = 0.41, t = 2.24, p = 0.025 → 显著``。"""
        verdict = "显著" if self.significant else "不显著"
        return (
            f"sharpe = {self.sharpe_annual:.2f}, "
            f"SE = {self.standard_error_annual:.2f}, "
            f"t = {self.t_stat:.2f}, "
            f"p = {self.p_value:.3f} "
            f"→ {verdict}（{int(self.confidence * 100)}% 水平，{self.method}）"
        )

    def report(self) -> str:
        """多行报告，给 `calculate_statistics(output=True)` 直接打印。"""
        lines = [
            self.summary_line(),
            f"  年化 {int(self.confidence * 100)}% 置信区间：[{self.ci_low_annual:.2f}, {self.ci_high_annual:.2f}]",
            f"  样本 n={self.n_periods}（≈{self.n_periods / self.annual_days:.2f} 年），"
            f"该样本量下需 Sharpe ≥ {self.required_sharpe_annual:.2f} 才显著",
            f"  SE 三档（年化）：iid正态 {self.se_iid_normal_annual:.2f} / "
            f"iid非正态 {self.se_iid_nonnormal_annual:.2f} / "
            f"HAC(lags={self.hac_lags}) {self.se_hac_annual:.2f}"
            f"（自相关放大 ×{self.hac_inflation:.2f}）",
            f"  bootstrap 第二意见：p = {self.bootstrap_p_value:.3f}，"
            f"SE = {self.bootstrap_se_annual:.2f}（{self.bootstrap_draws} 次重抽样）",
            f"  分布诊断：偏度 {self.skew:.2f}，峰度 {self.kurtosis:.2f}，"
            f"ρ₁ = {self.autocorr_lag1:.3f}，Ljung-Box p = {self.ljung_box_p:.3f}",
        ]
        lines.extend(f"  ⚠ {w}" for w in self.warnings)
        return "\n".join(lines)


# 主入口

def sharpe_inference(
    returns: Sequence[float] | np.ndarray,
    annual_days: int = 240,
    risk_free_period: float = 0.0,
    method: str = "hac",
    lags: int | str = "newey_west",
    confidence: float = 0.95,
    use_t_dist: bool = True,
    ddof: int = 1,
    n_boot: int = 999,
    mean_block: float | None = None,
    seed: int = 20260725,
) -> SharpeInference:
    """对一段【单期收益序列】做 Sharpe 的标准误与显著性检验。"""
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    n = r.size
    warn: list[str] = []

    if n < 8:
        raise ValueError(f"样本量 n={n} 太小，无法做 Sharpe 推断（至少需要 8 期）")
    if annual_days <= 0:
        raise ValueError(f"annual_days 必须为正，收到 {annual_days}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence 必须落在 (0,1)，收到 {confidence}")

    alpha = 1.0 - confidence
    mean = float(r.mean())
    sigma = float(r.std(ddof=ddof))
    scale = math.sqrt(annual_days)

    if sigma <= 0.0:
        raise ValueError("收益序列标准差为 0，Sharpe 无定义")

    sharpe_period = (mean - risk_free_period) / sigma
    sharpe_annual = sharpe_period * scale

    # 高阶矩（总体口径，与渐近理论一致）
    dev = r - mean
    m2 = float((dev ** 2).mean())
    m3 = float((dev ** 3).mean())
    m4 = float((dev ** 4).mean())
    skew = m3 / m2 ** 1.5 if m2 > 0 else 0.0
    kurt = m4 / m2 ** 2 if m2 > 0 else 3.0

    # 三档方差因子
    v_iid_normal = iid_normal_variance_factor(sharpe_period)
    v_iid_nonnormal = iid_nonnormal_variance_factor(sharpe_period, skew, kurt)

    u = np.column_stack([dev, dev ** 2 - m2])
    if isinstance(lags, str):
        if lags == "newey_west":
            hac_lags = newey_west_lags(n)
        elif lags == "andrews":
            hac_lags = andrews_lags(u)
        else:
            raise ValueError(f"未知的 lags 规则 {lags!r}，可用：'newey_west' / 'andrews' / 整数")
    else:
        hac_lags = int(lags)
    hac_lags = max(0, min(hac_lags, n - 2))

    if abs(sharpe_period) > IMPLAUSIBLE_PERIOD_SHARPE:
        warn.append(
            f"单期 Sharpe {sharpe_period:.3f} 异常大（年化 {sharpe_annual:.1f}），"
            "请确认 returns 确实是单期收益而非已年化的序列"
        )
    v_hac = sharpe_variance_factor(r, sharpe_period, hac_lags, strict=False)

    def _se(v: float) -> float:
        return math.sqrt(v / n) if v > 0 and math.isfinite(v) else float("nan")

    se_iid_normal = _se(v_iid_normal)
    se_iid_nonnormal = _se(v_iid_nonnormal)
    se_hac = _se(v_hac)

    if not math.isfinite(se_hac) or se_hac <= 0.0:
        warn.append("HAC 方差估计非正（小样本 + 长带宽的已知病理），已回退 iid 非正态 SE")
        se_hac = se_iid_nonnormal
        v_hac = v_iid_nonnormal

    chosen = {
        "hac": se_hac,
        "iid_nonnormal": se_iid_nonnormal,
        "iid_normal": se_iid_normal,
    }
    if method not in chosen:
        raise ValueError(f"未知 method {method!r}，可用：{sorted(chosen)}")
    se_period = chosen[method]

    dof = n - 1
    t_stat = sharpe_period / se_period if se_period > 0 else float("nan")
    p_two, dist_used = _two_sided_p(t_stat, dof, use_t_dist)
    p_one = p_two / 2.0 if t_stat > 0 else 1.0 - p_two / 2.0

    crit = _critical_value(alpha, one_sided=False, dof=dof, use_t_dist=use_t_dist)
    ci_low = (sharpe_period - crit * se_period) * scale
    ci_high = (sharpe_period + crit * se_period) * scale

    # bootstrap 第二意见
    if n_boot > 0:
        boot_p, boot_se = bootstrap_sharpe_test(
            r, risk_free_period=risk_free_period, n_boot=n_boot,
            mean_block=mean_block, seed=seed,
        )
    else:
        boot_p, boot_se = float("nan"), float("nan")

    lb_stat, lb_p = ljung_box(r, lags=min(10, max(1, n // 5)))
    rho1 = float(autocorrelations(r, 1)[0])

    hac_inflation = se_hac / se_iid_normal if se_iid_normal > 0 else float("nan")

    # One-sided, not two-sided. The decision-relevant hypothesis for a
    # long-only CTA is SR > 0; a two-sided test also calls a SIGNIFICANTLY
    # NEGATIVE Sharpe "significant". Measured: an annualized Sharpe of -5.16
    # came back significant=True and was written into the statistics dict,
    # where nothing downstream distinguishes the sign. p_two stays available as
    # a number for anyone who wants it; it just no longer drives the verdict.
    significant = bool(math.isfinite(p_one) and p_one < alpha and sharpe_annual > 0)

    required = required_sharpe(
        n_periods=n, annual_days=annual_days, alpha=alpha, one_sided=False,
        skew=skew, kurtosis=kurt, se_inflation=hac_inflation if method == "hac" else 1.0,
        dof=dof, use_t_dist=use_t_dist,
    )

    # ── 告警 ──────────────────────────────────────────────────────────
    if n < 250:
        warn.append(f"n={n} 不足一年，渐近正态近似与 HAC 修正都不可靠，结论只当参考")
    if math.isfinite(lb_p) and lb_p < 0.05:
        warn.append(f"Ljung-Box p={lb_p:.3f} < 0.05：序列相关显著，iid 公式会低估 SE，必须看 HAC 列")
    if math.isfinite(hac_inflation) and hac_inflation > 1.3:
        warn.append(f"HAC 把 SE 放大 ×{hac_inflation:.2f}，自相关是这条曲线不确定性的主要来源")
    if hac_lags > n // 20:
        warn.append(f"HAC 带宽 {hac_lags} 相对 n={n} 偏长，长期方差估计噪声大")
    if abs(kurt) > 10:
        warn.append(f"峰度 {kurt:.1f} 极高，尾部风险未被 Sharpe 捕捉；SE 也偏乐观")
    if (
        math.isfinite(boot_p)
        and math.isfinite(p_two)
        and (boot_p < alpha) != (p_two < alpha)
    ):
        warn.append(
            f"解析法 p={p_two:.3f} 与 bootstrap p={boot_p:.3f} 在 {alpha:.0%} 水平结论相反，"
            "按更保守的一方处理"
        )
    if significant and n < 2 * annual_days:
        warn.append("样本不足两年就判显著，样本外失效的历史先例很多，不要单凭这一条放大仓位")

    return SharpeInference(
        sharpe_period=sharpe_period,
        sharpe_annual=sharpe_annual,
        method=method,
        standard_error_period=se_period,
        standard_error_annual=se_period * scale,
        t_stat=t_stat,
        p_value=p_two,
        p_value_one_sided=p_one,
        ci_low_annual=ci_low,
        ci_high_annual=ci_high,
        confidence=confidence,
        significant=significant,
        dist_used=dist_used,
        se_iid_normal_annual=se_iid_normal * scale,
        se_iid_nonnormal_annual=se_iid_nonnormal * scale,
        se_hac_annual=se_hac * scale,
        hac_inflation=hac_inflation,
        bootstrap_p_value=boot_p,
        bootstrap_se_annual=boot_se * scale,
        bootstrap_draws=int(n_boot),
        n_periods=n,
        annual_days=annual_days,
        risk_free_period=risk_free_period,
        return_mean_period=mean,
        return_std_period=sigma,
        skew=skew,
        kurtosis=kurt,
        autocorr_lag1=rho1,
        ljung_box_stat=lb_stat,
        ljung_box_p=lb_p,
        hac_lags=hac_lags,
        required_sharpe_annual=required,
        warnings=tuple(warn),
    )


# "要多大 Sharpe 才显著" —— 闭式解

def required_sharpe(
    n_periods: int,
    annual_days: int = 240,
    alpha: float = 0.05,
    one_sided: bool = False,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    se_inflation: float = 1.0,
    dof: int | None = None,
    use_t_dist: bool = True,
) -> float:
    """给定样本量，达到 alpha 水平显著所需的【年化】Sharpe 下限。"""
    if n_periods <= 1:
        return float("nan")
    if dof is None:
        dof = n_periods - 1
    z = _critical_value(alpha, one_sided=one_sided, dof=dof, use_t_dist=use_t_dist)
    f = se_inflation if se_inflation and math.isfinite(se_inflation) and se_inflation > 0 else 1.0

    a = n_periods / (f * f) - z * z * (kurtosis - 1.0) / 4.0
    b = z * z * skew
    c = -z * z

    if a <= 0:                              # 分母被高阶矩吃穿：门槛发散
        return float("inf")
    disc = b * b - 4.0 * a * c
    if disc < 0:                            # pragma: no cover - c<0 时不可能发生
        return float("nan")
    sr_period = (-b + math.sqrt(disc)) / (2.0 * a)
    return sr_period * math.sqrt(annual_days)


def sharpe_significance_table(
    n_periods: int,
    annual_days: int = 240,
    inflations: Iterable[float] = (1.0, 1.1, 1.2, 1.3, 1.5),
    alpha: float = 0.05,
) -> str:
    """打印"不同自相关强度下所需 Sharpe"的小表，用于交易前的期望管理。"""
    years = n_periods / annual_days
    rows = [
        f"样本 n={n_periods}（≈{years:.2f} 年，annual_days={annual_days}），"
        f"{alpha:.0%} 水平所需年化 Sharpe：",
        "  SE放大f   隐含AR(1)ρ   双尾    单尾",
    ]
    for f in inflations:
        rho = (f * f - 1.0) / (f * f + 1.0)          # f = √((1+ρ)/(1−ρ)) 反解
        two = required_sharpe(n_periods, annual_days, alpha, False, se_inflation=f)
        one = required_sharpe(n_periods, annual_days, alpha, True, se_inflation=f)
        rows.append(f"  ×{f:<8.2f} {rho:<12.2f} {two:<7.2f} {one:.2f}")
    return "\n".join(rows)


# 多重比较：PSR / MinTRL / DSR —— 参数寻优必须过这一关

def probabilistic_sharpe_ratio(
    sharpe_period: float,
    n_periods: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_period: float = 0.0,
) -> float:
    """Bailey & López de Prado (2012) PSR：P(真实 SR > benchmark)。"""
    if n_periods <= 1:
        return float("nan")
    v = iid_nonnormal_variance_factor(sharpe_period, skew, kurtosis)
    if v <= 0:
        return float("nan")
    z = (sharpe_period - benchmark_period) * math.sqrt(n_periods - 1) / math.sqrt(v)
    return norm_cdf(z)


def minimum_track_record_length(
    sharpe_period: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_period: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """MinTRL：要让 PSR 达到 confidence，最少需要多少期数据。"""
    excess = sharpe_period - benchmark_period
    if excess <= 0:
        return float("inf")
    v = iid_nonnormal_variance_factor(sharpe_period, skew, kurtosis)
    if v <= 0:
        return float("nan")
    return 1.0 + v * (norm_ppf(confidence) / excess) ** 2


def expected_max_sharpe(n_trials: int, sharpe_std: float) -> float:
    """N 次独立试验中，即使真实 SR 全为 0，最大 SR 的期望值。"""
    if n_trials <= 1 or sharpe_std <= 0:
        return 0.0
    n = float(n_trials)
    term1 = (1.0 - EULER_MASCHERONI) * norm_ppf(1.0 - 1.0 / n)
    term2 = EULER_MASCHERONI * norm_ppf(1.0 - 1.0 / (n * math.e))
    return sharpe_std * (term1 + term2)


def deflated_sharpe_ratio(
    sharpe_period: float,
    n_periods: int,
    n_trials: int,
    sharpe_std: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """DSR：把"试了 N 组参数"折进去之后，真实 SR > 0 的概率。"""
    benchmark = expected_max_sharpe(n_trials, sharpe_std)
    return probabilistic_sharpe_ratio(
        sharpe_period, n_periods, skew, kurtosis, benchmark_period=benchmark
    )


#: 目标函数键里，row[1] 确实是（年化）夏普的那些。DSR 的整套推导都建立在
#: "这些试验值是夏普" 之上，喂别的量纲进去算出来的数没有任何统计含义。
SHARPE_TARGET_NAMES: frozenset[str] = frozenset({
    "sharpe_ratio", "ewm_sharpe", "robust_sharpe",
})


def deflate_optimization_results(
    results: Sequence[tuple],
    n_periods: int,
    target_name: str,
    annual_days: int = 240,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> dict[str, float | int | bool]:
    """直接吃 vnpy `run_bf_optimization` / `run_ga_optimization` 的返回值。

    `target_name` 必填，且必须是夏普族目标函数。vnpy 的 `evaluate()` 返回
    ``(setting, statistics.get(target_name, 0), statistics)`` —— row[1] 是**用户
    在寻优对话框里选的那个键**，可以是 `total_return`、`return_drawdown_ratio`
    等任意量纲。本函数把 row[1] 当年化夏普做 DSR 反缩。

    实测：把一组 `return_drawdown_ratio`（均值约 4.0，本项目面板里就有这个键）
    喂进来，得到 best_sharpe_annual=8.98、deflated_sharpe_ratio=1.0、
    **trustworthy=True**，零报错零告警 —— 也就是说，这道本该挡住多重比较的闸，
    会对错误输入盖章放行。所以宁可在这里 raise，也不要静默出一个假的绿灯。
    """
    if target_name not in SHARPE_TARGET_NAMES:
        raise ValueError(
            f"deflate_optimization_results 只能用于夏普族寻优目标，收到 {target_name!r}。"
            f"DSR 把 row[1] 当年化夏普处理，喂入 {target_name!r} 这种不同量纲的值"
            f"会算出一个没有统计含义、却看起来可信的数字。"
            f"可用目标：{', '.join(sorted(SHARPE_TARGET_NAMES))}"
        )
    if not results:
        return {"n_trials": 0, "deflated_sharpe_ratio": float("nan"), "trustworthy": False}

    values = np.array(
        [float(row[1]) for row in results if math.isfinite(float(row[1]))], dtype=float
    )
    if values.size == 0:
        return {"n_trials": 0, "deflated_sharpe_ratio": float("nan"), "trustworthy": False}

    scale = math.sqrt(annual_days)
    period_values = values / scale
    best_period = float(period_values.max())
    sharpe_std = float(period_values.std(ddof=1)) if period_values.size > 1 else 0.0

    dsr = deflated_sharpe_ratio(
        best_period, n_periods, int(values.size), sharpe_std, skew, kurtosis
    )
    benchmark = expected_max_sharpe(int(values.size), sharpe_std)
    return {
        "n_trials": int(values.size),
        "best_sharpe_annual": best_period * scale,
        "sharpe_std_annual": sharpe_std * scale,
        "expected_max_sharpe_annual": benchmark * scale,
        "deflated_sharpe_ratio": dsr,
        "trustworthy": bool(math.isfinite(dsr) and dsr >= 0.95),
    }


# vnpy 接口层

def inference_from_daily_df(
    daily_df: DataFrame | None,
    annual_days: int = 240,
    risk_free: float = 0.0,
    method: str = "hac",
    confidence: float = 0.95,
    n_boot: int = 999,
    return_column: str = "return",
) -> SharpeInference:
    """从 vnpy 的 ``daily_df`` 直接出推断结果。

    ``risk_free`` 沿用 ``BacktestingEngine`` 的约定：**百分数刻度的年化无风险利率**
    （2 表示 2%），与面板上的 ``sharpe_ratio`` 完全同源。而 ``daily_df["return"]``
    是分数刻度的对数收益，故这里必须除以 100 才能与之相减——漏掉这个 /100 会让
    risk_free=2 把 Sharpe 推偏约 180 个单位。由 test_risk_free_matches_upstream 钉住。
    """
    if daily_df is None or not hasattr(daily_df, "__getitem__"):
        raise ValueError("daily_df 为空或不是 DataFrame，无法做 Sharpe 推断")
    returns = np.asarray(daily_df[return_column], dtype=float)
    risk_free_period = (
        risk_free / math.sqrt(annual_days) / 100.0 if annual_days > 0 else 0.0
    )
    return sharpe_inference(
        returns,
        annual_days=annual_days,
        risk_free_period=risk_free_period,
        method=method,
        confidence=confidence,
        n_boot=n_boot,
    )


def statistics_fields(inference: SharpeInference) -> dict[str, float | int | bool | str]:
    """摊平成可直接并入 ``calculate_statistics()`` 返回 dict 的键值对。"""

    def p_or_insignificant(value: float) -> float:
        return value if math.isfinite(value) else 1.0

    def num(value: float) -> float:
        return value if math.isfinite(value) else 0.0

    return {
        "sharpe_se": num(inference.standard_error_annual),
        "sharpe_tstat": num(inference.t_stat),
        "sharpe_pvalue": p_or_insignificant(inference.p_value),
        "sharpe_pvalue_one_sided": p_or_insignificant(inference.p_value_one_sided),
        "sharpe_ci_low": num(inference.ci_low_annual),
        "sharpe_ci_high": num(inference.ci_high_annual),
        "sharpe_significant": inference.significant,
        "sharpe_method": inference.method,
        "sharpe_hac_lags": inference.hac_lags,
        "sharpe_hac_inflation": num(inference.hac_inflation),
        "sharpe_bootstrap_pvalue": p_or_insignificant(inference.bootstrap_p_value),
        "sharpe_required_for_significance": num(inference.required_sharpe_annual),
        "sharpe_skew": num(inference.skew),
        "sharpe_kurtosis": num(inference.kurtosis),
        "sharpe_autocorr_lag1": num(inference.autocorr_lag1),
        "sharpe_ljung_box_p": p_or_insignificant(inference.ljung_box_p),
    }
