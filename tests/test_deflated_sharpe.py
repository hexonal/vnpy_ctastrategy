"""Deflated Sharpe Ratio 的回归测试。

定义来自 Bailey & López de Prado (2014)。除了逐个函数的公式验证，本文件的重点是
【方法本身的有效性检验】—— 用蒙特卡洛证明这套统计主张确实成立：

  test_random_trials_are_never_significant_under_dsr
      纯随机序列里挑最好的一组，DSR 必须判"不显著"。这是全套测试的核心：
      如果一个显著性检验对随机数据也说"显著"，那它没有任何价值。
      同一份数据下不做 deflation 的 PSR 会有 ~99.8% 的假阳性率，对照鲜明。

  test_dsr_is_conservative_when_trials_are_correlated
      试验相关（真实寻优的常态：相邻参数的收益曲线高度相关）时，假阳性率
      仍不超过名义水平 —— 兑现模块 docstring 里"相关会让 DSR 偏保守"的承诺。

  test_dsr_still_detects_genuine_skill
      功效测试。没有这一条，"永远判不显著"也能通过上面两条。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from vnpy_ctastrategy.deflated_sharpe import (
    DEFAULT_CONFIDENCE,
    EULER_MASCHERONI,
    NORMAL_KURTOSIS,
    DeflatedSharpeResult,
    deflate_optimization,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    format_required_sharpe_table,
    main,
    minimum_sharpe_for_confidence,
    normal_cdf,
    normal_ppf,
    probabilistic_sharpe_ratio,
    required_sharpe_table,
    return_moments,
    sharpe_significance,
    sharpe_standard_error,
    sharpe_variance_factor,
    trial_sharpes_from_optimization,
)

HK_ANNUAL_DAYS = 247            # 港股口径；美股应传 252
SAMPLE_DAYS = 600               # 本项目典型回测样本长度
NOMINAL_ALPHA = 1.0 - DEFAULT_CONFIDENCE        # 名义显著性水平 5%


# ── 工具 ───────────────────────────────────────────────────────────────

def make_optimization_result(
    setting: dict[str, Any],
    sharpe: float,
    total_days: int = SAMPLE_DAYS,
    skew: float = 0.0,
    kurtosis: float = NORMAL_KURTOSIS,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    """构造一条与 vnpy_ctastrategy.backtesting.evaluate() 同形的寻优结果。

    真实形状（读 backtesting.py::evaluate 的 return 语句得来）：
        (setting, target_value, statistics)
    """
    statistics = {
        "total_days": total_days,
        "sharpe_ratio": sharpe,
        # sharpe_inference.py 写进面板的键名，deflate_optimization 直接读它
        "sharpe_skew": skew,
        "sharpe_kurtosis": kurtosis,
        "total_return": sharpe * 10.0,
    }
    return (setting, sharpe, statistics)


def null_trial_sharpes(
    rng: np.random.Generator,
    n_trials: int,
    n_obs: int,
    drift: float = 0.0,
    corr: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成 n_trials 条零技能（或带指定漂移）的日收益，返回 (收益矩阵, 年化夏普)。

    corr 为试验间的等相关系数：corr=0 是独立试验，corr>0 模拟"相邻参数的收益
    曲线高度相关"这一真实寻优的常态。
    """
    common = rng.standard_normal(n_obs)
    idiosyncratic = rng.standard_normal((n_trials, n_obs))
    returns = (
        idiosyncratic * math.sqrt(1.0 - corr) + common * math.sqrt(corr)
    ) * 0.01 + drift

    mean = returns.mean(axis=1)
    std = returns.std(axis=1, ddof=1)
    sharpes = mean / std * math.sqrt(HK_ANNUAL_DAYS)
    return returns, sharpes


def select_best_and_deflate(
    returns: np.ndarray, sharpes: np.ndarray, n_trials: int
) -> DeflatedSharpeResult:
    """按"挑夏普最高的那组"这一真实寻优行为选参数，再算 DSR。"""
    best = int(np.argmax(sharpes))
    moments = return_moments(returns[best])
    return deflated_sharpe_ratio(
        observed_sharpe=float(sharpes[best]),
        trial_sharpe_std=float(np.std(sharpes, ddof=1)),
        n_trials=n_trials,
        n_obs=returns.shape[1],
        skew=moments.skew,
        kurtosis=moments.kurtosis,
        annual_days=HK_ANNUAL_DAYS,
    )


# ── 基础：正态分布用标准库而非 scipy ───────────────────────────────────

def test_normal_dist_matches_scipy() -> None:
    """本模块用 statistics.NormalDist 替代 scipy.stats.norm，此处钉住等价性。

    scipy 只在测试环境需要（venv 里有 1.18.0），生产代码不依赖它 —— 本包
    pyproject 的 dependencies 只有 vnpy/pandas/plotly。
    """
    scipy_stats = pytest.importorskip("scipy.stats")

    for x in (-8.0, -5.0, -3.0, -1.0, 0.0, 0.5, 1.6449, 3.0, 5.0, 8.0):
        assert normal_cdf(x) == pytest.approx(scipy_stats.norm.cdf(x), abs=1e-14)

    for p in (1e-12, 1e-6, 0.001, 0.05, 0.5, 0.95, 0.999, 1 - 1e-9):
        assert normal_ppf(p) == pytest.approx(scipy_stats.norm.ppf(p), abs=1e-12)


def test_normal_ppf_rejects_out_of_range() -> None:
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            normal_ppf(bad)


# ── 收益矩 ─────────────────────────────────────────────────────────────

def test_return_moments_uses_population_estimators() -> None:
    """偏度/峰度必须是总体（plug-in）估计量，不是 pandas 的无偏修正版。"""
    rng = np.random.default_rng(0)
    values = rng.standard_normal(500)

    moments = return_moments(values)
    deviations = values - values.mean()
    m2 = np.mean(deviations ** 2)

    assert moments.skew == pytest.approx(np.mean(deviations ** 3) / m2 ** 1.5, rel=1e-12)
    assert moments.kurtosis == pytest.approx(np.mean(deviations ** 4) / m2 ** 2, rel=1e-12)
    # std 反过来必须是 ddof=1，才能和 vnpy 的 df["return"].std() 对齐
    assert moments.std == pytest.approx(np.std(values, ddof=1), rel=1e-12)


def test_kurtosis_is_pearson_not_fisher() -> None:
    """γ₄ 是非超额峰度（正态≈3）。pandas 的 .kurt() 返回超额峰度，混用会算错分母。"""
    rng = np.random.default_rng(1)
    moments = return_moments(rng.standard_normal(20000))

    assert moments.kurtosis == pytest.approx(3.0, abs=0.15)
    assert moments.excess_kurtosis == pytest.approx(0.0, abs=0.15)
    assert moments.kurtosis - moments.excess_kurtosis == pytest.approx(3.0)


def test_population_moments_guarantee_positive_variance_factor() -> None:
    """γ₄ ≥ γ₃² + 1 是经验分布的恒等式，它保证 PSR 分母永远开得出根号。

    这正是选用总体估计量的原因 —— 用 pandas 的无偏修正版本可以违反此不等式，
    小样本 + 强偏态时会让分母变成 NaN。
    """
    rng = np.random.default_rng(2)
    for size in (5, 10, 30, 600):
        for _ in range(200):
            sample = rng.standard_exponential(size) ** 2      # 强偏态、厚尾
            moments = return_moments(sample)
            assert moments.kurtosis >= moments.skew ** 2 + 1.0 - 1e-9
            assert sharpe_variance_factor(0.1, moments.skew, moments.kurtosis) > 0.0


def test_return_moments_needs_two_observations() -> None:
    with pytest.raises(ValueError):
        return_moments([0.01])


def test_return_moments_handles_constant_series() -> None:
    """常数序列的偏度/峰度无定义，退回正态值而不是抛 ZeroDivisionError。"""
    moments = return_moments([0.01] * 50)
    assert moments.std == 0.0
    assert moments.skew == 0.0
    assert moments.kurtosis == NORMAL_KURTOSIS


# ── 第一层：夏普标准误 ─────────────────────────────────────────────────

def test_sharpe_standard_error_reproduces_project_anchor() -> None:
    """钉住项目里实测过的那条曲线：134 日、年化夏普 −1.68，标准误约 1.36。

    这就是本模块存在的理由 —— 那条曲线连"真实夏普是负的"都没到 2σ，
    而现有面板上的任何一个指标都不会告诉你这件事。
    """
    n_obs = 134
    sharpe_annual = -1.68
    root = math.sqrt(HK_ANNUAL_DAYS)

    se_annual = sharpe_standard_error(sharpe_annual / root, n_obs) * root

    assert se_annual == pytest.approx(1.37, abs=0.01)
    assert abs(sharpe_annual) < 2.0 * se_annual        # 连符号都没到 2σ


def test_sharpe_standard_error_matches_classic_formula_under_normality() -> None:
    """正态、夏普接近 0 时退化为经典的 √(1/(T−1))。"""
    assert sharpe_standard_error(0.0, 601) == pytest.approx(1.0 / math.sqrt(600), rel=1e-12)


def test_sharpe_standard_error_shrinks_with_sample_size() -> None:
    errors = [sharpe_standard_error(0.05, n) for n in (100, 250, 600, 2000)]
    assert errors == sorted(errors, reverse=True)


def test_negative_skew_inflates_standard_error() -> None:
    """负偏度 + 正夏普 → 标准误变大。"平时小赚、偶尔巨亏"堆出来的夏普不可信。"""
    baseline = sharpe_standard_error(0.08, SAMPLE_DAYS, skew=0.0, kurtosis=3.0)
    negative_skew = sharpe_standard_error(0.08, SAMPLE_DAYS, skew=-1.0, kurtosis=6.0)
    positive_skew = sharpe_standard_error(0.08, SAMPLE_DAYS, skew=+1.0, kurtosis=6.0)

    assert negative_skew > baseline > positive_skew


def test_variance_factor_rejects_impossible_moment_pairs() -> None:
    """(skew, kurtosis) 不自洽时明确报错，而不是返回 NaN 往下游传。"""
    with pytest.raises(ValueError, match="γ₄ ≥ γ₃"):
        sharpe_variance_factor(sharpe=5.0, skew=0.0, kurtosis=0.0)


def test_sharpe_standard_error_needs_two_observations() -> None:
    with pytest.raises(ValueError):
        sharpe_standard_error(0.1, 1)


# ── 第一层：PSR ────────────────────────────────────────────────────────

def test_psr_reduces_to_one_sided_normal_test() -> None:
    """正态假设下 PSR(0) = Φ(SR·√(T−1))，即经典的单边 t 检验（大样本近似）。"""
    sharpe_daily = 0.04
    n_obs = 601
    expected = normal_cdf(sharpe_daily * math.sqrt(600) / math.sqrt(1 + 0.5 * sharpe_daily ** 2))

    assert probabilistic_sharpe_ratio(sharpe_daily, 0.0, n_obs) == pytest.approx(expected, rel=1e-12)


def test_psr_is_one_half_at_the_benchmark() -> None:
    """观测夏普恰好等于基准时，PSR = 0.5 —— 完全无信息。"""
    assert probabilistic_sharpe_ratio(0.05, 0.05, SAMPLE_DAYS) == pytest.approx(0.5, abs=1e-12)


def test_psr_increases_with_sharpe_and_sample_size() -> None:
    assert (
        probabilistic_sharpe_ratio(0.02, 0.0, SAMPLE_DAYS)
        < probabilistic_sharpe_ratio(0.06, 0.0, SAMPLE_DAYS)
    )
    assert (
        probabilistic_sharpe_ratio(0.04, 0.0, 150)
        < probabilistic_sharpe_ratio(0.04, 0.0, SAMPLE_DAYS)
    )


def test_psr_penalises_negative_skew_and_fat_tails() -> None:
    clean = probabilistic_sharpe_ratio(0.06, 0.0, SAMPLE_DAYS, skew=0.0, kurtosis=3.0)
    ugly = probabilistic_sharpe_ratio(0.06, 0.0, SAMPLE_DAYS, skew=-1.5, kurtosis=12.0)
    assert ugly < clean


# ── 第二层：期望最大夏普 SR* ───────────────────────────────────────────

def test_expected_max_sharpe_matches_monte_carlo() -> None:
    """SR* 是 N 个 iid N(0,V) 里最大值的期望，此处用蒙特卡洛验证 Bailey 的近似式。"""
    rng = np.random.default_rng(20260725)
    trial_std = 0.6

    for n_trials in (5, 20, 100, 1000):
        draws = rng.standard_normal((4000, n_trials)) * trial_std
        empirical = float(np.mean(draws.max(axis=1)))
        assert expected_max_sharpe(trial_std, n_trials) == pytest.approx(empirical, abs=0.05)


def test_expected_max_sharpe_uses_euler_mascheroni_formula() -> None:
    """逐字复现 Bailey (2014) 式 (5)，防止有人"优化"掉常数项。"""
    trial_std, n_trials = 0.5, 250
    expected = trial_std * (
        (1.0 - EULER_MASCHERONI) * normal_ppf(1.0 - 1.0 / n_trials)
        + EULER_MASCHERONI * normal_ppf(1.0 - 1.0 / (n_trials * math.e))
    )
    assert expected_max_sharpe(trial_std, n_trials) == pytest.approx(expected, rel=1e-12)


def test_expected_max_sharpe_is_zero_without_multiple_testing() -> None:
    """N=1 无多重检验，SR*=0，DSR 退化为普通 PSR（原式在 N=1 处发散，必须特判）。"""
    assert expected_max_sharpe(0.7, 1) == 0.0
    assert expected_max_sharpe(0.0, 1000) == 0.0


def test_expected_max_sharpe_grows_with_trial_count() -> None:
    values = [expected_max_sharpe(0.5, n) for n in (2, 10, 100, 1000, 10000)]
    assert values == sorted(values)


def test_expected_max_sharpe_scales_linearly_in_trial_std() -> None:
    assert expected_max_sharpe(1.0, 500) * 3.0 == pytest.approx(expected_max_sharpe(3.0, 500), rel=1e-12)


def test_expected_max_sharpe_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        expected_max_sharpe(0.5, 0)
    with pytest.raises(ValueError):
        expected_max_sharpe(-0.1, 100)


# ── 门槛反解 ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("benchmark", [0.0, 0.02, 0.08])
@pytest.mark.parametrize("skew", [-1.0, 0.0, 0.7])
@pytest.mark.parametrize("kurtosis", [3.0, 9.0])
@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
def test_minimum_sharpe_round_trips_through_psr(
    benchmark: float, skew: float, kurtosis: float, confidence: float
) -> None:
    """解析解的正确性：把反解出来的夏普喂回 PSR，必须精确得到目标置信度。"""
    required = minimum_sharpe_for_confidence(
        benchmark, SAMPLE_DAYS, skew, kurtosis, confidence
    )
    assert math.isfinite(required)
    assert required > benchmark
    achieved = probabilistic_sharpe_ratio(required, benchmark, SAMPLE_DAYS, skew, kurtosis)
    assert achieved == pytest.approx(confidence, abs=1e-12)


def test_minimum_sharpe_returns_infinity_when_unreachable() -> None:
    """极端厚尾 + 极短样本时检验统计量有上界，任何夏普都达不到 —— 返回 inf 而非假值。

    统计量上界为 √(4(T−1)/(γ₄−1))；T=5、γ₄=200 时上界 ≈ 0.57 < z₉₅ = 1.645。
    """
    assert minimum_sharpe_for_confidence(0.0, n_obs=5, kurtosis=200.0) == math.inf


def test_minimum_sharpe_grows_with_benchmark_and_confidence() -> None:
    assert (
        minimum_sharpe_for_confidence(0.00, SAMPLE_DAYS)
        < minimum_sharpe_for_confidence(0.05, SAMPLE_DAYS)
    )
    assert (
        minimum_sharpe_for_confidence(0.0, SAMPLE_DAYS, confidence=0.90)
        < minimum_sharpe_for_confidence(0.0, SAMPLE_DAYS, confidence=0.99)
    )


# ── DSR 主函数 ─────────────────────────────────────────────────────────

def test_dsr_equals_psr_when_only_one_trial() -> None:
    """N=1 时没有选择偏差，DSR 必须等于 PSR(0)。"""
    result = deflated_sharpe_ratio(
        observed_sharpe=1.2, trial_sharpe_std=0.5, n_trials=1,
        n_obs=SAMPLE_DAYS, annual_days=HK_ANNUAL_DAYS,
    )
    assert result.deflated_sharpe_ratio == pytest.approx(result.probabilistic_sharpe_ratio)
    assert result.expected_max_sharpe_annual == 0.0


def test_dsr_falls_as_trial_count_rises() -> None:
    """同一条曲线，试过的参数越多，DSR 越低 —— 这就是多重检验惩罚。"""
    values = [
        deflated_sharpe_ratio(
            observed_sharpe=1.8, trial_sharpe_std=0.5, n_trials=n,
            n_obs=SAMPLE_DAYS, annual_days=HK_ANNUAL_DAYS,
        ).deflated_sharpe_ratio
        for n in (1, 10, 100, 1000, 10000)
    ]
    assert values == sorted(values, reverse=True)


def test_dsr_annualised_and_per_period_paths_agree() -> None:
    """annualised=True/False 只是单位换算，结论必须一致。"""
    root = math.sqrt(HK_ANNUAL_DAYS)
    annual = deflated_sharpe_ratio(
        observed_sharpe=1.9, trial_sharpe_std=0.55, n_trials=250,
        n_obs=SAMPLE_DAYS, skew=-0.4, kurtosis=7.0,
        annual_days=HK_ANNUAL_DAYS, annualised=True,
    )
    daily = deflated_sharpe_ratio(
        observed_sharpe=1.9 / root, trial_sharpe_std=0.55 / root, n_trials=250,
        n_obs=SAMPLE_DAYS, skew=-0.4, kurtosis=7.0,
        annual_days=HK_ANNUAL_DAYS, annualised=False,
    )
    assert annual.deflated_sharpe_ratio == pytest.approx(daily.deflated_sharpe_ratio, rel=1e-12)
    assert annual.observed_sharpe_daily == pytest.approx(daily.observed_sharpe_daily, rel=1e-12)


def test_dsr_required_sharpe_is_the_decision_boundary() -> None:
    """把 required_sharpe 当观测值喂回去，DSR 应恰好等于置信度 —— 面板上的门槛可信。"""
    base = deflated_sharpe_ratio(
        observed_sharpe=1.0, trial_sharpe_std=0.5, n_trials=100,
        n_obs=SAMPLE_DAYS, skew=-0.3, kurtosis=6.0, annual_days=HK_ANNUAL_DAYS,
    )
    at_boundary = deflated_sharpe_ratio(
        observed_sharpe=base.required_sharpe_annual, trial_sharpe_std=0.5, n_trials=100,
        n_obs=SAMPLE_DAYS, skew=-0.3, kurtosis=6.0, annual_days=HK_ANNUAL_DAYS,
    )
    assert at_boundary.deflated_sharpe_ratio == pytest.approx(DEFAULT_CONFIDENCE, abs=1e-10)
    assert at_boundary.significant


def test_dsr_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(1.0, 0.5, 100, n_obs=1)
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(1.0, 0.5, 100, n_obs=SAMPLE_DAYS, confidence=1.0)
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(1.0, 0.5, 100, n_obs=SAMPLE_DAYS, annual_days=0)


def test_result_summary_is_renderable() -> None:
    result = deflated_sharpe_ratio(
        observed_sharpe=2.4, trial_sharpe_std=0.5, n_trials=100,
        n_obs=SAMPLE_DAYS, annual_days=HK_ANNUAL_DAYS,
    )
    text = result.summary()
    assert "DSR=" in text
    assert isinstance(result.as_dict()["deflated_sharpe_ratio"], float)


# ━━━ 方法有效性检验（本文件的核心）━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_random_trials_are_never_significant_under_dsr() -> None:
    """【核心】纯随机序列里挑最好的一组，DSR 必须判"不显著"。

    构造：100 组零均值日收益，各 600 根（= 本项目真实样本长度），互相独立，
    真实夏普全部为 0。按"挑夏普最高的那组"这一真实寻优行为选参数。

    三条断言：
      1. DSR 的假阳性率 ≤ 5%（名义水平）。这是"方法有效"的直接证据。
      2. 同一份数据下不做 deflation 的 PSR(0) 假阳性率 ≥ 80% —— 说明这不是
         "检验太钝所以谁都判不显著"，而是 deflation 精确地扣掉了选择偏差。
      3. DSR 的均值 ≈ 0.5。SR* 是最大夏普的【期望】，所以零假设下观测最大夏普
         落在 SR* 两侧的概率各半，DSR 应当以 0.5 为中心 —— 这比单看假阳性率
         更强，它说明整个分布位置正确，不是碰巧被压到 0 附近。
    """
    rng = np.random.default_rng(20260725)
    n_trials, reps = 100, 400

    dsr_values: list[float] = []
    psr_values: list[float] = []
    for _ in range(reps):
        returns, sharpes = null_trial_sharpes(rng, n_trials, SAMPLE_DAYS)
        result = select_best_and_deflate(returns, sharpes, n_trials)
        dsr_values.append(result.deflated_sharpe_ratio)
        psr_values.append(result.probabilistic_sharpe_ratio)

    dsr_false_positive = float(np.mean(np.asarray(dsr_values) >= DEFAULT_CONFIDENCE))
    psr_false_positive = float(np.mean(np.asarray(psr_values) >= DEFAULT_CONFIDENCE))

    assert dsr_false_positive <= 0.05, f"DSR 对随机数据的假阳性率 {dsr_false_positive:.3f} 超过名义水平"
    assert psr_false_positive >= 0.80, f"未 deflate 的 PSR 假阳性率仅 {psr_false_positive:.3f}，对照失效"
    assert float(np.mean(dsr_values)) == pytest.approx(0.5, abs=0.06)


@pytest.mark.parametrize("n_trials", [10, 1000])
def test_random_trials_stay_insignificant_across_trial_counts(n_trials: int) -> None:
    """上一条在别的 N 下同样成立，排除"恰好在 N=100 调准了"的可能。"""
    rng = np.random.default_rng(1000 + n_trials)
    reps = 200

    hits = 0
    for _ in range(reps):
        returns, sharpes = null_trial_sharpes(rng, n_trials, SAMPLE_DAYS)
        hits += select_best_and_deflate(returns, sharpes, n_trials).significant

    assert hits / reps <= 0.05


def test_dsr_is_conservative_when_trials_are_correlated() -> None:
    """试验相关时假阳性率仍不超过名义水平。

    真实寻优里相邻参数的收益曲线高度相关，Bailey 的 E[max] 公式假设试验 iid，
    用相关试验去套会高估 E[max] → SR* 偏高 → 检验偏保守。模块 docstring 的
    局限 1 写的就是这件事，此处兑现它：即便相关系数高到 0.9，假阳性率也没有
    突破 5%。
    """
    rng = np.random.default_rng(4242)
    n_trials, reps = 100, 300

    for corr in (0.5, 0.9):
        hits = 0
        for _ in range(reps):
            returns, sharpes = null_trial_sharpes(rng, n_trials, SAMPLE_DAYS, corr=corr)
            hits += select_best_and_deflate(returns, sharpes, n_trials).significant
        assert hits / reps <= NOMINAL_ALPHA, f"corr={corr} 时假阳性率 {hits / reps:.3f}"


def test_dsr_still_detects_genuine_skill() -> None:
    """功效测试：真有技能时 DSR 必须判显著。

    没有这一条，一个"永远返回不显著"的实现也能通过上面所有的假阳性测试。
    构造：全部 100 组试验都带真实漂移（真实年化夏普 ≈ 1.57）。
    """
    rng = np.random.default_rng(777)
    n_trials, reps = 100, 200
    drift = 0.0010                       # 日均 0.10%，日波动 1% → 年化夏普 ≈ 1.57

    hits = 0
    for _ in range(reps):
        returns, sharpes = null_trial_sharpes(rng, n_trials, SAMPLE_DAYS, drift=drift)
        hits += select_best_and_deflate(returns, sharpes, n_trials).significant

    assert hits / reps >= 0.90, f"真实技能下的检出率仅 {hits / reps:.3f}，检验过钝"


def test_single_random_curve_is_not_significant() -> None:
    """不涉及多重检验的那一层：单条随机曲线，sharpe_significance 应判不显著。"""
    rng = np.random.default_rng(31415)

    hits = 0
    reps = 500
    for _ in range(reps):
        returns = rng.standard_normal(SAMPLE_DAYS) * 0.01
        hits += sharpe_significance(returns, HK_ANNUAL_DAYS).significant_at_95

    assert hits / reps <= 0.08          # 名义 5%，留蒙特卡洛误差


def test_single_curve_with_real_drift_is_significant() -> None:
    rng = np.random.default_rng(27182)
    returns = rng.standard_normal(SAMPLE_DAYS) * 0.01 + 0.0015

    significance = sharpe_significance(returns, HK_ANNUAL_DAYS)

    assert significance.sharpe_annual > 1.5
    assert significance.significant_at_95
    assert "显著" in significance.summary()


def test_sharpe_significance_accepts_external_sharpe() -> None:
    """vnpy 的 sharpe_ratio 会扣 risk_free，自算不扣 —— 允许传入以保持口径一致。"""
    rng = np.random.default_rng(5)
    returns = rng.standard_normal(300) * 0.01

    forced = sharpe_significance(returns, HK_ANNUAL_DAYS, sharpe_annual=2.5)

    assert forced.sharpe_annual == 2.5
    assert forced.sharpe_daily == pytest.approx(2.5 / math.sqrt(HK_ANNUAL_DAYS))
    assert forced.t_stat == pytest.approx(forced.sharpe_annual / forced.std_error_annual, rel=1e-12)


def test_sharpe_significance_handles_flat_curve() -> None:
    """完全没交易（净值不动）时不应崩，夏普为 0、判不显著。"""
    significance = sharpe_significance([0.0] * 100, HK_ANNUAL_DAYS)
    assert significance.sharpe_annual == 0.0
    assert significance.psr_zero == pytest.approx(0.5)
    assert not significance.significant_at_95


# ── 接 vnpy 寻优流程 ───────────────────────────────────────────────────

def test_trial_sharpes_reads_every_trial_not_just_the_top() -> None:
    """run_bf_optimization 返回的是【全部】试验（只排序不截断），此处钉住这一假设。"""
    results = [make_optimization_result({"window": w}, sharpe=w / 10.0) for w in range(1, 51)]

    trials = trial_sharpes_from_optimization(results)

    assert trials.n_total == 50
    assert trials.n_usable == 50
    assert trials.n_missing == 0
    assert trials.std() == pytest.approx(float(np.std(np.arange(1, 51) / 10.0, ddof=1)))


def test_trial_sharpes_counts_blown_up_backtests_separately() -> None:
    """回测爆仓时 calculate_statistics 返回 {}，该组进 n_total 但不进方差估计。"""
    results: list[Any] = [make_optimization_result({"window": w}, sharpe=1.0 + w * 0.1) for w in range(5)]
    results.append(({"window": 99}, 0.0, {}))                  # 爆仓：空 statistics
    results.append(({"window": 98}, 0.0, {"sharpe_ratio": float("inf")}))
    results.append(({"window": 97}, 0.0, {"sharpe_ratio": "nan-ish"}))

    trials = trial_sharpes_from_optimization(results)

    assert trials.n_total == 8
    assert trials.n_usable == 5
    assert trials.n_missing == 3


def test_trial_sharpes_accepts_plain_statistics_dicts() -> None:
    trials = trial_sharpes_from_optimization([{"sharpe_ratio": 1.0}, {"sharpe_ratio": 2.0}])
    assert trials.n_usable == 2
    assert trials.as_dict()["max"] == 2.0


def test_trial_sharpes_std_is_zero_below_two_samples() -> None:
    """样本不足以估方差时返回 0.0 —— 等价于不做 deflation，而不是抛异常或给假值。"""
    assert trial_sharpes_from_optimization([{"sharpe_ratio": 1.0}]).std() == 0.0
    assert trial_sharpes_from_optimization([]).std() == 0.0


def test_deflate_optimization_end_to_end() -> None:
    """从寻优结果直接出 DSR：默认选 results[0]（已按目标值降序，即真正上实盘的那组）。"""
    rng = np.random.default_rng(9)
    sharpes = np.sort(rng.normal(0.4, 0.5, size=200))[::-1]
    results = [
        make_optimization_result({"window": i}, float(s), skew=-0.4, kurtosis=7.0)
        for i, s in enumerate(sharpes)
    ]

    result = deflate_optimization(results, annual_days=HK_ANNUAL_DAYS)

    assert result.n_trials == 200
    assert result.n_obs == SAMPLE_DAYS
    assert result.skew == -0.4
    assert result.kurtosis == 7.0
    assert result.observed_sharpe_annual == pytest.approx(float(sharpes[0]))
    assert result.trial_sharpe_std_annual == pytest.approx(float(np.std(sharpes, ddof=1)))
    # 挑 200 组挑出来的最好一组，纯选择偏差就值 SR* 这么多，DSR 判不显著
    assert result.expected_max_sharpe_annual > 1.0
    assert not result.significant


def test_deflate_optimization_falls_back_to_normal_moments() -> None:
    """statistics 里没导出收益矩时退回正态 (0, 3)，不静默用错值。"""
    results = [({"w": i}, 0.5, {"total_days": SAMPLE_DAYS, "sharpe_ratio": 0.5 + i * 0.01}) for i in range(30)]

    result = deflate_optimization(results, annual_days=HK_ANNUAL_DAYS)

    assert result.skew == 0.0
    assert result.kurtosis == NORMAL_KURTOSIS


def test_deflate_optimization_reads_sharpe_inference_moments() -> None:
    """收益矩直接取 sharpe_inference.py 已写进面板的键，无需再改 calculate_statistics。"""
    results = [
        ({"w": i}, 1.0, {
            "total_days": SAMPLE_DAYS,
            "sharpe_ratio": 1.0 + i * 0.01,
            "sharpe_skew": -0.62,
            "sharpe_kurtosis": 8.4,
        })
        for i in range(40)
    ]

    result = deflate_optimization(results, annual_days=HK_ANNUAL_DAYS)

    assert result.skew == -0.62
    assert result.kurtosis == 8.4


def test_deflate_optimization_rejects_not_computed_moment_sentinel() -> None:
    """`sharpe_inference.statistics_fields()` 未计算时把两个矩填 0.0。

    γ₄ = 0 不是任何分布的峰度（恒有 γ₄ ≥ γ₃² + 1 ≥ 1），当真值用会低估 PSR 分母、
    高估显著性。必须识别为哨兵并退回正态，而不是照单全收。
    """
    results = [
        ({"w": i}, 1.0, {
            "total_days": SAMPLE_DAYS,
            "sharpe_ratio": 1.0 + i * 0.01,
            "sharpe_skew": 0.0,
            "sharpe_kurtosis": 0.0,          # 哨兵：未计算
        })
        for i in range(40)
    ]

    result = deflate_optimization(results, annual_days=HK_ANNUAL_DAYS)

    assert result.kurtosis == NORMAL_KURTOSIS
    assert result.skew == 0.0


def test_deflate_optimization_rejects_inconsistent_moment_pairs() -> None:
    """(γ₃, γ₄) 不满足 γ₄ ≥ γ₃² + 1 时整组退回正态，而不是让下游抛 ValueError。"""
    results = [
        ({"w": i}, 1.0, {
            "total_days": SAMPLE_DAYS,
            "sharpe_ratio": 1.0 + i * 0.01,
            "sharpe_skew": 3.0,              # 需要 γ₄ ≥ 10，但只给了 2.0
            "sharpe_kurtosis": 2.0,
        })
        for i in range(40)
    ]

    result = deflate_optimization(results, annual_days=HK_ANNUAL_DAYS)

    assert (result.skew, result.kurtosis) == (0.0, NORMAL_KURTOSIS)


def test_deflate_optimization_prefers_explicit_moments_over_statistics() -> None:
    results = [make_optimization_result({"w": i}, 1.0 + i * 0.01, skew=-0.5, kurtosis=9.0) for i in range(40)]

    result = deflate_optimization(results, HK_ANNUAL_DAYS, skew=0.2, kurtosis=4.0)

    assert (result.skew, result.kurtosis) == (0.2, 4.0)


def test_deflate_optimization_honours_overrides() -> None:
    """GA 场景：n_trials 与 trial_sharpe_std 都要能被独立采样估出来的值覆盖。"""
    results = [make_optimization_result({"w": i}, 2.6 - i * 0.05) for i in range(20)]

    result = deflate_optimization(
        results, annual_days=HK_ANNUAL_DAYS, n_trials=5000, trial_sharpe_std=0.9,
    )

    assert result.n_trials == 5000
    assert result.trial_sharpe_std_annual == 0.9
    assert result.expected_max_sharpe_annual == pytest.approx(expected_max_sharpe(0.9, 5000))


def test_deflate_optimization_can_select_another_trial() -> None:
    results = [make_optimization_result({"w": i}, 2.0 - i * 0.1) for i in range(10)]

    assert deflate_optimization(results, HK_ANNUAL_DAYS, selected_index=3).observed_sharpe_annual == pytest.approx(1.7)
    assert deflate_optimization(results, HK_ANNUAL_DAYS, selected_index=-1).observed_sharpe_annual == pytest.approx(1.1)


def test_deflate_optimization_error_paths() -> None:
    results = [make_optimization_result({"w": i}, 1.0 + i * 0.1) for i in range(4)]

    with pytest.raises(ValueError, match="寻优结果为空"):
        deflate_optimization([], HK_ANNUAL_DAYS)
    with pytest.raises(IndexError):
        deflate_optimization(results, HK_ANNUAL_DAYS, selected_index=99)
    with pytest.raises(KeyError, match="爆仓"):
        deflate_optimization([({"w": 1}, 0.0, {})], HK_ANNUAL_DAYS)
    with pytest.raises(TypeError):
        deflate_optimization(["not-a-result"], HK_ANNUAL_DAYS)


def test_deflate_optimization_explains_missing_total_days() -> None:
    """缺 total_days 时给出可操作的报错，而不是裸 KeyError('total_days')。

    真实 `calculate_statistics()` 一定带这个键，会走到这条路径的只有手工构造的
    results；报错必须直接告诉调用方"显式传 n_obs"。
    """
    truncated = [({"w": 1}, 1.5, {"sharpe_ratio": 1.5})]

    with pytest.raises(KeyError, match="n_obs"):
        deflate_optimization(truncated, HK_ANNUAL_DAYS)

    # 显式传 n_obs 就能跑通，不必伪造 statistics
    result = deflate_optimization(truncated, HK_ANNUAL_DAYS, n_obs=SAMPLE_DAYS)
    assert result.n_obs == SAMPLE_DAYS
    assert result.observed_sharpe_annual == pytest.approx(1.5)


# ── 门槛表 ─────────────────────────────────────────────────────────────

def test_required_sharpe_table_pins_the_headline_numbers() -> None:
    """钉住报告里给出的门槛数字（T=600、annual_days=247、正态、95%）。"""
    rows = required_sharpe_table(
        n_trials_list=[1, 100, 1000],
        trial_std_list=[0.5],
        n_obs=SAMPLE_DAYS,
        annual_days=HK_ANNUAL_DAYS,
    )
    thresholds = {int(row["n_trials"]): float(row["required_sharpe"]) for row in rows}

    assert thresholds[1] == pytest.approx(1.06, abs=0.01)
    assert thresholds[100] == pytest.approx(2.33, abs=0.01)
    assert thresholds[1000] == pytest.approx(2.69, abs=0.01)


def test_required_sharpe_table_is_monotone() -> None:
    rows = required_sharpe_table(
        n_trials_list=[1, 10, 100, 1000, 10000],
        trial_std_list=[0.25, 0.5, 1.0],
        n_obs=SAMPLE_DAYS,
        annual_days=HK_ANNUAL_DAYS,
    )
    by_std: dict[float, list[float]] = {}
    for row in rows:
        by_std.setdefault(float(row["trial_sharpe_std"]), []).append(float(row["required_sharpe"]))

    for thresholds in by_std.values():
        assert thresholds == sorted(thresholds)          # N 越大门槛越高

    stds = sorted(by_std)
    for tighter, looser in zip(stds, stds[1:], strict=False):
        # 同一个 N 下，试验离散度越大门槛越高（N=1 除外，无选择偏差）
        assert by_std[tighter][-1] < by_std[looser][-1]


def test_table_formatting_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    rows = required_sharpe_table([1, 100], [0.5], n_obs=SAMPLE_DAYS, annual_days=HK_ANNUAL_DAYS)
    assert "门槛年化SR" in format_required_sharpe_table(rows)

    assert main(["--n-obs", "600", "--n-trials", "1", "100", "--trial-std", "0.5"]) == 0
    assert "DSR 门槛表" in capsys.readouterr().out


def test_table_marks_unreachable_thresholds() -> None:
    rows = required_sharpe_table([100], [0.5], n_obs=5, kurtosis=200.0)
    assert "不可达" in format_required_sharpe_table(rows)
