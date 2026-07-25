"""DSR 核心统计主张的精简验证集（完整 112 例回归见 test_deflated_sharpe.py）。

重点是【方法本身有效吗】，不是逐行覆盖：
  - 纯随机序列挑最优 → 必须判"不显著"（否则这个检验一文不值）
  - 同一份数据不做 deflation 的 PSR → 假阳性率极高（证明 deflation 真的在起作用）
  - 真有技能 → 判显著（否则"永远说不显著"也能骗过上一条）
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vnpy_ctastrategy.deflated_sharpe import (
    DEFAULT_CONFIDENCE,
    EULER_MASCHERONI,
    deflate_optimization,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    minimum_sharpe_for_confidence,
    normal_ppf,
    probabilistic_sharpe_ratio,
    required_sharpe_table,
    return_moments,
    sharpe_significance,
    sharpe_standard_error,
    trial_sharpes_from_optimization,
)

HK_ANNUAL_DAYS = 247
SAMPLE_DAYS = 600
ROOT = math.sqrt(HK_ANNUAL_DAYS)


def best_of_n(rng: np.random.Generator, n_trials: int, n_obs: int, drift: float = 0.0, corr: float = 0.0):
    """模拟一次寻优：n_trials 组日收益，按"挑夏普最高的那组"选参数，返回 DSR 结果。"""
    common = rng.standard_normal(n_obs)
    idio = rng.standard_normal((n_trials, n_obs))
    returns = (idio * math.sqrt(1.0 - corr) + common * math.sqrt(corr)) * 0.01 + drift
    sharpes = returns.mean(axis=1) / returns.std(axis=1, ddof=1) * ROOT
    best = int(np.argmax(sharpes))
    m = return_moments(returns[best])
    return deflated_sharpe_ratio(
        observed_sharpe=float(sharpes[best]),
        trial_sharpe_std=float(np.std(sharpes, ddof=1)),
        n_trials=n_trials,
        n_obs=n_obs,
        skew=m.skew,
        kurtosis=m.kurtosis,
        annual_days=HK_ANNUAL_DAYS,
    )


# ── 核心有效性检验 ────────────────────────────────────────────────────

def test_random_trials_are_not_significant_under_dsr() -> None:
    """【核心】600 日 × 100 组纯随机序列挑最优：DSR 假阳性率 ≤5%，PSR 却 ≥80%。"""
    rng = np.random.default_rng(20260725)
    dsr, psr = [], []
    for _ in range(400):
        r = best_of_n(rng, 100, SAMPLE_DAYS)
        dsr.append(r.deflated_sharpe_ratio)
        psr.append(r.probabilistic_sharpe_ratio)

    assert float(np.mean(np.asarray(dsr) >= DEFAULT_CONFIDENCE)) <= 0.05
    assert float(np.mean(np.asarray(psr) >= DEFAULT_CONFIDENCE)) >= 0.80
    # SR* 是最大夏普的期望，零假设下 DSR 应以 0.5 为中心（分布位置正确，不是被压到 0）
    assert float(np.mean(dsr)) == pytest.approx(0.5, abs=0.06)


@pytest.mark.parametrize("n_trials", [10, 1000])
def test_random_stays_insignificant_at_other_trial_counts(n_trials: int) -> None:
    rng = np.random.default_rng(1000 + n_trials)
    hits = sum(best_of_n(rng, n_trials, SAMPLE_DAYS).significant for _ in range(200))
    assert hits / 200 <= 0.05


def test_correlated_trials_stay_conservative() -> None:
    """真实寻优里相邻参数高度相关；相关会让 SR* 偏高 → 检验偏保守，方向安全。"""
    rng = np.random.default_rng(4242)
    for corr in (0.5, 0.9):
        hits = sum(best_of_n(rng, 100, SAMPLE_DAYS, corr=corr).significant for _ in range(300))
        assert hits / 300 <= 0.05


def test_dsr_detects_genuine_skill() -> None:
    """功效：真实年化夏普≈1.57 且全部试验都有技能时必须判显著。"""
    rng = np.random.default_rng(777)
    hits = sum(best_of_n(rng, 100, SAMPLE_DAYS, drift=0.0010).significant for _ in range(200))
    assert hits / 200 >= 0.90


def test_single_random_curve_is_not_significant() -> None:
    rng = np.random.default_rng(31415)
    hits = sum(
        sharpe_significance(rng.standard_normal(SAMPLE_DAYS) * 0.01, HK_ANNUAL_DAYS).significant_at_95
        for _ in range(500)
    )
    assert hits / 500 <= 0.08


# ── 公式正确性 ────────────────────────────────────────────────────────

def test_expected_max_sharpe_matches_monte_carlo() -> None:
    rng = np.random.default_rng(7)
    for n in (5, 20, 100, 1000):
        draws = rng.standard_normal((4000, n)) * 0.6
        assert expected_max_sharpe(0.6, n) == pytest.approx(float(draws.max(axis=1).mean()), abs=0.05)


def test_expected_max_sharpe_is_bailey_equation_5() -> None:
    n, v = 250, 0.5
    expected = v * (
        (1.0 - EULER_MASCHERONI) * normal_ppf(1.0 - 1.0 / n)
        + EULER_MASCHERONI * normal_ppf(1.0 - 1.0 / (n * math.e))
    )
    assert expected_max_sharpe(v, n) == pytest.approx(expected, rel=1e-12)
    assert expected_max_sharpe(v, 1) == 0.0          # N=1 无多重检验


def test_psr_reduces_to_one_sided_normal_test() -> None:
    sr, t = 0.04, 601
    from vnpy_ctastrategy.deflated_sharpe import normal_cdf
    expected = normal_cdf(sr * math.sqrt(600) / math.sqrt(1 + 0.5 * sr**2))
    assert probabilistic_sharpe_ratio(sr, 0.0, t) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("skew", [-1.0, 0.0, 0.7])
@pytest.mark.parametrize("kurtosis", [3.0, 9.0])
def test_threshold_round_trips_through_psr(skew: float, kurtosis: float) -> None:
    """反解出的门槛喂回 PSR，必须精确得到 95%。"""
    x = minimum_sharpe_for_confidence(0.05, SAMPLE_DAYS, skew, kurtosis)
    assert probabilistic_sharpe_ratio(x, 0.05, SAMPLE_DAYS, skew, kurtosis) == pytest.approx(0.95, abs=1e-12)


def test_project_anchor_134_day_curve() -> None:
    """项目实测锚点：134 日、年化夏普 −1.68，标准误 1.37 —— 连符号都没到 2σ。"""
    se = sharpe_standard_error(-1.68 / ROOT, 134) * ROOT
    assert se == pytest.approx(1.37, abs=0.01)
    assert abs(-1.68) < 2.0 * se


def test_dsr_falls_as_trial_count_rises() -> None:
    values = [
        deflated_sharpe_ratio(1.8, 0.5, n, SAMPLE_DAYS, annual_days=HK_ANNUAL_DAYS).deflated_sharpe_ratio
        for n in (1, 10, 100, 1000, 10000)
    ]
    assert values == sorted(values, reverse=True)


# ── 接 vnpy 寻优流程 ──────────────────────────────────────────────────

def test_reads_every_trial_from_vnpy_optimization_results() -> None:
    """run_bf_optimization 返回全部试验的 (setting, target, statistics) 三元组，不截断。"""
    results = [
        ({"w": i}, i / 10.0, {"total_days": SAMPLE_DAYS, "sharpe_ratio": i / 10.0,
                              "sharpe_skew": -0.4, "sharpe_kurtosis": 7.0})
        for i in range(50, 0, -1)
    ]
    results.append(({"w": 99}, 0.0, {}))                       # 爆仓：statistics 为空

    trials = trial_sharpes_from_optimization(results)
    assert (trials.n_total, trials.n_usable, trials.n_missing) == (51, 50, 1)

    result = deflate_optimization(results, annual_days=HK_ANNUAL_DAYS)
    assert result.n_trials == 51
    assert result.n_obs == SAMPLE_DAYS
    assert (result.skew, result.kurtosis) == (-0.4, 7.0)       # 读 sharpe_inference 写的矩
    assert result.observed_sharpe_annual == pytest.approx(5.0)


def test_not_computed_moment_sentinel_falls_back_to_normal() -> None:
    """sharpe_inference 未计算时把矩填 0.0；γ₄=0 不是任何分布的峰度，必须识别为哨兵。"""
    results = [({"w": i}, 1.0, {"total_days": SAMPLE_DAYS, "sharpe_ratio": 1.0 + i * 0.01,
                                "sharpe_skew": 0.0, "sharpe_kurtosis": 0.0}) for i in range(40)]
    result = deflate_optimization(results, annual_days=HK_ANNUAL_DAYS)
    assert (result.skew, result.kurtosis) == (0.0, 3.0)


def test_threshold_table_pins_headline_numbers() -> None:
    """报告里给出的门槛数字（T=600、annual_days=247、正态、95%）。"""
    rows = required_sharpe_table([1, 100, 1000], [0.5], n_obs=SAMPLE_DAYS, annual_days=HK_ANNUAL_DAYS)
    got = {int(r["n_trials"]): float(r["required_sharpe"]) for r in rows}
    assert got[1] == pytest.approx(1.06, abs=0.01)
    assert got[100] == pytest.approx(2.33, abs=0.01)
    assert got[1000] == pytest.approx(2.69, abs=0.01)
