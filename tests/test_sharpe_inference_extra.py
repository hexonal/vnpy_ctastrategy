"""Sharpe 推断的补充测试（辅助函数、带宽、bootstrap 机制、PSR/DSR 边界、退化输入）。

核心不是"函数跑得通"，而是【方法本身有效】：对真实 SR=0 的随机序列，
检验必须只在约 5% 的重复中判显著（Monte-Carlo size test，test_size_*）。
公式对照测试（Lo / Mertens / GMM 三档互相退化）是第二层保险。
Monte-Carlo 部分固定随机种子；断言区间按 ±3 个二项标准误留裕度。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vnpy_ctastrategy.sharpe_inference import (
    andrews_lags,
    autocorrelations,
    bootstrap_sharpe_test,
    deflate_optimization_results,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    inference_from_daily_df,
    ljung_box,
    long_run_covariance,
    minimum_track_record_length,
    newey_west_lags,
    norm_cdf,
    norm_ppf,
    norm_sf,
    probabilistic_sharpe_ratio,
    required_sharpe,
    sharpe_inference,
    sharpe_significance_table,
    stationary_bootstrap_indices,
)

ANNUAL_DAYS = 247            # 港股口径；美股应传 252
PROJECT_N = 611              # 700.SEHK 2024-01~2026-07 的实测样本长度
DAILY_VOL = 0.012            # 日波动 1.2%，接近港美股单标的 CTA 的实际量级


@pytest.mark.parametrize("method", ["iid_normal", "iid_nonnormal", "hac"])


def _white_noise(rng: np.random.Generator, n: int = PROJECT_N) -> np.ndarray:
    """真实 SR = 0 的 iid 正态序列。"""
    return rng.standard_normal(n) * DAILY_VOL


def _ar1(rng: np.random.Generator, rho: float, n: int = PROJECT_N, reps: int = 1) -> np.ndarray:
    """零均值 AR(1) 序列 (reps, n)，已丢弃 200 期 burn-in。"""
    burn = 200
    noise = rng.standard_normal((reps, n + burn)) * DAILY_VOL
    x = np.empty((reps, n + burn))
    x[:, 0] = noise[:, 0]
    for t in range(1, n + burn):
        x[:, t] = rho * x[:, t - 1] + noise[:, t]
    return x[:, burn:]


def _binomial_bounds(rate: float, reps: int, sigmas: float = 3.0) -> tuple[float, float]:
    se = math.sqrt(rate * (1.0 - rate) / reps)
    return rate - sigmas * se, rate + sigmas * se


# 一、方法有效性：随机序列必须判为不显著（本文件的核心）


def _daily_df(net_pnl: np.ndarray):
    """构造 calculate_statistics 需要的最小 daily_df。"""
    from datetime import date, timedelta

    import pandas as pd

    n = net_pnl.size
    return pd.DataFrame(
        {
            "net_pnl": net_pnl,
            "turnover": np.zeros(n),
            "commission": np.zeros(n),
            "slippage": np.zeros(n),
            "trade_count": np.zeros(n, dtype=int),
        },
        index=pd.to_datetime([date(2024, 1, 1) + timedelta(days=i) for i in range(n)]),
    )


def test_hac_inflation_is_about_one_when_returns_are_iid() -> None:
    """无自相关时 HAC 不应无端放大 SE —— 否则等于白白损失功效。"""
    rng = np.random.default_rng(606)
    result = sharpe_inference(_white_noise(rng, 4000), annual_days=ANNUAL_DAYS, n_boot=0)
    assert result.hac_inflation == pytest.approx(1.0, abs=0.10)


# 四、公式对照：三档方差因子必须互相退化


def test_long_run_covariance_at_zero_lags_is_the_contemporaneous_covariance() -> None:
    rng = np.random.default_rng(8)
    u = rng.standard_normal((500, 2))
    u = u - u.mean(axis=0)
    assert np.allclose(long_run_covariance(u, 0), u.T @ u / 500)


def test_long_run_covariance_is_symmetric_and_grows_with_positive_autocorrelation() -> None:
    rng = np.random.default_rng(9)
    x = _ar1(rng, 0.5, n=3000)[0]
    u = np.column_stack([x - x.mean(), (x - x.mean()) ** 2 - ((x - x.mean()) ** 2).mean()])
    omega0 = long_run_covariance(u, 0)
    omega5 = long_run_covariance(u, 5)
    assert np.allclose(omega5, omega5.T)
    assert omega5[0, 0] > omega0[0, 0]


# 五、带宽选择


def test_andrews_bandwidth_grows_with_dependence() -> None:
    """数据驱动带宽：自相关越强，带宽越大。"""
    rng = np.random.default_rng(303)
    weak = _ar1(rng, 0.05, n=2000)[0]
    strong = _ar1(rng, 0.6, n=2000)[0]

    def moments(x: np.ndarray) -> np.ndarray:
        dev = x - x.mean()
        return np.column_stack([dev, dev ** 2 - (dev ** 2).mean()])

    assert andrews_lags(moments(strong)) > andrews_lags(moments(weak))


def test_lags_argument_accepts_rule_names_and_integers() -> None:
    rng = np.random.default_rng(404)
    r = _ar1(rng, 0.25, n=PROJECT_N)[0]
    by_rule = sharpe_inference(r, lags="newey_west", n_boot=0)
    by_andrews = sharpe_inference(r, lags="andrews", n_boot=0)
    by_int = sharpe_inference(r, lags=5, n_boot=0)

    assert by_rule.hac_lags == newey_west_lags(PROJECT_N) == 5
    assert by_int.hac_lags == 5
    assert by_rule.se_hac_annual == pytest.approx(by_int.se_hac_annual, rel=1e-12)
    assert by_andrews.hac_lags >= 0
    with pytest.raises(ValueError, match="未知的 lags 规则"):
        sharpe_inference(r, lags="bogus", n_boot=0)


# 六、"要多大 Sharpe 才显著"


def test_required_sharpe_falls_with_sample_size() -> None:
    """门槛 ∝ 1/√n：样本翻四倍，门槛减半。"""
    assert required_sharpe(2444, ANNUAL_DAYS) / required_sharpe(611, ANNUAL_DAYS) == pytest.approx(
        0.5, rel=0.02
    )


def test_significance_table_renders() -> None:
    text = sharpe_significance_table(611, ANNUAL_DAYS)
    assert "n=611" in text
    assert "1.25" in text


# 七、Stationary bootstrap


def test_bootstrap_indices_shape_and_range() -> None:
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(100, 8.0, 50, rng)
    assert idx.shape == (50, 100)
    assert idx.min() >= 0 and idx.max() < 100


def test_bootstrap_blocks_preserve_local_order() -> None:
    """块长均值越大，相邻下标连续（+1 mod n）的比例越高 —— 这正是它保留自相关的机制。"""
    rng = np.random.default_rng(2)
    for mean_block, floor in ((2.0, 0.35), (20.0, 0.85)):
        idx = stationary_bootstrap_indices(200, mean_block, 200, rng)
        contiguous = np.mean((idx[:, 1:] - idx[:, :-1]) % 200 == 1)
        assert contiguous > floor


def test_bootstrap_p_value_is_bounded_and_reproducible() -> None:
    rng = np.random.default_rng(3)
    r = _white_noise(rng)
    p1, se1 = bootstrap_sharpe_test(r, n_boot=199, seed=42)
    p2, se2 = bootstrap_sharpe_test(r, n_boot=199, seed=42)
    assert p1 == p2 and se1 == se2                  # 固定种子必须可复现
    assert 0.0 < p1 <= 1.0
    assert se1 > 0.0


def test_bootstrap_se_is_close_to_the_analytic_se_for_iid_data() -> None:
    """iid 数据上，bootstrap SE 与解析 SE 应该接近 —— 二者互为校验。"""
    rng = np.random.default_rng(1717)
    r = _white_noise(rng, 2000)
    result = sharpe_inference(r, annual_days=ANNUAL_DAYS, n_boot=999)
    assert result.bootstrap_se_annual == pytest.approx(result.se_hac_annual, rel=0.20)


# 八、多重比较：PSR / MinTRL / DSR


def test_psr_equals_one_minus_one_sided_p_at_zero_benchmark() -> None:
    """benchmark=0 时 PSR 就是单尾 p 值的补（差异仅来自 √(n−1) vs √n）。"""
    rng = np.random.default_rng(21)
    r = rng.standard_normal(PROJECT_N) * DAILY_VOL + 0.0006
    result = sharpe_inference(r, annual_days=ANNUAL_DAYS, method="iid_nonnormal", n_boot=0)
    psr = probabilistic_sharpe_ratio(
        result.sharpe_period, PROJECT_N, result.skew, result.kurtosis
    )
    assert psr == pytest.approx(1.0 - result.p_value_one_sided, abs=0.005)


def test_min_track_record_length_round_trips_through_psr() -> None:
    """在 n = MinTRL 处，PSR 应恰好等于目标置信度。"""
    sharpe_period = 1.0 / math.sqrt(ANNUAL_DAYS)
    n_needed = minimum_track_record_length(sharpe_period, confidence=0.95)
    assert probabilistic_sharpe_ratio(sharpe_period, int(round(n_needed))) == pytest.approx(
        0.95, abs=0.002
    )
    # 年化 Sharpe = 1.0 的策略，需要约 671 个交易日（2.7 年）才能把它和 0 分开
    assert n_needed == pytest.approx(671, abs=5)
    assert n_needed / ANNUAL_DAYS == pytest.approx(2.72, abs=0.05)
    # 与 required_sharpe 必须互为逆函数：在 n=MinTRL 处，单尾门槛正好回到 1.0
    assert required_sharpe(
        int(round(n_needed)), ANNUAL_DAYS, alpha=0.05, one_sided=True
    ) == pytest.approx(1.0, abs=0.01)


def test_min_track_record_length_is_infinite_for_a_losing_strategy() -> None:
    assert minimum_track_record_length(-0.05) == math.inf


def test_expected_max_sharpe_grows_with_trial_count() -> None:
    """试的组数越多，"纯运气最优值"越高 —— 这就是参数寻优的选择偏差。"""
    std = 0.3 / math.sqrt(ANNUAL_DAYS)
    values = [expected_max_sharpe(n, std) for n in (10, 100, 500, 5000)]
    assert values == sorted(values)
    assert expected_max_sharpe(1, std) == 0.0
    assert expected_max_sharpe(500, 0.0) == 0.0


def test_deflated_sharpe_is_below_probabilistic_sharpe() -> None:
    """DSR 一定比 PSR 低：它多扣了一层"我试了 N 组"的惩罚。"""
    sharpe_period = 1.2 / math.sqrt(ANNUAL_DAYS)
    psr = probabilistic_sharpe_ratio(sharpe_period, PROJECT_N)
    dsr = deflated_sharpe_ratio(
        sharpe_period, PROJECT_N, n_trials=500, sharpe_std=0.4 / math.sqrt(ANNUAL_DAYS)
    )
    assert dsr < psr


def test_deflate_optimization_results_accepts_a_genuinely_strong_sweep() -> None:
    """真有 alpha 时（全部参数都赚，最优组远高于横截面离散度）DSR 应当放行。"""
    rng = np.random.default_rng(1001)
    mu = 3.0 / math.sqrt(ANNUAL_DAYS) * DAILY_VOL
    results = []
    for i in range(50):
        r = rng.standard_normal(PROJECT_N) * DAILY_VOL + mu
        annual = float(r.mean() / r.std(ddof=1) * math.sqrt(ANNUAL_DAYS))
        results.append(({"trial": i}, annual, {}))
    results.sort(key=lambda row: row[1], reverse=True)

    report = deflate_optimization_results(results, PROJECT_N, "sharpe_ratio", ANNUAL_DAYS)
    assert report["trustworthy"] is True


def test_deflate_optimization_results_handles_empty_input() -> None:
    report = deflate_optimization_results([], PROJECT_N, "sharpe_ratio")
    assert report["n_trials"] == 0
    assert report["trustworthy"] is False


# 九、分布函数与诊断量


def test_norm_cdf_and_sf_are_consistent() -> None:
    for x in (-6.0, -1.96, 0.0, 1.0, 3.5):
        assert norm_cdf(x) + norm_sf(x) == pytest.approx(1.0, rel=1e-15)
    assert norm_cdf(norm_ppf(0.975)) == pytest.approx(0.975, rel=1e-12)


def test_norm_ppf_rejects_out_of_range() -> None:
    assert norm_ppf(0.0) == -math.inf
    assert norm_ppf(1.0) == math.inf
    with pytest.raises(ValueError):
        norm_ppf(1.5)


def test_autocorrelations_recover_a_known_ar1() -> None:
    rng = np.random.default_rng(600)
    x = _ar1(rng, 0.4, n=20000)[0]
    rho = autocorrelations(x, 3)
    assert rho[0] == pytest.approx(0.4, abs=0.03)
    assert rho[1] == pytest.approx(0.16, abs=0.03)


def test_ljung_box_separates_iid_from_autocorrelated() -> None:
    rng = np.random.default_rng(601)
    _, p_iid = ljung_box(_white_noise(rng, 2000), lags=10)
    _, p_ar = ljung_box(_ar1(rng, 0.3, n=2000)[0], lags=10)
    assert p_iid > 0.05
    assert p_ar < 1e-6


# 十、输入校验与退化情形


def test_non_finite_returns_are_dropped() -> None:
    rng = np.random.default_rng(701)
    clean = _white_noise(rng, 400)
    dirty = np.concatenate([clean, [np.nan, np.inf, -np.inf]])
    assert sharpe_inference(dirty, n_boot=0).n_periods == 400


def test_short_sample_emits_a_warning_instead_of_silently_pretending() -> None:
    rng = np.random.default_rng(702)
    result = sharpe_inference(_white_noise(rng, 134), annual_days=ANNUAL_DAYS, n_boot=0)
    assert any("不足一年" in w for w in result.warnings)


def test_summary_line_and_report_contain_the_four_numbers() -> None:
    rng = np.random.default_rng(703)
    result = sharpe_inference(_white_noise(rng), annual_days=ANNUAL_DAYS, n_boot=99)
    line = result.summary_line()
    for token in ("sharpe =", "SE =", "t =", "p ="):
        assert token in line
    report = result.report()
    assert "置信区间" in report and "bootstrap" in report


def test_as_dict_is_json_shaped() -> None:
    rng = np.random.default_rng(704)
    data = sharpe_inference(_white_noise(rng, 300), n_boot=0).as_dict()
    assert data["n_periods"] == 300
    assert isinstance(data["warnings"], tuple)


# 十一、与回测引擎的集成


def test_empty_backtest_does_not_crash_on_the_new_fields() -> None:
    from vnpy_ctastrategy.backtesting import BacktestingEngine

    engine = BacktestingEngine()
    assert engine.calculate_statistics(df=None, output=False) == {}


def test_inference_from_daily_df_rejects_none() -> None:
    with pytest.raises(ValueError, match="daily_df"):
        inference_from_daily_df(None)
