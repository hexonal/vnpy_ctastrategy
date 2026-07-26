"""Sharpe 标准误与显著性检验的测试。

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
    bootstrap_sharpe_test,
    deflate_optimization_results,
    iid_nonnormal_variance_factor,
    iid_normal_variance_factor,
    inference_from_daily_df,
    newey_west_lags,
    norm_ppf,
    norm_sf,
    required_sharpe,
    sharpe_inference,
    sharpe_variance_factor,
    statistics_fields,
)

ANNUAL_DAYS = 247            # 港股口径；美股应传 252
PROJECT_N = 611              # 700.SEHK 2024-01~2026-07 的实测样本长度
DAILY_VOL = 0.012            # 日波动 1.2%，接近港美股单标的 CTA 的实际量级


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


def test_a_single_random_series_is_not_significant() -> None:
    """最直白的一条：纯随机序列，检验必须说"不显著"。"""
    rng = np.random.default_rng(20260725)
    result = sharpe_inference(_white_noise(rng), annual_days=ANNUAL_DAYS)

    assert not result.significant
    assert result.p_value > 0.05
    assert abs(result.t_stat) < 1.96
    # 置信区间必须覆盖 0 —— 与"不显著"是同一件事的两种说法
    assert result.ci_low_annual < 0.0 < result.ci_high_annual


@pytest.mark.parametrize("method", ["iid_normal", "iid_nonnormal", "hac"])
def test_size_on_iid_normal_returns_is_about_five_percent(method: str) -> None:
    """Monte-Carlo size test：真实 SR = 0 时，5% 水平的拒绝率必须 ≈ 5%。"""
    reps = 1500
    rng = np.random.default_rng(1234)
    rejections = sum(
        sharpe_inference(
            _white_noise(rng), annual_days=ANNUAL_DAYS, method=method, n_boot=0
        ).significant
        for _ in range(reps)
    )
    rate = rejections / reps
    low, high = _binomial_bounds(0.05, reps)
    assert low < rate < high, f"{method} 的经验 size = {rate:.4f}，偏离名义 5% 太远"


def test_size_holds_under_fat_tails_and_skew() -> None:
    """厚尾 / 偏态下 size 仍需受控 —— 交易策略的日收益从来不是正态的。"""
    reps = 1200
    rng = np.random.default_rng(555)

    skewed = [
        (np.exp(rng.standard_normal(PROJECT_N) * 0.5) - math.exp(0.125)) * 0.01
        for _ in range(reps)
    ]
    rate_normal = sum(
        sharpe_inference(r, annual_days=ANNUAL_DAYS, method="iid_normal", n_boot=0).significant
        for r in skewed
    ) / reps
    rate_mertens = sum(
        sharpe_inference(r, annual_days=ANNUAL_DAYS, method="iid_nonnormal", n_boot=0).significant
        for r in skewed
    ) / reps

    low, high = _binomial_bounds(0.05, reps, sigmas=4.0)
    assert rate_mertens < high, f"Mertens 修正后 size = {rate_mertens:.4f} 仍然过度拒绝"
    assert abs(rate_mertens - 0.05) <= abs(rate_normal - 0.05) + 0.004, (
        f"高阶矩修正没有改善 size：正态 {rate_normal:.4f} → 非正态 {rate_mertens:.4f}"
    )


def test_bootstrap_also_says_not_significant_on_random_data() -> None:
    """独立第二意见（stationary bootstrap）在随机数据上同样不能判显著。"""
    reps = 200
    rng = np.random.default_rng(99)
    rejections = 0
    for i in range(reps):
        p, _ = bootstrap_sharpe_test(_white_noise(rng), n_boot=299, seed=7000 + i)
        rejections += p < 0.05
    rate = rejections / reps
    assert rate < 0.12, f"bootstrap 经验 size = {rate:.3f}，检验无效"


def test_a_random_series_stays_insignificant_at_any_annualization() -> None:
    """t 与 p 对 annual_days 缩放不变 —— 年化只是展示层，不能改变结论。"""
    rng = np.random.default_rng(4242)
    returns = _white_noise(rng)
    a = sharpe_inference(returns, annual_days=247, n_boot=0)
    b = sharpe_inference(returns, annual_days=252, n_boot=0)

    assert a.t_stat == pytest.approx(b.t_stat, rel=1e-12)
    assert a.p_value == pytest.approx(b.p_value, rel=1e-12)
    assert a.significant == b.significant
    # 年化展示层确实按 √annual_days 缩放
    assert b.sharpe_annual / a.sharpe_annual == pytest.approx(math.sqrt(252 / 247), rel=1e-12)


# 二、功效：真有 alpha 的序列必须被识别出来


def test_a_strong_drift_is_detected() -> None:
    """真实年化 Sharpe = 2.5 时，611 日样本应当稳定判显著。"""
    rng = np.random.default_rng(31337)
    mu = 2.5 / math.sqrt(ANNUAL_DAYS) * DAILY_VOL
    returns = rng.standard_normal(PROJECT_N) * DAILY_VOL + mu

    result = sharpe_inference(returns, annual_days=ANNUAL_DAYS)
    assert result.significant
    assert result.p_value < 0.01
    assert result.ci_low_annual > 0.0
    assert result.bootstrap_p_value < 0.05


def test_power_at_the_significance_threshold_is_only_about_half() -> None:
    """样本 SR 刚好等于门槛 ⇒ 真实 SR 等于门槛时功效只有约 50%。

    门槛必须与 significant 的判定规则同侧。significant 判的是单尾
    "SR>0"（交易上只有正夏普算赚钱，负夏普再显著也是显著地在亏），
    所以这里的真实 SR 也要取单尾门槛，否则测的是两个不同的假设。
    """
    reps = 600
    rng = np.random.default_rng(2718)
    threshold = required_sharpe(PROJECT_N, ANNUAL_DAYS, alpha=0.05, one_sided=True)
    mu = threshold / math.sqrt(ANNUAL_DAYS) * DAILY_VOL

    detected = sum(
        sharpe_inference(
            rng.standard_normal(PROJECT_N) * DAILY_VOL + mu,
            annual_days=ANNUAL_DAYS, n_boot=0,
        ).significant
        for _ in range(reps)
    )
    power = detected / reps
    assert 0.40 < power < 0.60, f"门槛处功效 = {power:.3f}，理论应≈0.50"


# 三、序列相关：为什么必须用 HAC 修正


def test_iid_formula_over_rejects_under_autocorrelation_and_hac_fixes_it() -> None:
    """AR(1) 自相关下 iid 公式会把 size 从 5% 推到约 15%，HAC 必须把它拉回来。"""
    reps = 800
    rho = 0.3
    rng = np.random.default_rng(864)
    series = _ar1(rng, rho, reps=reps)

    rate_iid = sum(
        sharpe_inference(series[i], annual_days=ANNUAL_DAYS, method="iid_normal", n_boot=0).significant
        for i in range(reps)
    ) / reps
    rate_hac = sum(
        sharpe_inference(series[i], annual_days=ANNUAL_DAYS, method="hac", n_boot=0).significant
        for i in range(reps)
    ) / reps

    theoretical = 2.0 * norm_sf(1.959964 / math.sqrt((1 + rho) / (1 - rho)))
    assert rate_iid > 0.10, f"iid 在 ρ={rho} 下拒绝率仅 {rate_iid:.3f}，与理论 {theoretical:.3f} 不符"
    assert rate_hac < rate_iid - 0.03, f"HAC 没有改善：iid {rate_iid:.3f} → hac {rate_hac:.3f}"
    # HAC 在 600 样本上仍有已知的有限样本偏误，不会精确回到 5% —— 如实钉住
    assert rate_hac < 0.10, f"HAC 修正后拒绝率 {rate_hac:.3f} 仍然过高"


def test_hac_inflation_tracks_the_ar1_theoretical_factor() -> None:
    """HAC/iid 的 SE 比值应接近 √((1+ρ)/(1−ρ))。"""
    rng = np.random.default_rng(20260101)
    rho = 0.3
    series = _ar1(rng, rho, n=4000)[0]
    result = sharpe_inference(series, annual_days=ANNUAL_DAYS, n_boot=0)

    expected = math.sqrt((1 + rho) / (1 - rho))
    assert result.hac_inflation == pytest.approx(expected, rel=0.20)
    assert result.autocorr_lag1 == pytest.approx(rho, abs=0.05)
    assert result.ljung_box_p < 0.01          # 自相关必须被诊断出来
    assert any("序列相关" in w or "自相关" in w for w in result.warnings)


def test_hac_with_zero_lags_equals_iid_nonnormal_closed_form() -> None:
    """lags=0 的 GMM 三明治 = Mertens (2002) 闭式，必须逐位相等。"""
    rng = np.random.default_rng(77)
    r = rng.standard_t(4, PROJECT_N) * 0.006 + 0.0005

    mean = float(r.mean())
    sharpe = (mean - 0.0) / float(r.std(ddof=1))
    dev = r - mean
    m2 = float((dev ** 2).mean())
    skew = float((dev ** 3).mean()) / m2 ** 1.5
    kurt = float((dev ** 4).mean()) / m2 ** 2

    assert sharpe_variance_factor(r, sharpe, 0) == pytest.approx(
        iid_nonnormal_variance_factor(sharpe, skew, kurt), rel=1e-12
    )


def test_mertens_reduces_to_lo_under_normality() -> None:
    """γ₃=0、γ₄=3 时 Mertens 闭式退化为 Lo 的 1 + SR²/2。"""
    for sharpe in (-0.2, 0.0, 0.05, 0.3):
        assert iid_nonnormal_variance_factor(sharpe, 0.0, 3.0) == pytest.approx(
            iid_normal_variance_factor(sharpe), rel=1e-15
        )


def test_lo_2002_standard_error_matches_the_textbook_formula() -> None:
    """SE = √((1 + SR²/2)/n)，Lo (2002) 式(9)。"""
    rng = np.random.default_rng(11)
    r = rng.standard_normal(PROJECT_N) * DAILY_VOL + 0.0004
    result = sharpe_inference(r, annual_days=ANNUAL_DAYS, method="iid_normal", n_boot=0)

    expected = math.sqrt((1 + result.sharpe_period ** 2 / 2) / PROJECT_N)
    assert result.standard_error_period == pytest.approx(expected, rel=1e-12)
    assert result.standard_error_annual == pytest.approx(
        expected * math.sqrt(ANNUAL_DAYS), rel=1e-12
    )


def test_reproduces_the_reported_134_day_case() -> None:
    """复刻项目里实际观察到的那条曲线：n=134，年化 sharpe=−1.68 → SE≈1.36。"""
    n = 134
    sharpe_annual = -1.68
    sharpe_period = sharpe_annual / math.sqrt(ANNUAL_DAYS)
    se_period = math.sqrt((1 + sharpe_period ** 2 / 2) / n)

    assert se_period * math.sqrt(ANNUAL_DAYS) == pytest.approx(1.36, abs=0.01)
    assert abs(sharpe_period / se_period) < 1.96          # 不显著


def test_variance_factor_rejects_an_annualized_sharpe() -> None:
    """传年化 Sharpe 进单期公式是最常见的实现错误，必须直接报错。"""
    rng = np.random.default_rng(5)
    r = rng.standard_normal(300) * DAILY_VOL
    with pytest.raises(ValueError, match="年化"):
        sharpe_variance_factor(r, 1.5, 5)
    # strict=False 时放行（引擎内部路径用），但不静默
    assert math.isfinite(sharpe_variance_factor(r, 1.5, 5, strict=False))


def test_newey_west_bandwidth_rule() -> None:
    """⌊4·(n/100)^(2/9)⌋，n=611 → 5。"""
    assert newey_west_lags(611) == 5
    assert newey_west_lags(100) == 4
    assert newey_west_lags(1) == 0
    assert newey_west_lags(0) == 0


def test_required_sharpe_for_the_project_sample_size() -> None:
    """本项目 611 日样本的门槛：双尾 5% 需年化 Sharpe ≈ 1.25（港股 247 日）。"""
    two_sided = required_sharpe(611, ANNUAL_DAYS, alpha=0.05, one_sided=False)
    one_sided = required_sharpe(611, ANNUAL_DAYS, alpha=0.05, one_sided=True)

    assert two_sided == pytest.approx(1.25, abs=0.01)
    assert one_sided == pytest.approx(1.05, abs=0.01)
    assert one_sided < two_sided

    # 美股 252 日口径略高，因为同样 611 天折算成更少的"年"
    assert required_sharpe(611, 252, 0.05, False) == pytest.approx(1.26, abs=0.01)


def test_required_sharpe_closed_form_matches_the_actual_test() -> None:
    """把门槛值构造成样本 Sharpe，检验的 t 必须恰好落在临界值上。"""
    n = 611
    threshold_annual = required_sharpe(n, ANNUAL_DAYS, 0.05, False)
    threshold_period = threshold_annual / math.sqrt(ANNUAL_DAYS)

    # 构造一条恰好具有该 Sharpe 的正态序列（偏度 0、峰度 3 才对得上闭式）
    rng = np.random.default_rng(1)
    base = rng.standard_normal(n)
    base = (base - base.mean()) / base.std(ddof=1)
    returns = base * DAILY_VOL + threshold_period * DAILY_VOL

    result = sharpe_inference(returns, annual_days=ANNUAL_DAYS, method="iid_normal", n_boot=0)
    assert result.sharpe_annual == pytest.approx(threshold_annual, rel=1e-9)
    assert result.p_value == pytest.approx(0.05, abs=0.005)


def test_required_sharpe_rises_with_serial_correlation() -> None:
    """SE 放大 f 倍 ⇒ 门槛抬高约 f 倍。"""
    plain = required_sharpe(611, ANNUAL_DAYS, se_inflation=1.0)
    inflated = required_sharpe(611, ANNUAL_DAYS, se_inflation=1.3)
    assert inflated / plain == pytest.approx(1.3, rel=0.01)


def test_deflate_optimization_results_rejects_a_pure_noise_sweep() -> None:
    """把 400 组【纯噪声】回测喂进去，最优组的 DSR 必须判为不可信。"""
    rng = np.random.default_rng(1000)
    results = []
    for i in range(400):
        r = rng.standard_normal(PROJECT_N) * DAILY_VOL
        annual = float(r.mean() / r.std(ddof=1) * math.sqrt(ANNUAL_DAYS))
        results.append(({"trial": i}, annual, {}))
    results.sort(key=lambda row: row[1], reverse=True)

    report = deflate_optimization_results(results, PROJECT_N, "sharpe_ratio", ANNUAL_DAYS)
    assert report["n_trials"] == 400
    assert report["best_sharpe_annual"] > 0.7        # 纯运气也能挑出漂亮的 Sharpe
    assert report["deflated_sharpe_ratio"] < 0.95
    assert report["trustworthy"] is False


def test_norm_ppf_matches_scipy() -> None:
    scipy_special = pytest.importorskip("scipy.special")
    probabilities = [1e-12, 1e-9, 1e-6, 1e-3, 0.02425, 0.05, 0.5, 0.95, 0.975, 0.999, 1 - 1e-9]
    for p in probabilities:
        assert norm_ppf(p) == pytest.approx(float(scipy_special.ndtri(p)), rel=1e-13, abs=1e-13)


def test_rejects_degenerate_inputs() -> None:
    rng = np.random.default_rng(700)
    r = _white_noise(rng, 300)

    with pytest.raises(ValueError, match="太小"):
        sharpe_inference(r[:5])
    with pytest.raises(ValueError, match="annual_days"):
        sharpe_inference(r, annual_days=0)
    with pytest.raises(ValueError, match="confidence"):
        sharpe_inference(r, confidence=1.5)
    with pytest.raises(ValueError, match="标准差为 0"):
        sharpe_inference(np.zeros(300))
    with pytest.raises(ValueError, match="未知 method"):
        sharpe_inference(r, method="bogus")


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


def test_statistics_dict_exposes_the_significance_fields() -> None:
    """statistics 里必须能取到 SE / t / p，否则寻优和 GUI 都看不到。"""
    from vnpy_ctastrategy.backtesting import BacktestingEngine

    rng = np.random.default_rng(3)
    engine = BacktestingEngine()
    engine.capital = 1_000_000
    engine.annual_days = ANNUAL_DAYS
    # 均值取 0：这里要的就是"纯噪音"。原本写的是 200/天漂移（真实年化
    # SR≈0.26），配上一次运气好的抽样得到样本 SR=1.14、t=1.83 —— 旧的两尾
    # 规则替它挡下来了，测试因此看起来是在测噪音，实际不是。
    stats = engine.calculate_statistics(
        df=_daily_df(rng.normal(0.0, 12_000.0, PROJECT_N)), output=False
    )

    for key in (
        "sharpe_se", "sharpe_tstat", "sharpe_pvalue", "sharpe_ci_low", "sharpe_ci_high",
        "sharpe_significant", "sharpe_required_for_significance", "sharpe_hac_inflation",
    ):
        assert key in stats, f"statistics 缺少 {key}"

    # 纯随机 PnL：不能判显著
    assert not stats["sharpe_significant"]
    assert stats["sharpe_pvalue"] > 0.05
    # t 必须与面板上的 sharpe_ratio 自洽
    assert stats["sharpe_tstat"] == pytest.approx(
        stats["sharpe_ratio"] / stats["sharpe_se"], rel=1e-9
    )


def test_significance_is_one_sided_and_requires_a_positive_sharpe() -> None:
    """判定必须是单尾的 SR>0，不是两尾的 SR≠0。

    交易上只有正夏普算赚钱：一条显著为负的曲线是"显著地在亏"，两尾规则
    会把它标成 significant=True，读的人会当成"这条策略通过了检验"。

    实测（HK 240 交易日、600 天）：
      亏损曲线  SR=-7.35  两尾 p≈0 → 旧规则 True；单尾+符号 → False
      边界曲线  SR=+1.14  t=1.83   → 两尾 p=0.067 不显著；单尾 p=0.034 显著
    第二条是两个规则唯一分歧的区间，改回两尾会让它掉出来。
    """
    losing = sharpe_inference(
        np.full(PROJECT_N, -DAILY_VOL * 0.5), annual_days=ANNUAL_DAYS, n_boot=0
    )
    assert losing.sharpe_annual < 0
    assert not losing.significant, "显著为负绝不能报显著"

    rng = np.random.default_rng(3)
    borderline = sharpe_inference(
        rng.standard_normal(PROJECT_N) * DAILY_VOL + 200.0 / 1_000_000,
        annual_days=ANNUAL_DAYS, n_boot=0,
    )
    if 1.65 < borderline.t_stat < 1.96:
        assert borderline.significant, "单尾门槛内的正夏普应当判显著"


def test_engine_sharpe_matches_module_sharpe_bit_for_bit() -> None:
    """引擎里不能有第二套 Sharpe：模块算出的年化值必须等于 statistics["sharpe_ratio"]。"""
    from vnpy_ctastrategy.backtesting import BacktestingEngine

    rng = np.random.default_rng(31)
    engine = BacktestingEngine()
    engine.capital = 1_000_000
    engine.annual_days = ANNUAL_DAYS
    df = _daily_df(rng.normal(900.0, 9_000.0, 500))
    stats = engine.calculate_statistics(df=df, output=False)

    direct = inference_from_daily_df(df, annual_days=ANNUAL_DAYS, n_boot=0)
    assert direct.sharpe_annual == pytest.approx(stats["sharpe_ratio"], rel=1e-9)


def test_nan_p_values_degrade_to_one_not_to_zero() -> None:
    """p 值缺失必须兜成 1.0（不显著）。"""
    rng = np.random.default_rng(705)
    result = sharpe_inference(_white_noise(rng, 300), n_boot=0)   # n_boot=0 → bootstrap p 为 NaN
    assert math.isnan(result.bootstrap_p_value)

    fields = statistics_fields(result)
    assert fields["sharpe_bootstrap_pvalue"] == 1.0
    assert np.nan_to_num(fields["sharpe_bootstrap_pvalue"]) == 1.0
