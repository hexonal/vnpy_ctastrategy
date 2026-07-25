"""分块重排显著性检验的测试。

分四层：
  1. 组件层 —— 块长选择、索引生成、p 值公式各自的数学性质
  2. **有效性层（最关键）** —— 对纯随机策略，检验必须给出"不显著"，
     且 p 值在零假设下近似均匀、第一类错误率不超过 α。
     检验方法本身如果连这关都过不了，功效再高也只是在批量制造假阳性。
  3. 功效层 —— 真有 edge 时能不能检出，以及 600 日样本的功效实测
  4. 集成层 —— 与 daily_df / statistics dict / 寻优的对接，含失败降级
"""

from __future__ import annotations

import numpy as np
import pytest
from pandas import DataFrame

from vnpy_ctastrategy.permutation_test import (
    PermutationScheme,
    attach_permutation_statistics,
    block_permutation_indices,
    circular_block_indices,
    empirical_p_value,
    gated_statistic,
    make_statistic,
    minimum_detectable_sharpe,
    optimal_block_length,
    permutation_statistics,
    permutation_test_bars,
    permutation_test_positions,
    permutation_test_returns,
    permute_ohlc,
    permute_price_series,
    random_rotation_indices,
    sharpe_standard_error,
    stationary_bootstrap_indices,
)

CAPITAL = 1_000_000.0
ANNUAL_DAYS = 252


# ══════════════════════════════════════════════════════════════════════
# 测试用具：合成 daily_df 与合成策略
# ══════════════════════════════════════════════════════════════════════

def build_daily_df(
    close: np.ndarray,
    exposure: np.ndarray,
    size: float = 1.0,
    cost_per_notional: float = 0.0,
) -> DataFrame:
    """按 vnpy DailyResult 的口径造一个 daily_df。

    holding_pnl = start_pos × (close − pre_close) × size，与 backtesting.py 一致；
    成本按仓位变动的名义金额收取。这样造出来的 df 能被 permutation_test_positions
    完美重构（reconstruction_error ≈ 0），便于把重构误差与统计噪音分开。
    """
    close = np.asarray(close, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    assert close.size == exposure.size

    pre_close = np.concatenate([[close[0]], close[:-1]])
    holding_pnl = exposure * (close - pre_close) * size

    padded = np.concatenate([[0.0], exposure])
    changes = np.abs(np.diff(padded))
    turnover = changes * close * size
    commission = turnover * cost_per_notional

    return DataFrame(
        {
            "close_price": close,
            "pre_close": pre_close,
            "trade_count": (changes > 0).astype(int),
            "start_pos": exposure,
            "end_pos": exposure,
            "turnover": turnover,
            "commission": commission,
            "slippage": np.zeros_like(close),
            "trading_pnl": np.zeros_like(close),
            "holding_pnl": holding_pnl,
            "total_pnl": holding_pnl,
            "net_pnl": holding_pnl - commission,
        }
    )


def persistent_exposure(
    n: int, rng: np.random.Generator, stay: float = 0.95, units: float = 1000.0
) -> np.ndarray:
    """两状态马尔可夫链造持仓：平均持仓段长 1/(1−stay)，模仿海龟的长持仓。"""
    state = np.zeros(n, dtype=float)
    current = 0.0
    for t in range(n):
        if rng.random() > stay:
            current = units - current
        state[t] = current
    return state


def ar1_returns(n: int, rng: np.random.Generator, phi: float, sigma: float) -> np.ndarray:
    values = np.zeros(n, dtype=float)
    for t in range(1, n):
        values[t] = phi * values[t - 1] + rng.normal(0.0, sigma)
    return values


def random_walk_prices(
    n: int, rng: np.random.Generator, drift: float = 0.0, sigma: float = 0.015, start: float = 100.0
) -> np.ndarray:
    steps = rng.normal(drift, sigma, size=n)
    return start * np.exp(np.cumsum(steps))


def ma_crossover_sharpe(
    prices: np.ndarray, fast: int = 10, slow: int = 40, annual_days: int = ANNUAL_DAYS
) -> float:
    """long-only 双均线策略的年化夏普。permutation_test_bars 的重跑回调。

    完整实现，不是示意：快线上穿慢线后满仓持有，下穿后空仓，次日生效（避免前视）。
    """
    prices = np.asarray(prices, dtype=float)
    if prices.size <= slow + 1:
        return 0.0
    series = np.asarray(prices, dtype=float)
    cumsum = np.cumsum(np.concatenate([[0.0], series]))
    fast_ma = (cumsum[fast:] - cumsum[:-fast]) / fast
    slow_ma = (cumsum[slow:] - cumsum[:-slow]) / slow
    aligned_fast = fast_ma[slow - fast :]
    signal = (aligned_fast > slow_ma).astype(float)

    # signal[i] 在第 (slow-1+i) 根收盘时才可知，故持有的是第 (slow+i) 根的收益
    position = signal[:-1]
    market = np.diff(np.log(series))[slow - 1 :]
    strategy = position * market

    std = strategy.std(ddof=1)
    if std == 0:
        return 0.0
    return float(strategy.mean() / std * np.sqrt(annual_days))


# ══════════════════════════════════════════════════════════════════════
# 1. 组件层
# ══════════════════════════════════════════════════════════════════════

def test_optimal_block_length_is_short_for_white_noise() -> None:
    """白噪声没有需要保留的依赖结构，最优块长应该很小。"""
    rng = np.random.default_rng(1)
    result = optimal_block_length(rng.normal(size=600))
    assert result.circular <= 4
    assert result.stationary <= 4
    assert abs(result.lag1_autocorrelation) < 0.15


def test_optimal_block_length_grows_with_autocorrelation() -> None:
    """自相关越强，需要保留的块越长 —— Politis-White 的核心行为。"""
    rng = np.random.default_rng(2)
    lengths = [
        optimal_block_length(ar1_returns(600, rng, phi, sigma=0.01)).circular
        for phi in (0.0, 0.5, 0.9)
    ]
    assert lengths[0] < lengths[1] < lengths[2]


def test_optimal_block_length_handles_degenerate_input() -> None:
    """常数序列 / 过短序列不能抛异常，返回块长 1 由上层的块数检查兜住。"""
    assert optimal_block_length(np.ones(500)).circular == 1
    assert optimal_block_length(np.array([1.0, 2.0, 3.0])).circular == 1
    assert optimal_block_length(np.array([])).circular == 1


def test_block_permutation_is_a_true_permutation() -> None:
    """严格置换：每个位置恰好出现一次。夏普恒等不变正是由这条性质推出来的。"""
    rng = np.random.default_rng(3)
    for block_length in (1, 7, 25, 137):
        indices = block_permutation_indices(600, block_length, rng)
        assert indices.size == 600
        assert np.array_equal(np.sort(indices), np.arange(600))


def test_block_permutation_keeps_blocks_contiguous() -> None:
    """块内相邻关系必须保住，否则"分块"二字就没意义了。"""
    rng = np.random.default_rng(4)
    indices = block_permutation_indices(100, 10, rng)
    steps = np.diff(indices)
    # 10 块 → 最多 9 个断点，其余全是 +1
    assert int(np.sum(steps != 1)) <= 9


def test_resampling_indices_stay_in_range() -> None:
    rng = np.random.default_rng(5)
    for generator in (
        lambda: circular_block_indices(300, 20, rng),
        lambda: stationary_bootstrap_indices(300, 20.0, rng),
        lambda: random_rotation_indices(300, rng),
    ):
        indices = generator()
        assert indices.size == 300
        assert indices.min() >= 0
        assert indices.max() < 300


def test_stationary_bootstrap_mean_block_length_matches_target() -> None:
    """块长服从几何分布，平均值应贴近设定值。"""
    rng = np.random.default_rng(6)
    breaks = 0
    trials = 200
    for _ in range(trials):
        indices = stationary_bootstrap_indices(500, 20.0, rng)
        breaks += int(np.sum(np.diff(indices) % 500 != 1)) + 1
    observed_mean_block = trials * 500 / breaks
    assert 14.0 < observed_mean_block < 28.0


def test_rotation_is_a_permutation_and_non_trivial() -> None:
    rng = np.random.default_rng(7)
    indices = random_rotation_indices(50, rng)
    assert np.array_equal(np.sort(indices), np.arange(50))
    assert not np.array_equal(indices, np.arange(50))


def test_empirical_p_value_never_returns_zero() -> None:
    """+1 修正：B 次全部低于观测值，也只能说 p = 1/(B+1)，不能说 p = 0。"""
    null = np.zeros(999)
    assert empirical_p_value(10.0, null) == pytest.approx(1.0 / 1000.0)
    assert empirical_p_value(-10.0, null) == pytest.approx(1.0)


def test_empirical_p_value_alternatives_and_nan_handling() -> None:
    null = np.array([1.0, 2.0, 3.0, np.nan])
    assert empirical_p_value(2.0, null, "greater") == pytest.approx(3.0 / 5.0)
    assert empirical_p_value(2.0, null, "less") == pytest.approx(3.0 / 5.0)
    assert empirical_p_value(2.0, null, "two-sided") == pytest.approx(1.0)
    with pytest.raises(ValueError):
        empirical_p_value(1.0, null, "bigger")


def test_empirical_p_value_is_bounded() -> None:
    rng = np.random.default_rng(8)
    null = rng.normal(size=500)
    for observed in (-5.0, 0.0, 5.0):
        for alternative in ("greater", "less", "two-sided"):
            p = empirical_p_value(observed, null, alternative)
            assert 0.0 < p <= 1.0


def test_sharpe_statistic_matches_vnpy_formula() -> None:
    """统计量必须与 backtesting.calculate_statistics 同口径，否则观测值对不上。"""
    rng = np.random.default_rng(9)
    pnl = rng.normal(500.0, 5000.0, size=400)

    balance = CAPITAL + np.cumsum(pnl)
    previous = np.concatenate([[CAPITAL], balance[:-1]])
    log_returns = np.log(balance / previous)
    expected = (
        log_returns.mean() * 100 / (log_returns.std(ddof=1) * 100) * np.sqrt(ANNUAL_DAYS)
    )

    statistic = make_statistic("sharpe_ratio", CAPITAL, ANNUAL_DAYS)
    assert statistic.evaluate_one(pnl) == pytest.approx(expected, rel=1e-12)


def test_unknown_statistic_name_raises() -> None:
    with pytest.raises(ValueError, match="未知统计量"):
        make_statistic("calmar", CAPITAL, ANNUAL_DAYS)


# ══════════════════════════════════════════════════════════════════════
# 2. 有效性层：随机序列必须判为不显著
# ══════════════════════════════════════════════════════════════════════

def test_random_strategies_are_not_significant() -> None:
    """纯随机策略：持仓与市场收益毫无关系，检验必须判为不显著。

    刻意用 12 条独立随机曲线取中位数，而不是单挑一条断言 p>0.05 ——
    零假设成立时本来就有 5% 的曲线会偶然显著，对单条曲线断言"必须不显著"
    是个按构造就会偶发失败的测试，那种测试不是在检验方法而是在检验运气。
    """
    p_values = []
    for replication in range(12):
        rng = np.random.default_rng(100 + replication)
        n = 600
        prices = random_walk_prices(n, rng)
        exposure = persistent_exposure(n, rng)          # 与价格独立生成
        result = permutation_test_positions(
            build_daily_df(prices, exposure),
            CAPITAL,
            ANNUAL_DAYS,
            n_permutations=499,
            seed=replication,
        )
        p_values.append(result.p_value)

    assert float(np.median(p_values)) > 0.15
    assert sum(p <= 0.05 for p in p_values) <= 3


def test_type_one_error_rate_respects_alpha() -> None:
    """**方法有效性的核心检验**：零假设下 p 值近似均匀，α=0.05 的拒绝率不超过 α。

    200 次独立复现，每次持仓与市场收益独立生成。拒绝率的标准误约 1.5pp，
    上界放到 0.12 —— 若方法系统性乐观（例如误用 iid 重排），拒绝率会明显冲破它。
    """
    rejections = 0
    p_values = []
    replications = 200
    for replication in range(replications):
        rng = np.random.default_rng(1000 + replication)
        n = 400
        prices = random_walk_prices(n, rng)
        exposure = persistent_exposure(n, rng)
        daily_df = build_daily_df(prices, exposure)
        result = permutation_test_positions(
            daily_df,
            CAPITAL,
            ANNUAL_DAYS,
            n_permutations=199,
            seed=replication,
        )
        p_values.append(result.p_value)
        rejections += int(result.p_value <= 0.05)

    rejection_rate = rejections / replications
    assert rejection_rate <= 0.12, f"第一类错误率 {rejection_rate:.3f} 超标"
    # 均匀性粗检：p 值均值应在 0.5 附近
    assert 0.35 < float(np.mean(p_values)) < 0.65


def test_iid_shuffle_understates_null_variance_under_autocorrelation() -> None:
    """**为什么必须分块**的直接证据。

    市场收益带正自相关、持仓有持续性时，把块打散（L=1）会造出现实中不可能的
    持仓路径，零分布方差被系统性低估 —— 于是同一个观测值算出来的 p 值更小（更乐观）。
    """
    rng = np.random.default_rng(202)
    n = 600
    market = ar1_returns(n, rng, phi=0.35, sigma=0.015)
    prices = 100.0 * np.exp(np.cumsum(market))
    exposure = persistent_exposure(n, rng, stay=0.97)
    daily_df = build_daily_df(prices, exposure)

    blocked = permutation_test_positions(
        daily_df, CAPITAL, ANNUAL_DAYS, n_permutations=999, block_length=40, seed=5
    )
    shuffled = permutation_test_positions(
        daily_df, CAPITAL, ANNUAL_DAYS, n_permutations=999, block_length=1, seed=5
    )
    assert blocked.null_std > shuffled.null_std
    assert blocked.p_value >= shuffled.p_value


def test_auto_block_length_tracks_holding_period() -> None:
    """块长在被重排的【持仓序列】上估，应落在平均持仓周期量级，而不是 1-2 天。"""
    rng = np.random.default_rng(303)
    n = 600
    prices = random_walk_prices(n, rng)
    exposure = persistent_exposure(n, rng, stay=0.97)  # 平均持仓段约 33 天
    daily_df = build_daily_df(prices, exposure)

    result = permutation_test_positions(
        daily_df, CAPITAL, ANNUAL_DAYS, n_permutations=99, seed=6
    )
    assert result.block_length >= 8
    assert any("自动选块长" in w for w in result.warnings)


def test_sharpe_is_near_invariant_under_return_permutation() -> None:
    """方案 A 的致命缺陷必须被检出并告警，而不是安静地给出一个假 p 值。

    注意是"近似"不变而非严格不变：vnpy 的夏普走滚动净值的对数收益，
    重排会经复利路径留下二阶残差（实测约为观测值的 0.7%-0.9%）。
    残差非零意味着零分布看上去正常、p 值看上去正常 —— 这比严格退化更危险，
    所以探针必须是相对尺度的，而不是"标准差是不是恰好等于 0"。
    """
    ratios = []
    for seed in (404, 55, 77):
        rng = np.random.default_rng(seed)
        n = 500
        daily_df = build_daily_df(
            random_walk_prices(n, rng, drift=0.0008), persistent_exposure(n, rng)
        )
        result = permutation_test_returns(
            daily_df,
            CAPITAL,
            ANNUAL_DAYS,
            statistic="sharpe_ratio",
            scheme=PermutationScheme.RETURNS_BLOCK,
            n_permutations=300,
            seed=7,
        )
        assert result.null_std > 0.0                       # 不是严格的零
        ratios.append(result.null_std / abs(result.observed))
        assert any("没有功效" in w for w in result.warnings)

    assert max(ratios) < 0.02, f"零分布残差 {max(ratios):.4f} 大于预期的复利二阶量级"


def test_return_permutation_still_works_for_path_dependent_statistics() -> None:
    """同一方案配路径依赖统计量就有功效：收益回撤比在重排下会变。"""
    rng = np.random.default_rng(505)
    n = 500
    prices = random_walk_prices(n, rng, drift=0.001)
    exposure = np.full(n, 1000.0)
    daily_df = build_daily_df(prices, exposure)

    result = permutation_test_returns(
        daily_df,
        CAPITAL,
        ANNUAL_DAYS,
        statistic="return_drawdown_ratio",
        n_permutations=299,
        seed=8,
    )
    assert result.null_std > 0.0
    assert not any("恒等不变" in w for w in result.warnings)


def test_recentred_bootstrap_rejects_a_positive_mean() -> None:
    """RETURNS_BOOTSTRAP_H0 检验 E[收益]=0：给一条明显正漂移的曲线应当显著。"""
    rng = np.random.default_rng(606)
    n = 600
    pnl = rng.normal(1200.0, 8000.0, size=n)         # 日均 +0.12%，年化夏普约 2.4
    daily_df = build_daily_df(
        np.full(n, 100.0), np.zeros(n)
    )
    daily_df["net_pnl"] = pnl

    result = permutation_test_returns(
        daily_df,
        CAPITAL,
        ANNUAL_DAYS,
        statistic="sharpe_ratio",
        scheme=PermutationScheme.RETURNS_BOOTSTRAP_H0,
        n_permutations=999,
        seed=9,
    )
    assert result.p_value < 0.05
    assert any("去均值" in w for w in result.warnings)


def test_recentred_bootstrap_accepts_pure_noise() -> None:
    """同一方案对零均值噪声必须不显著 —— 与上一条配对，防"永远显著"。"""
    rng = np.random.default_rng(707)
    n = 600
    pnl = rng.normal(0.0, 8000.0, size=n)
    daily_df = build_daily_df(np.full(n, 100.0), np.zeros(n))
    daily_df["net_pnl"] = pnl

    result = permutation_test_returns(
        daily_df,
        CAPITAL,
        ANNUAL_DAYS,
        statistic="sharpe_ratio",
        scheme=PermutationScheme.RETURNS_BOOTSTRAP_H0,
        n_permutations=999,
        seed=10,
    )
    assert result.p_value > 0.05


def test_long_only_beta_is_not_counted_as_edge() -> None:
    """**方案 C 的立身之本**：牛市里随机择时的 long-only 策略赚了钱，但不该显著。

    市场年化漂移约 +25%，随机持仓一样能赚 —— 若检验把这个 beta 当 edge，
    项目那种"样本外 -19.5%"的事故就会重演。
    """
    rng = np.random.default_rng(808)
    n = 600
    prices = random_walk_prices(n, rng, drift=0.001, sigma=0.014)
    exposure = persistent_exposure(n, rng, stay=0.96)
    daily_df = build_daily_df(prices, exposure)

    result = permutation_test_positions(
        daily_df, CAPITAL, ANNUAL_DAYS, n_permutations=999, seed=12
    )
    assert result.observed > 0.0            # 确实赚了钱
    assert result.null_mean > 0.0           # 但随机择时也赚
    assert result.p_value > 0.05            # 所以不算 edge


# ══════════════════════════════════════════════════════════════════════
# 3. 功效层
# ══════════════════════════════════════════════════════════════════════

def build_edged_case(
    n: int, rng: np.random.Generator, edge: float, sigma: float = 0.015
) -> DataFrame:
    """造一条持仓真有预测力的曲线：在场那天的市场收益均值被抬高 edge。"""
    exposure = persistent_exposure(n, rng, stay=0.95)
    in_market = exposure > 0
    market = rng.normal(0.0, sigma, size=n) + edge * in_market
    prices = 100.0 * np.exp(np.cumsum(market))
    return build_daily_df(prices, exposure)


def test_strong_edge_is_detected() -> None:
    """真 edge 必须被检出，否则这个检验只会误杀。

    注意即便是这么大的 edge（策略年化夏普约 1.8），p 值也只到 0.01-0.05 这一档 ——
    600 日样本就是这个水平，见模块文档第三节。
    """
    rng = np.random.default_rng(909)
    daily_df = build_edged_case(600, rng, edge=0.0025)
    result = permutation_test_positions(
        daily_df, CAPITAL, ANNUAL_DAYS, n_permutations=999, seed=13
    )
    assert result.p_value < 0.05
    assert result.z_score > 1.6


def test_power_at_600_days() -> None:
    """600 日 / B=199 的功效实测，钉住模块文档第三节那张表。

    edge=0.0030 对应策略夏普约 2.18、择时增量约 1.06、零分布 std 约 0.47，
    200 次复现测得功效 0.72-0.74。这里用 60 次复现，下界放到 0.55
    （二项标准误约 0.06，留 3 个标准误余量）。
    块长逻辑或成本逻辑被改坏时，功效会明显掉出这个区间。
    """
    replications = 60
    rejections = 0
    sharpe_values = []
    increments = []
    for replication in range(replications):
        rng = np.random.default_rng(90000 + replication)
        daily_df = build_edged_case(600, rng, edge=0.0030)
        result = permutation_test_positions(
            daily_df, CAPITAL, ANNUAL_DAYS, n_permutations=199, seed=replication
        )
        sharpe_values.append(result.observed)
        increments.append(result.observed - result.null_mean)
        rejections += int(result.p_value <= 0.05)

    power = rejections / replications
    assert power >= 0.55, f"功效 {power:.2f} 低于预期，块长或成本逻辑可能被改坏"
    assert float(np.mean(sharpe_values)) > 1.8
    # 零分布中心明显为正：市场 beta 被正确地记在市场头上而非策略头上
    assert float(np.mean(increments)) < float(np.mean(sharpe_values)) * 0.75


def test_weak_edge_is_mostly_undetectable_at_600_days() -> None:
    """诚实的另一面：600 日样本对小 edge 基本没有功效，不显著 ≠ 没用。

    edge=0.0005（策略夏普约 0.41）在 200 次复现下实测功效仅 0.13 ——
    这正是"真实 CTA 常见水平本检验看不见"的直接证据。
    """
    replications = 40
    rejections = 0
    for replication in range(replications):
        rng = np.random.default_rng(90000 + replication)
        daily_df = build_edged_case(600, rng, edge=0.0005)
        result = permutation_test_positions(
            daily_df, CAPITAL, ANNUAL_DAYS, n_permutations=199, seed=replication
        )
        rejections += int(result.p_value <= 0.05)
    assert rejections / replications < 0.35


def test_sharpe_standard_error_matches_known_values() -> None:
    """对上项目实测：134 日曲线的年化夏普标准误约 1.36。"""
    assert sharpe_standard_error(134, 252) == pytest.approx(1.371, abs=0.01)
    assert sharpe_standard_error(600, 252) == pytest.approx(0.648, abs=0.01)
    assert sharpe_standard_error(1, 252) == float("inf")


def test_minimum_detectable_sharpe_at_600_days() -> None:
    assert minimum_detectable_sharpe(600, 252, power=0.80) == pytest.approx(1.61, abs=0.02)
    assert minimum_detectable_sharpe(600, 252, power=0.50) == pytest.approx(1.07, abs=0.02)
    # 样本越短门槛越高
    assert minimum_detectable_sharpe(134, 252) > minimum_detectable_sharpe(600, 252)


def test_min_detectable_effect_scales_with_null_std() -> None:
    """结果对象里的门槛是用实测零分布标准差算的，不依赖夏普渐近公式。"""
    rng = np.random.default_rng(1111)
    daily_df = build_edged_case(600, rng, edge=0.002)
    result = permutation_test_positions(
        daily_df, CAPITAL, ANNUAL_DAYS, n_permutations=499, seed=14
    )
    assert result.min_detectable_effect == pytest.approx(2.486 * result.null_std, rel=1e-3)


# ══════════════════════════════════════════════════════════════════════
# 4. 价格重排（方案 B）
# ══════════════════════════════════════════════════════════════════════

def test_permute_price_series_preserves_start_and_return_multiset() -> None:
    rng = np.random.default_rng(1212)
    prices = random_walk_prices(300, rng)
    indices = block_permutation_indices(299, 20, rng)
    permuted = permute_price_series(prices, indices)

    assert permuted[0] == pytest.approx(prices[0])
    assert permuted.size == prices.size
    assert np.sort(np.diff(np.log(permuted))) == pytest.approx(
        np.sort(np.diff(np.log(prices)))
    )
    # 终点也守恒：对数收益总和不变
    assert permuted[-1] == pytest.approx(prices[-1])


def test_permute_price_series_rejects_bad_input() -> None:
    rng = np.random.default_rng(1313)
    prices = random_walk_prices(50, rng)
    with pytest.raises(ValueError, match="长度"):
        permute_price_series(prices, np.arange(50, dtype=np.int64))
    with pytest.raises(ValueError, match="正数"):
        permute_price_series(np.array([1.0, -2.0, 3.0]), np.array([1, 0], dtype=np.int64))


def test_permute_ohlc_keeps_bar_geometry_valid() -> None:
    """重排后的 K 线必须仍然满足 high ≥ max(open, close) ≥ min(open, close) ≥ low。"""
    rng = np.random.default_rng(1414)
    n = 300
    close = random_walk_prices(n, rng)
    open_ = close * np.exp(rng.normal(0.0, 0.004, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, 0.004, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, 0.004, n)))

    indices = block_permutation_indices(n - 1, 15, rng)
    new_open, new_high, new_low, new_close = permute_ohlc(open_, high, low, close, indices)

    assert np.all(new_high >= np.maximum(new_open, new_close) - 1e-9)
    assert np.all(new_low <= np.minimum(new_open, new_close) + 1e-9)
    assert np.all(new_close > 0)


def test_permutation_test_bars_finds_no_edge_in_a_random_walk() -> None:
    """随机游走上的双均线策略：跑完整重跑流程，结论必须是不显著。"""
    rng = np.random.default_rng(1515)
    prices = random_walk_prices(600, rng, drift=0.0)
    result = permutation_test_bars(
        prices, ma_crossover_sharpe, n_permutations=199, seed=15
    )
    assert result.p_value > 0.05


def test_permutation_test_bars_finds_a_planted_trend() -> None:
    """人工种入长趋势段的价格序列上，趋势跟随应当显著。

    这同时说明方案 B 的零假设是"没有趋势结构"：块长以内的自相关被保留，
    块长以上的趋势被打散，所以能检出的正是跨块的趋势。
    """
    rng = np.random.default_rng(1616)
    n = 600
    regime = np.repeat(rng.choice([-1.0, 1.0], size=n // 60), 60)
    steps = 0.0035 * regime + rng.normal(0.0, 0.008, size=n)
    prices = 100.0 * np.exp(np.cumsum(steps))

    result = permutation_test_bars(
        prices, ma_crossover_sharpe, n_permutations=199, block_length=10, seed=16
    )
    assert result.p_value < 0.05


# ══════════════════════════════════════════════════════════════════════
# 5. 集成层
# ══════════════════════════════════════════════════════════════════════

def test_permutation_statistics_keys_survive_nan_to_num() -> None:
    """并进 statistics dict 后要能安全通过 calculate_statistics 末尾那圈过滤。"""
    rng = np.random.default_rng(1717)
    daily_df = build_edged_case(300, rng, edge=0.001)
    stats = permutation_statistics(
        daily_df, CAPITAL, ANNUAL_DAYS, n_permutations=99, seed=17
    )

    assert set(stats) >= {"perm_p_value", "perm_z_score", "perm_block_length", "perm_scheme"}
    for key, value in stats.items():
        filtered = np.nan_to_num(value) if not isinstance(value, str) else value
        assert filtered is not None, key
    assert 0.0 < float(stats["perm_p_value"]) <= 1.0  # type: ignore[arg-type]


def test_attach_permutation_statistics_preserves_existing_keys() -> None:
    rng = np.random.default_rng(1818)
    daily_df = build_edged_case(300, rng, edge=0.001)
    base = {"sharpe_ratio": 1.23, "max_ddpercent": -8.0}
    merged = attach_permutation_statistics(
        base, daily_df, CAPITAL, ANNUAL_DAYS, n_permutations=99, seed=18
    )
    assert merged["sharpe_ratio"] == 1.23
    assert "perm_p_value" in merged
    assert base == {"sharpe_ratio": 1.23, "max_ddpercent": -8.0}   # 原 dict 不被改


def test_attach_permutation_statistics_fails_open() -> None:
    """daily_df 不合法时不能把整条回测流程带崩，只写 perm_error。"""
    merged = attach_permutation_statistics(
        {"sharpe_ratio": 1.0}, DataFrame(), CAPITAL, ANNUAL_DAYS
    )
    assert merged["sharpe_ratio"] == 1.0
    assert "perm_error" in merged
    assert "perm_p_value" not in merged


def test_missing_columns_raise_a_clear_error() -> None:
    with pytest.raises(ValueError, match="缺少列"):
        permutation_test_positions(
            DataFrame({"close_price": [1.0, 2.0]}), CAPITAL, ANNUAL_DAYS
        )


def test_gated_statistic_blocks_insignificant_parameters() -> None:
    """寻优闸：p 不达标就把目标值压到 0，让寻优器自动避开。"""
    assert gated_statistic({"sharpe_ratio": 2.0, "perm_p_value": 0.01}) == 2.0
    assert gated_statistic({"sharpe_ratio": 2.0, "perm_p_value": 0.30}) == 0.0
    assert gated_statistic({"sharpe_ratio": 2.0}) == 0.0                   # 没跑检验
    assert gated_statistic({"perm_p_value": 0.01}) == 0.0                  # 没有目标键
    assert gated_statistic({"sharpe_ratio": 2.0, "perm_p_value": 0.10}, alpha=0.20) == 2.0


def test_reconstruction_error_is_reported_and_flagged() -> None:
    """重构口径与引擎 net_pnl 的偏差要量化上报，偏大时必须告警。"""
    rng = np.random.default_rng(1919)
    daily_df = build_edged_case(400, rng, edge=0.001)
    clean = permutation_test_positions(
        daily_df, CAPITAL, ANNUAL_DAYS, n_permutations=99, seed=19
    )
    assert abs(clean.reconstruction_error) < 1e-9
    assert not any("逐日相对偏差" in w for w in clean.warnings)

    dirty = daily_df.copy()
    dirty["net_pnl"] = dirty["net_pnl"] * 0.5      # 假装引擎里还有一大块日内盈亏
    flagged = permutation_test_positions(
        dirty, CAPITAL, ANNUAL_DAYS, n_permutations=99, seed=19
    )
    assert flagged.reconstruction_error == pytest.approx(1.0, abs=0.02)
    assert any("逐日相对偏差" in w for w in flagged.warnings)


def test_reconstruction_error_is_stable_for_a_break_even_strategy() -> None:
    """逐日口径而非总额口径：策略接近打平时，健康的重构不该被误报成大偏差。

    真实引擎实测踩过这个坑 —— 总收益 -0.0% 时，用"总额偏差 ÷ |总盈亏|"
    会把一个完好的重构算成 29% 误差并触发误告警。
    """
    # 找一条自然打平的曲线：总盈亏 / 逐日盈亏绝对值之和 < 2%
    daily_df = None
    for seed in range(200):
        rng = np.random.default_rng(4000 + seed)
        candidate = build_daily_df(
            random_walk_prices(400, rng), persistent_exposure(400, rng)
        )
        pnl = candidate["net_pnl"].to_numpy(dtype=float)
        if abs(pnl.sum()) / np.abs(pnl).sum() < 0.02:
            daily_df = candidate
            break
    assert daily_df is not None, "没找到接近打平的样本，测试构造失败"

    # 给引擎盈亏加一点点日内残差（逐日 2% 量级），模拟"策略不完全在收盘价成交"
    rng = np.random.default_rng(4242)
    pnl = daily_df["net_pnl"].to_numpy(dtype=float)
    scale = float(np.abs(pnl).mean())
    perturbed = daily_df.copy()
    perturbed["net_pnl"] = pnl + rng.normal(0.0, 0.02 * scale, size=pnl.size)

    result = permutation_test_positions(
        perturbed, CAPITAL, ANNUAL_DAYS, n_permutations=99, seed=26
    )
    # 逐日口径如实反映"2% 量级的残差"，不告警
    assert result.reconstruction_error < 0.05
    assert not any("逐日相对偏差" in w for w in result.warnings)

    # 同一份数据换成旧的总额口径：分母是接近 0 的总盈亏，同样的残差被放大。
    # 判据落到实际后果上 —— 旧口径会越过 10% 告警线误报，新口径不会。
    engine_pnl = perturbed["net_pnl"].to_numpy(dtype=float)
    total_style_error = abs(pnl.sum() - engine_pnl.sum()) / abs(engine_pnl.sum())
    assert total_style_error > 0.10 > result.reconstruction_error


def test_costs_are_recharged_on_permuted_paths() -> None:
    """重排会改变换手次数；不重算成本的零分布偏乐观，p 值会偏小。"""
    rng = np.random.default_rng(2020)
    n = 600
    prices = random_walk_prices(n, rng)
    exposure = persistent_exposure(n, rng, stay=0.97)
    daily_df = build_daily_df(prices, exposure, cost_per_notional=0.002)

    with_costs = permutation_test_positions(
        daily_df, CAPITAL, ANNUAL_DAYS, n_permutations=499, seed=20, include_costs=True
    )
    without_costs = permutation_test_positions(
        daily_df, CAPITAL, ANNUAL_DAYS, n_permutations=499, seed=20, include_costs=False
    )
    assert with_costs.null_mean < without_costs.null_mean
    assert with_costs.p_value <= without_costs.p_value


def test_rotation_scheme_runs_and_flags_its_resolution_limit() -> None:
    rng = np.random.default_rng(2121)
    daily_df = build_edged_case(400, rng, edge=0.002)
    result = permutation_test_positions(
        daily_df,
        CAPITAL,
        ANNUAL_DAYS,
        scheme=PermutationScheme.POSITIONS_ROTATE,
        n_permutations=399,
        seed=21,
    )
    assert result.p_value < 0.05
    assert any("循环平移" in w for w in result.warnings)


def test_min_blocks_guard_clamps_oversized_blocks() -> None:
    """块长大到块数不足时压回去并留痕，避免零分布退化成几个点。"""
    rng = np.random.default_rng(2222)
    daily_df = build_edged_case(200, rng, edge=0.001)
    result = permutation_test_positions(
        daily_df, CAPITAL, ANNUAL_DAYS, n_permutations=99, block_length=150, seed=22
    )
    assert result.block_length <= 200 // 8
    assert any("块数" in w for w in result.warnings)


def test_zero_exposure_yields_a_degenerate_but_safe_result() -> None:
    """从未开仓：零分布退化，检验必须不显著且不抛异常。"""
    n = 300
    rng = np.random.default_rng(2323)
    daily_df = build_daily_df(random_walk_prices(n, rng), np.zeros(n))
    result = permutation_test_positions(
        daily_df, CAPITAL, ANNUAL_DAYS, n_permutations=99, seed=23
    )
    assert result.p_value == pytest.approx(1.0)
    assert result.warnings


def test_results_are_reproducible_under_a_fixed_seed() -> None:
    rng = np.random.default_rng(2424)
    daily_df = build_edged_case(300, rng, edge=0.001)
    first = permutation_test_positions(
        daily_df, CAPITAL, ANNUAL_DAYS, n_permutations=199, seed=42
    )
    second = permutation_test_positions(
        daily_df, CAPITAL, ANNUAL_DAYS, n_permutations=199, seed=42
    )
    assert first.p_value == second.p_value
    assert first.null_std == pytest.approx(second.null_std)


def test_summary_renders_without_error() -> None:
    rng = np.random.default_rng(2525)
    daily_df = build_edged_case(300, rng, edge=0.001)
    result = permutation_test_positions(
        daily_df, CAPITAL, ANNUAL_DAYS, n_permutations=99, seed=25
    )
    text = result.summary()
    assert "重排检验" in text
    assert "p 值" in text
    assert isinstance(result.as_dict(), dict)


# ══════════════════════════════════════════════════════════════════════
# 5. 退化判据层（回归测试：曾经漏判满仓场景）
# ══════════════════════════════════════════════════════════════════════

def build_flat_long_df(n: int, rng: np.random.Generator, deploy_frac: float) -> DataFrame:
    """恒定满仓/部分仓的 buy-and-hold 曲线，deploy_frac = 名义金额 / 本金。"""
    prices = random_walk_prices(n, rng, drift=0.0008)
    units = CAPITAL * deploy_frac / 100.0        # random_walk_prices 起点为 100
    return build_daily_df(prices, np.full(n, units))


@pytest.mark.parametrize("deploy_frac", [0.05, 0.10, 0.25, 0.50, 0.75, 1.00])
def test_invariant_statistic_is_flagged_at_every_deployment_level(
    deploy_frac: float,
) -> None:
    """**回归测试**：夏普×RETURNS_BLOCK 在【任何仓位】下都必须判为无功效。

    历史 bug：旧版判据是 null_std/|观测值| < 5% 这个幅度探针，而复利残差
    ∝ 日盈亏/本金 —— 仓位 5%-25% 时探针 5/5 次响，50% 时 1/5，
    **75%-100% 时 0/5 完全失灵**。港美股现货 long-only 常态就是满仓，
    于是它恰好在本项目的真实场景里最哑，安静地吐出一个衡量"复利记账顺序"的假 p 值。
    现改用多重集合恒等的静态解析判据，与仓位无关。
    """
    rng = np.random.default_rng(4242)
    daily_df = build_flat_long_df(500, rng, deploy_frac)

    result = permutation_test_returns(
        daily_df,
        CAPITAL,
        ANNUAL_DAYS,
        statistic="sharpe_ratio",
        scheme=PermutationScheme.RETURNS_BLOCK,
        n_permutations=299,
        seed=7,
    )
    assert result.degenerate is True
    assert result.has_power is False
    assert result.significant is False
    assert any("没有功效" in w for w in result.warnings)


@pytest.mark.parametrize("statistic", ["sharpe_ratio", "total_return", "annual_return"])
def test_all_multiset_statistics_are_flagged_under_strict_permutation(
    statistic: str,
) -> None:
    """三个多重集合函数在严格置换下全部恒等不变，必须一律判死。"""
    rng = np.random.default_rng(4343)
    daily_df = build_flat_long_df(400, rng, 1.00)
    result = permutation_test_returns(
        daily_df,
        CAPITAL,
        ANNUAL_DAYS,
        statistic=statistic,
        scheme=PermutationScheme.RETURNS_BLOCK,
        n_permutations=199,
        seed=8,
    )
    assert result.degenerate is True


@pytest.mark.parametrize("statistic", ["max_ddpercent", "return_drawdown_ratio"])
def test_path_dependent_statistics_keep_power_under_strict_permutation(
    statistic: str,
) -> None:
    """反向保护：路径依赖统计量【不能】被误判为退化，否则方案 A 整个作废。"""
    rng = np.random.default_rng(4444)
    daily_df = build_flat_long_df(400, rng, 1.00)
    result = permutation_test_returns(
        daily_df,
        CAPITAL,
        ANNUAL_DAYS,
        statistic=statistic,
        scheme=PermutationScheme.RETURNS_BLOCK,
        n_permutations=199,
        seed=9,
    )
    assert result.degenerate is False
    assert result.has_power is True
    assert result.null_std > 0.0


def test_positions_block_with_sharpe_keeps_power() -> None:
    """默认组合（POSITIONS_BLOCK × 夏普）必须始终有功效 —— 它是全模块的主力路径。"""
    rng = np.random.default_rng(4545)
    daily_df = build_edged_case(600, rng, edge=0.001)
    result = permutation_test_positions(
        daily_df, CAPITAL, ANNUAL_DAYS, n_permutations=499, seed=10
    )
    assert result.degenerate is False
    assert result.has_power is True
    assert result.null_std > 0.1          # 实测约 0.45


def test_degenerate_result_can_never_be_significant() -> None:
    """没有功效的检验不允许输出"显著"，哪怕复利残差凑出了一个小 p 值。"""
    rng = np.random.default_rng(4646)
    daily_df = build_flat_long_df(400, rng, 1.00)
    result = permutation_test_returns(
        daily_df,
        CAPITAL,
        ANNUAL_DAYS,
        statistic="total_return",
        scheme=PermutationScheme.RETURNS_BLOCK,
        n_permutations=199,
        seed=11,
    )
    assert result.degenerate is True
    assert result.significant is False
    assert "无功效" in result.summary()


def test_gated_statistic_blocks_degenerate_results() -> None:
    """寻优闸必须读 perm_has_power，而不是只看 p 值。"""
    # 有功效 + p 达标 → 放行
    assert gated_statistic(
        {"sharpe_ratio": 2.0, "perm_p_value": 0.01, "perm_has_power": True}
    ) == 2.0
    # 无功效 + p 看起来达标 → 必须拦下
    assert gated_statistic(
        {"sharpe_ratio": 2.0, "perm_p_value": 0.01, "perm_has_power": False}
    ) == 0.0
    # 缺该键 → 向后兼容，按有功效处理
    assert gated_statistic({"sharpe_ratio": 2.0, "perm_p_value": 0.01}) == 2.0


def test_statistics_dict_exposes_power_flag() -> None:
    """perm_has_power / perm_significant 要进 statistics dict，供发布闸机器判读。"""
    rng = np.random.default_rng(4747)
    daily_df = build_edged_case(300, rng, edge=0.001)
    stats = permutation_statistics(
        daily_df, CAPITAL, ANNUAL_DAYS, n_permutations=99, seed=12
    )
    assert "perm_has_power" in stats
    assert "perm_significant" in stats
    assert isinstance(stats["perm_has_power"], bool)


@pytest.mark.parametrize("drift", [0.0, 0.0004, 0.0008])
def test_beta_never_becomes_edge_regardless_of_market_drift(drift: float) -> None:
    """**方案 C 的核心断言**：市场漂移从 0 涨到年化 +20%，拒绝率必须钉在 α 附近。

    若检验把 long-only 的市场 beta 算进 edge，拒绝率会随漂移单调上升 ——
    那正是"牛市里任何长期在场的规则都显著"的事故模式。
    实测（60 次复现）三档漂移拒绝率均为 0.050。
    """
    rejections = 0
    replications = 40
    for replication in range(replications):
        rng = np.random.default_rng(6000 + replication)
        n = 500
        prices = random_walk_prices(n, rng, drift=drift, sigma=0.015)
        exposure = persistent_exposure(n, rng, stay=0.95)   # 与价格独立 = 零择时
        result = permutation_test_positions(
            build_daily_df(prices, exposure),
            CAPITAL,
            ANNUAL_DAYS,
            n_permutations=199,
            seed=replication,
        )
        rejections += int(result.p_value <= 0.05)

    assert rejections / replications <= 0.15, (
        f"漂移 {drift} 时拒绝率 {rejections / replications:.3f} 超标 —— beta 被当成了 edge"
    )


def test_wrong_exposure_column_is_flagged_and_names_the_right_one() -> None:
    """暴露口径选错要报出来，且**建议的列不能是当前这一列**。

    历史 bug：无论当前用哪一列，告警都写死"换 exposure_column='end_pos' 试试"，
    于是 end_pos 本身出错时会建议你换成 end_pos。

    实测意义（真 DailyResult 端到端）：收盘价成交的策略误用 end_pos，
    会把当天刚建的仓位算成全天持有 = 前视偏差，夏普从 1.00 抬到 2.04、
    p 值从 0.08 压到 0.002 —— 一个纯粹由口径错误造出来的"显著"。
    """
    rng = np.random.default_rng(5151)
    n = 400
    prices = random_walk_prices(n, rng, drift=0.0006)
    exposure = persistent_exposure(n, rng, stay=0.95)

    daily_df = build_daily_df(prices, exposure)
    # end_pos 造成偏移一天的暴露 = 前视
    daily_df["end_pos"] = np.concatenate([exposure[1:], exposure[-1:]])

    result = permutation_test_positions(
        daily_df,
        CAPITAL,
        ANNUAL_DAYS,
        n_permutations=99,
        exposure_column="end_pos",
        seed=21,
    )
    flagged = [w for w in result.warnings if "偏差" in w]
    assert flagged, "口径错误必须告警"
    assert "'start_pos'" in flagged[0], "应当建议换到另一列，而不是当前这一列"
    assert "前视" in flagged[0]
