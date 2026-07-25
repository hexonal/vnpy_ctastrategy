"""RAR / R-Cubed / Robust Sharpe 的回归测试。

指标定义来自社区帖 https://www.vnpy.com/forum/topic/32894，测试按该定义验证，
并额外钉住两条容易被误读的性质（RAR 的逆时间加权、R³ 相对 return_drawdown_ratio 的方向）。
"""

from __future__ import annotations

import numpy as np
import pytest

from vnpy_ctastrategy.robust_metrics import (
    average_top_drawdowns,
    calculate_robust_metrics,
    cumulative_return_curve,
    drawdown_episodes,
    rar_sample_weights,
    regressed_annual_return,
)


CAPITAL = 1_000_000.0
ANNUAL_DAYS = 247            # 港股口径；美股应传 252


# ── RAR ────────────────────────────────────────────────────────────────

def test_rar_on_a_perfectly_linear_curve_matches_endpoint_annualization() -> None:
    """完全线性增长时，回归年化应与端点法年化一致 —— 这是 RAR 的定义锚点。"""
    n = 240
    # 每期累计收益 +0.1 个百分点，即一条过原点的直线
    cumulative = np.arange(1, n + 1, dtype=float) * 0.1
    rar, slope = regressed_annual_return(cumulative, ANNUAL_DAYS)

    assert slope == pytest.approx(0.1, rel=1e-12)
    assert rar == pytest.approx(0.1 * ANNUAL_DAYS, rel=1e-12)

    endpoint_annual = cumulative[-1] / n * ANNUAL_DAYS
    assert rar == pytest.approx(endpoint_annual, rel=1e-12)


def test_rar_penalizes_a_late_spike_versus_steady_growth() -> None:
    """帖子的核心主张：同样的期末收益，靠最后几天暴涨拉起来的曲线 RAR 更低。

    端点法年化对两者给出完全相同的数字 —— 这正是 RAR 想补的盲区。
    """
    n = 240
    steady = np.linspace(0, 24.0, n)                  # 稳步涨到 +24%
    late_spike = np.zeros(n)
    late_spike[-5:] = np.linspace(0, 24.0, 5)         # 最后 5 天才涨到 +24%

    assert steady[-1] == pytest.approx(late_spike[-1])   # 端点相同

    rar_steady, _ = regressed_annual_return(steady, ANNUAL_DAYS)
    rar_spike, _ = regressed_annual_return(late_spike, ANNUAL_DAYS)
    assert rar_spike < rar_steady


def test_rar_rewards_an_early_gain_over_a_late_one() -> None:
    """与上一条同源的另一面：涨得早，RAR 更高。"""
    n = 240
    early = np.full(n, 20.0)
    early[0] = 0.0                                     # 第 1 天就涨到 +20% 后横盘
    late = np.zeros(n)
    late[-1] = 20.0                                    # 最后 1 天才涨到 +20%

    rar_early, _ = regressed_annual_return(early, ANNUAL_DAYS)
    rar_late, _ = regressed_annual_return(late, ANNUAL_DAYS)
    assert rar_early > rar_late


def test_rar_regression_passes_through_origin() -> None:
    """帖子指定无截距（y = a·x）。给一条整体平移过的曲线，斜率不应被截距吸收。"""
    n = 100
    x = np.arange(1, n + 1, dtype=float)
    shifted = 5.0 + 0.2 * x            # 若拟合了截距，斜率会正好是 0.2
    _, slope = regressed_annual_return(shifted, ANNUAL_DAYS)
    # 过原点回归会把那个 +5 的平移也算进斜率，所以必然大于 0.2
    assert slope > 0.2


def test_rar_empty_and_degenerate_inputs() -> None:
    assert regressed_annual_return(np.array([]), ANNUAL_DAYS) == (0.0, 0.0)
    assert regressed_annual_return(np.zeros(10), ANNUAL_DAYS) == (0.0, 0.0)


# ── RAR 的逆时间加权（必须钉住，否则会被误用作样本外判据）──────────────

def test_rar_weights_decay_monotonically_to_near_zero() -> None:
    """RAR 对单期收益的隐含权重 ∝ (n²−j²)，随时间单调递减、末期趋近 0。

    这条性质决定了 RAR 不适合用来判断"最近是否仍然有效" —— 它恰好把最新的证据
    权重压到接近零。文档里写了，这里用数字钉住，防止有人当它是等权指标。
    """
    n = 250
    weights = rar_sample_weights(n)

    assert weights.size == n
    assert weights.mean() == pytest.approx(1.0, rel=1e-12)      # 归一到等权=1
    assert np.all(np.diff(weights) < 0), "权重必须单调递减"

    assert weights[0] == pytest.approx(1.497, abs=0.01)          # 首期 ≈1.50×
    assert weights[-1] < 0.02                                    # 末期 ≈0.012×

    # 与等权的交叉点在 n/√3 ≈ 57.7% 处
    crossing = int(np.argmin(np.abs(weights - 1.0)))
    assert 0.55 * n < crossing < 0.60 * n


# ── 回撤段切分 ─────────────────────────────────────────────────────────

def test_drawdown_episodes_are_independent_not_the_same_valley() -> None:
    """必须切成独立回撤段。若直接对逐日回撤取前 N 大，它们会落在同一个谷底，
    平均下来就退化成最大回撤，R³ 也就退化成 return_drawdown_ratio。"""
    balance = np.array([
        100.0, 110.0, 99.0, 110.0,      # 段1：110 → 99，-10%
        120.0, 108.0, 120.0,            # 段2：120 → 108，-10%
        130.0, max(0.0, 91.0), 130.0,   # 段3：130 → 91，-30%
    ])
    episodes = drawdown_episodes(balance)
    assert len(episodes) == 3
    assert episodes[0] == pytest.approx(10.0, abs=0.01)
    assert episodes[1] == pytest.approx(10.0, abs=0.01)
    assert episodes[2] == pytest.approx(30.0, abs=0.01)


def test_unrecovered_final_drawdown_still_counts() -> None:
    """仍在回撤中（尚未创新高）的最后一段必须计入，否则正在深度回撤的曲线
    反而因为"还没结束"而不被扣分。"""
    balance = np.array([100.0, 120.0, 90.0])       # 从 120 跌到 90 且未收复
    episodes = drawdown_episodes(balance)
    assert len(episodes) == 1
    assert episodes[0] == pytest.approx(25.0, abs=0.01)


def test_monotonic_curve_has_no_drawdown_episode() -> None:
    assert drawdown_episodes(np.array([100.0, 101.0, 102.0])) == []
    assert average_top_drawdowns(np.array([100.0, 101.0])) == (0.0, 0)


def test_average_top_drawdowns_does_not_pad_with_zeros() -> None:
    """段数不足 N 时对现有段取平均。补零会凭空压低分母、虚高 R³。"""
    balance = np.array([100.0, 110.0, 99.0, 110.0, 120.0, 108.0, 120.0])
    avg, count = average_top_drawdowns(balance, top_n=5)
    assert count == 2
    assert avg == pytest.approx(10.0, abs=0.01)      # 而不是 (10+10+0+0+0)/5 = 4


# ── 三个指标合成 ───────────────────────────────────────────────────────

def _linear_balance(n: int, total_return_pct: float) -> np.ndarray:
    return CAPITAL * (1.0 + np.linspace(0.0, total_return_pct / 100.0, n))


def test_calculate_robust_metrics_end_to_end() -> None:
    balance = _linear_balance(240, 24.0)
    returns = np.diff(balance) / balance[:-1]
    returns = np.concatenate([[0.0], returns])

    m = calculate_robust_metrics(balance, returns, CAPITAL, ANNUAL_DAYS)

    assert m.sample_size == 240
    assert m.annual_days == ANNUAL_DAYS
    assert m.regressed_annual_return > 0
    assert m.robust_sharpe > 0
    # 单调上行曲线没有回撤段 → R³ 分母为 0，按 vnpy 惯例返回 0
    assert m.drawdown_episode_count == 0
    assert m.r_cubed == 0.0


def test_robust_sharpe_definition() -> None:
    """RobustSharpe = RAR / (日收益std × √annual_days)，逐项复算。"""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.01, 250)
    balance = CAPITAL * np.cumprod(1.0 + returns)

    m = calculate_robust_metrics(balance, returns, CAPITAL, ANNUAL_DAYS)

    expected_std = float(np.std(returns, ddof=1) * 100.0)
    expected = m.regressed_annual_return / (expected_std * np.sqrt(ANNUAL_DAYS))
    assert m.return_std == pytest.approx(expected_std, rel=1e-12)
    assert m.robust_sharpe == pytest.approx(expected, rel=1e-12)


def test_r_cubed_definition() -> None:
    """R³ = RAR / 前 N 大回撤段的平均幅度，逐项复算。"""
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0004, 0.012, 300)
    balance = CAPITAL * np.cumprod(1.0 + returns)

    m = calculate_robust_metrics(balance, returns, CAPITAL, ANNUAL_DAYS)
    avg, count = average_top_drawdowns(balance, top_n=5)

    assert m.avg_top_drawdown == pytest.approx(avg, rel=1e-12)
    assert m.drawdown_episode_count == count
    if avg:
        assert m.r_cubed == pytest.approx(m.regressed_annual_return / avg, rel=1e-12)


def test_r_cubed_denominator_is_smaller_than_max_drawdown() -> None:
    """R³ 的分母（前 N 大回撤的【平均】）必然小于最大回撤，所以在【分母层面】
    它比 return_drawdown_ratio 宽松。R³ 的"稳健"指的是估计量方差小
    （不被单次极端回撤独占分母），不是结论更保守。

    但【不能】据此推断 R³ 的数值一定大于 return_drawdown_ratio —— 两者分子
    口径不同：R³ 用 RAR（年化），return_drawdown_ratio 用 total_return（总收益）。
    实测 700.SEHK 2024-01..2026-07：avg_top5=1.94 < max_dd=2.68（分母如预期），
    但 R³=1.77 < return_drawdown_ratio=2.90，因为分子是 3.44 对 7.77。
    两个指标不可直接比大小，只能各自纵向对比。"""
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0005, 0.012, 400)
    balance = CAPITAL * np.cumprod(1.0 + returns)

    avg_dd, count = average_top_drawdowns(balance, top_n=5)
    peak = np.maximum.accumulate(balance)
    max_dd = float(np.max((peak - balance) / peak) * 100.0)

    assert count >= 5, "样本需要足够多的回撤段才能体现这条性质"
    assert avg_dd < max_dd


def test_zero_capital_is_rejected() -> None:
    with pytest.raises(ValueError):
        cumulative_return_curve(np.array([1.0, 2.0]), 0.0)


# ── 与回测引擎的集成 ───────────────────────────────────────────────────

def _daily_df(net_pnl: np.ndarray):
    """构造 calculate_statistics 需要的最小 daily_df（列名取自引擎实际读取的字段）。"""
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
        index=pd.to_datetime([date(2026, 1, 1) + timedelta(days=i) for i in range(n)]),
    )


def test_statistics_dict_exposes_the_new_metrics() -> None:
    """calculate_statistics 必须把三个指标放进 statistics 字典，
    否则参数寻优和 GUI 都取不到。"""
    from vnpy_ctastrategy.backtesting import BacktestingEngine

    rng = np.random.default_rng(3)
    net_pnl = rng.normal(500.0, 8000.0, 250)

    engine = BacktestingEngine()
    engine.capital = int(CAPITAL)
    engine.annual_days = ANNUAL_DAYS
    stats = engine.calculate_statistics(df=_daily_df(net_pnl), output=False)

    for key in ("regressed_annual_return", "r_cubed", "robust_sharpe", "drawdown_episode_count"):
        assert key in stats, f"statistics 缺少 {key}"

    # 与直接调用模块的结果必须一致：引擎里不能有第二套算法
    balance = np.cumsum(net_pnl) + CAPITAL
    returns = np.concatenate([[0.0], np.diff(balance) / balance[:-1]])
    direct = calculate_robust_metrics(balance, returns, CAPITAL, ANNUAL_DAYS)
    assert stats["regressed_annual_return"] == pytest.approx(
        direct.regressed_annual_return, rel=1e-9
    )


def test_empty_backtest_still_returns_the_new_keys_absent() -> None:
    """空回测走提前返回分支，不应因为新增指标而抛异常。"""
    from vnpy_ctastrategy.backtesting import BacktestingEngine

    engine = BacktestingEngine()
    assert engine.calculate_statistics(df=None, output=False) == {}
