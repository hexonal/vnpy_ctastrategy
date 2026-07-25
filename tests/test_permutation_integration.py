"""BacktestingEngine × permutation_test 的接线测试。

`tests/test_permutation_test.py` 检验的是**方法本身**（第一类错误率、功效、块长）。
本文件只检验**接线**：开关语义、字段契约、fail-open、以及"引擎这条路径上跑出来的
结论与直接调用模块一致"。两者刻意分开，改坏接线时不会被方法层的测试掩盖。

其中 `test_random_timing_is_not_significant_through_the_engine` 是核心统计主张在
引擎层的复核：持仓与价格独立生成 ⇒ 没有任何择时能力 ⇒ 必须判为不显著。
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from pandas import DataFrame, date_range

from vnpy_ctastrategy import permutation_test as pt
from vnpy_ctastrategy.backtesting import BacktestingEngine

CAPITAL = 1_000_000
ANNUAL_DAYS = 252
COST_PER_NOTIONAL = 0.0005


# ══════════════════════════════════════════════════════════════════════
# 测试用具
# ══════════════════════════════════════════════════════════════════════

def persistent_exposure(
    n: int, rng: np.random.Generator, stay: float = 0.95, units: float = 1000.0
) -> np.ndarray:
    """两状态马尔可夫持仓，平均持仓段长 1/(1−stay)，模仿海龟的长持仓。"""
    state = np.zeros(n, dtype=float)
    current = 0.0
    for t in range(n):
        if rng.random() > stay:
            current = units - current
        state[t] = current
    return state


def build_daily_df(close: np.ndarray, exposure: np.ndarray) -> DataFrame:
    """按 vnpy DailyResult 的列与口径造 daily_df（收盘价成交，trading_pnl=0）。"""
    close = np.asarray(close, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    pre_close = np.concatenate([[close[0]], close[:-1]])
    holding_pnl = exposure * (close - pre_close)

    changes = np.abs(np.diff(np.concatenate([[0.0], exposure])))
    turnover = changes * close
    commission = turnover * COST_PER_NOTIONAL

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
        },
        index=date_range("2024-01-02", periods=close.size, freq="B"),
    )


def random_timing_df(n: int, seed: int, drift: float = 0.0004) -> DataFrame:
    """持仓与价格【独立】生成：零假设成立的世界。drift 默认为正，制造牛市 beta。"""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.015, size=n)))
    return build_daily_df(close, persistent_exposure(n, rng))


def edged_df(n: int, seed: int, edge: float) -> DataFrame:
    """持仓真有预测力：在场那天的市场收益均值被抬高 edge。"""
    rng = np.random.default_rng(seed)
    exposure = persistent_exposure(n, rng)
    market = rng.normal(0.0, 0.015, size=n) + edge * (exposure > 0)
    return build_daily_df(100.0 * np.exp(np.cumsum(market)), exposure)


def make_engine(**settings: object) -> BacktestingEngine:
    engine = BacktestingEngine()
    engine.capital = CAPITAL
    engine.annual_days = ANNUAL_DAYS
    engine.size = 1
    engine.risk_free = 0
    if settings:
        engine.enable_permutation_test(**settings)
    return engine


# ══════════════════════════════════════════════════════════════════════
# 1. 字段契约（漂移守卫）
# ══════════════════════════════════════════════════════════════════════

def test_setting_keys_match_permutation_statistics_signature() -> None:
    """白名单与 permutation_statistics 的签名必须一一对应。

    这是漂移守卫：给 permutation_statistics 加了参数却忘了加进白名单，
    用户会收到"未知参数"的假报错；反过来则会在调用时炸 TypeError。
    """
    parameters = set(inspect.signature(pt.permutation_statistics).parameters)
    engine_owned = {"daily_df", "capital", "annual_days"}
    assert parameters - engine_owned == set(pt.PERMUTATION_SETTING_KEYS)


def test_field_defaults_cover_every_returned_key() -> None:
    """占位字典必须覆盖 permutation_statistics 的全部返回键。

    漏一个键 ⇒ 检验关闭时该键缺失、打开时突然出现，statistics dict 的形状不稳定，
    下游按键取值的代码会时灵时不灵。
    """
    returned = set(
        pt.permutation_statistics(
            random_timing_df(200, seed=1), CAPITAL, ANNUAL_DAYS, n_permutations=49
        )
    )
    assert returned <= set(pt.PERMUTATION_FIELD_DEFAULTS)
    # perm_error 只在失败时出现，故只存在于占位字典里。
    assert set(pt.PERMUTATION_FIELD_DEFAULTS) - returned == {"perm_error"}


def test_placeholder_defaults_mean_not_computed_not_significant() -> None:
    """占位值的语义必须是"没算"而不是"算了但不显著"，且不可误判为显著。"""
    assert pt.PERMUTATION_FIELD_DEFAULTS["perm_p_value"] == 1.0
    assert pt.PERMUTATION_FIELD_DEFAULTS["perm_significant"] is False
    assert pt.PERMUTATION_FIELD_DEFAULTS["perm_has_power"] is False
    assert pt.PERMUTATION_FIELD_DEFAULTS["perm_statistic"] == "not_computed"


# ══════════════════════════════════════════════════════════════════════
# 2. 开关语义
# ══════════════════════════════════════════════════════════════════════

def test_disabled_by_default_yields_neutral_placeholders() -> None:
    """默认关闭：键齐全、语义中性、不额外耗时。"""
    statistics = make_engine().calculate_statistics(
        random_timing_df(300, 2), output=False
    )

    for key in pt.PERMUTATION_FIELD_DEFAULTS:
        assert key in statistics, key
    assert statistics["perm_p_value"] == 1.0
    assert not statistics["perm_significant"]
    assert not statistics["perm_has_power"]
    assert statistics["perm_statistic"] == "not_computed"


def test_enable_then_disable_restores_placeholders() -> None:
    """关掉之后必须回到占位值，不能留着上一轮的 p 值继续骗人。"""
    engine = make_engine(n_permutations=99, seed=3)
    daily_df = edged_df(400, seed=4, edge=0.0025)
    assert engine.calculate_statistics(daily_df, output=False)["perm_has_power"]

    engine.enable_permutation_test(False)
    statistics = engine.calculate_statistics(daily_df, output=False)
    assert statistics["perm_p_value"] == 1.0
    assert statistics["perm_statistic"] == "not_computed"


def test_enable_permutation_test_rejects_unknown_settings() -> None:
    """未知参数当场报错，而不是被下游 fail-open 吞成一条 perm_error。"""
    engine = make_engine()
    with pytest.raises(ValueError, match="未知的重排检验参数"):
        engine.enable_permutation_test(n_permutation=100)  # 少了个 s
    assert not engine.permutation_enabled


@pytest.mark.parametrize("key", sorted(pt.PERMUTATION_SETTING_KEYS))
def test_every_whitelisted_setting_is_accepted(key: str) -> None:
    """白名单里的每个键都必须真的能传进去。"""
    make_engine().enable_permutation_test(**{key: None})


def test_module_defaults_are_never_mutated() -> None:
    """引擎必须拷贝占位字典；就地改会污染后续所有回测。"""
    snapshot = dict(pt.PERMUTATION_FIELD_DEFAULTS)
    engine = make_engine(n_permutations=99, seed=5)
    engine.calculate_statistics(edged_df(300, seed=6, edge=0.002), output=False)
    assert snapshot == pt.PERMUTATION_FIELD_DEFAULTS


# ══════════════════════════════════════════════════════════════════════
# 3. 核心统计主张在引擎层的复核
# ══════════════════════════════════════════════════════════════════════

def test_random_timing_is_not_significant_through_the_engine() -> None:
    """**核心检验**：随机择时经引擎跑出来必须判为不显著。

    持仓与价格独立生成，且刻意给市场加正漂移（牛市）—— 于是策略夏普为正、
    total_return 为正，一切样本内描述统计都好看，而真实择时能力为零。
    这正是本项目栽过的那个坑（样本外 -19.5%），检验必须能识破。

    取 12 条独立曲线的中位 p 值，而不是单挑一条断言 p>0.05：零假设成立时
    本来就有约 5% 的曲线会偶然显著，对单条断言等于在检验运气。
    """
    engine = make_engine(n_permutations=499)
    p_values = []
    sharpes = []
    for replication in range(12):
        statistics = engine.calculate_statistics(
            random_timing_df(600, seed=200 + replication), output=False
        )
        p_values.append(statistics["perm_p_value"])
        sharpes.append(statistics["sharpe_ratio"])

    # 前提确认：样本内描述统计确实"好看"，所以这不是个平凡的测试
    assert float(np.mean(sharpes)) > 0.0
    # 结论：择时无能力 ⇒ 不显著
    assert float(np.median(p_values)) > 0.15
    assert sum(p <= 0.05 for p in p_values) <= 3


def test_long_only_beta_alone_never_becomes_edge() -> None:
    """买入持有（永远满仓）在牛市里夏普很高，但择时增量恒为零 ⇒ 必须不显著。

    这是方案 C 的立身之本：market beta 同时出现在观测值与零分布中心里被约掉。
    暴露恒定时零分布退化，`has_power` 必须为 False —— 而不是吐出一个漂亮的 p 值。
    """
    rng = np.random.default_rng(77)
    n = 600
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0008, 0.015, size=n)))
    daily_df = build_daily_df(close, np.full(n, 1000.0))

    statistics = make_engine(n_permutations=499, seed=9).calculate_statistics(
        daily_df, output=False
    )
    assert statistics["sharpe_ratio"] > 0.5           # 样本内很好看
    assert not statistics["perm_significant"]          # 但没有择时证据
    assert not statistics["perm_has_power"]            # 且诚实承认没有功效


def test_real_edge_is_detected_through_the_engine() -> None:
    """有真 edge 时必须能检出，否则这个闸只会误杀。"""
    statistics = make_engine(n_permutations=999, seed=13).calculate_statistics(
        edged_df(600, seed=909, edge=0.0030), output=False
    )
    assert statistics["perm_has_power"]
    assert statistics["perm_significant"]
    assert statistics["perm_p_value"] < 0.05
    assert statistics["perm_z_score"] > 1.6


def test_engine_path_matches_direct_module_call() -> None:
    """引擎这条路径与直接调用模块必须逐位一致（种子固定）。"""
    daily_df = edged_df(400, seed=21, edge=0.002)
    engine = make_engine(n_permutations=299, seed=42)
    via_engine = engine.calculate_statistics(daily_df, output=False)
    direct = pt.permutation_statistics(
        daily_df,
        CAPITAL,
        ANNUAL_DAYS,
        n_permutations=299,
        seed=42,
        size=1,
        risk_free=0,
    )
    assert via_engine["perm_p_value"] == pytest.approx(direct["perm_p_value"])
    assert via_engine["perm_observed"] == pytest.approx(direct["perm_observed"])
    assert via_engine["perm_block_length"] == direct["perm_block_length"]


def test_block_length_tracks_holding_period_not_return_autocorrelation() -> None:
    """块长应落在持仓周期的量级上（stay=0.95 ⇒ 平均段长约 20 天），而不是收益的 L≈2。

    误在收益序列上估块长会把几周一次的持仓打成两天一换，零分布方差被严重低估。
    """
    statistics = make_engine(n_permutations=99, seed=8).calculate_statistics(
        random_timing_df(600, seed=31), output=False
    )
    assert statistics["perm_block_length"] >= 10


# ══════════════════════════════════════════════════════════════════════
# 4. 健壮性：诊断指标绝不许拖垮回测
# ══════════════════════════════════════════════════════════════════════

def test_missing_columns_fail_open_and_keep_the_backtest_alive() -> None:
    """daily_df 缺列时写 perm_error 并保留占位值，其余统计指标照常产出。"""
    daily_df = random_timing_df(300, seed=41).drop(columns=["start_pos"])
    statistics = make_engine(n_permutations=99, seed=11).calculate_statistics(
        daily_df, output=False
    )
    assert statistics["perm_error"]
    assert "start_pos" in str(statistics["perm_error"])
    assert statistics["perm_p_value"] == 1.0
    assert not statistics["perm_significant"]
    # 回测本体毫发无伤
    assert statistics["total_days"] == 300
    assert statistics["sharpe_ratio"] != 0


def test_statistics_survive_the_inf_nan_filter() -> None:
    """calculate_statistics 末尾那圈 np.nan_to_num 不能把 perm_* 字段搞坏。"""
    statistics = make_engine(n_permutations=99, seed=12).calculate_statistics(
        edged_df(300, seed=51, edge=0.002), output=False
    )
    assert str(statistics["perm_scheme"]) == "positions_block"
    assert str(statistics["perm_statistic"]) == "sharpe_ratio"
    assert 0.0 < float(statistics["perm_p_value"]) <= 1.0
    assert np.isfinite(float(statistics["perm_z_score"]))
    assert int(statistics["perm_n_permutations"]) == 99


def test_settings_override_engine_size_and_risk_free() -> None:
    """显式设置优先于引擎值，且不因重复传参炸 TypeError。"""
    engine = make_engine(n_permutations=99, seed=14, size=1, risk_free=0.0)
    statistics = engine.calculate_statistics(
        edged_df(300, seed=61, edge=0.002), output=False
    )
    assert not statistics["perm_error"]
    assert statistics["perm_has_power"]


def test_alternative_statistic_can_be_selected() -> None:
    """换统计量（路径依赖的收益回撤比）也能走通引擎这条路。"""
    engine = make_engine(
        n_permutations=199, seed=15, statistic="return_drawdown_ratio"
    )
    statistics = engine.calculate_statistics(
        edged_df(400, seed=71, edge=0.002), output=False
    )
    assert str(statistics["perm_statistic"]) == "return_drawdown_ratio"
    assert statistics["perm_has_power"]


def test_output_report_renders_without_error() -> None:
    """output=True 时的报告分支必须能跑（含退化与失败两条岔路）。"""
    engine = make_engine(n_permutations=99, seed=16)
    engine.calculate_statistics(edged_df(300, seed=81, edge=0.002), output=True)

    rng = np.random.default_rng(82)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, size=300)))
    engine.calculate_statistics(
        build_daily_df(close, np.full(300, 1000.0)), output=True
    )

    engine.calculate_statistics(
        random_timing_df(300, seed=83).drop(columns=["close_price"]), output=True
    )


def test_no_result_returns_empty_statistics() -> None:
    """没有回测结果时走早退分支，不该因为多了重排检验而变化。

    注意早退只在 df 省略【且】engine.daily_df 为空时发生；显式传一个空 DataFrame
    会在上游 `df["net_pnl"]` 处抛 KeyError（vnpy 原有行为，与本次接线无关，
    重排检验的代码在 positive_balance 分支里，那时根本轮不到它执行）。
    """
    engine = make_engine(n_permutations=99)
    assert engine.daily_df.empty
    assert engine.calculate_statistics(output=False) == {}


def test_permutation_code_is_unreachable_when_balance_is_wiped_out() -> None:
    """爆仓时 positive_balance=False，重排检验整段不执行，占位值原样留下。"""
    n = 200
    close = np.full(n, 100.0)
    exposure = np.full(n, 1000.0)
    daily_df = build_daily_df(close, exposure)
    daily_df["net_pnl"] = -2.0 * CAPITAL / n          # 稳稳打穿本金

    statistics = make_engine(n_permutations=99, seed=17).calculate_statistics(
        daily_df, output=False
    )
    assert statistics["perm_statistic"] == "not_computed"
    assert statistics["perm_p_value"] == 1.0
    assert not statistics["perm_has_power"]
