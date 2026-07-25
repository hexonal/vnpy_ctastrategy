"""Sharpe 推断的单位契约、Monte-Carlo 有效性与 611 日样本门槛。

本文件补 test_sharpe_inference.py / _extra.py 漏掉的三块：

1. **单位契约（回归测试）**。上游 ``sharpe_ratio`` 用的是百分数刻度
   （``daily_return = df["return"].mean() * 100``），因此它的 ``daily_risk_free
   = risk_free / √annual_days`` 也是百分数刻度；而我们把**分数刻度**的
   ``df["return"]`` 喂给 ``sharpe_inference``，同一个利率必须再除以 100。
   原实现漏了这个 /100，risk_free=2 时引擎报 −3.33 而推断报 −181.88。
   既有测试全部用默认 risk_free=0，所以整条单位链一次都没被走到——
   这里用非零 risk_free 把它钉死。

2. **方法有效性的独立复核**。既有 size test 只跑 200-300 次重复；这里跑 2000 次，
   用二项分布精确区间判定，确认"纯随机序列被判显著"的比例真的收敛到名义 5%。
   这是检验方法本身是否有效的关键测试：一个永远说"显著"的检验毫无用处，
   一个永远说"不显著"的检验同样毫无用处。

3. **611 日样本的门槛与功效**。把"要多大 Sharpe 才显著"从闭式解和模拟两条路
   各算一遍并互相验证，防止公式推导错了却没人发现。

跑法::

    /Users/flink/tradingview/vnpy/.venv/bin/python -m pytest \
        tests/test_sharpe_inference_units.py -q
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from vnpy_ctastrategy.sharpe_inference import (
    inference_from_daily_df,
    required_sharpe,
    sharpe_inference,
)

# 本项目实测样本：700.SEHK 2024-01~2026-07 = 611 个交易日
PROJECT_N: int = 611
ANNUAL_DAYS: int = 240


def _daily_df(net_pnl: np.ndarray):
    """构造 calculate_statistics 需要的最小 daily_df。"""
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


def _engine(risk_free: float):
    from vnpy_ctastrategy.backtesting import BacktestingEngine

    engine = BacktestingEngine()
    engine.capital = 1_000_000
    engine.annual_days = ANNUAL_DAYS
    engine.risk_free = risk_free
    return engine


# 一、单位契约：非零 risk_free 下，推断的 Sharpe 必须等于面板上的 sharpe_ratio


@pytest.mark.parametrize("risk_free", [0.0, 2.0, 4.5])
def test_risk_free_matches_upstream(risk_free: float) -> None:
    """引擎里不能有第二套 Sharpe —— 非零 risk_free 才能走到单位换算这条路。"""
    rng = np.random.default_rng(9091)
    df = _daily_df(rng.normal(400.0, 11_000.0, PROJECT_N))

    stats = _engine(risk_free).calculate_statistics(df=df, output=False)

    # sharpe_tstat = sharpe_annual / se_annual，两边同乘 se 即还原 sharpe_annual。
    # 单位错位时 tstat 会跟着 sharpe_annual 一起偏，故直接比 sharpe_ratio 自身。
    assert stats["sharpe_tstat"] == pytest.approx(
        stats["sharpe_ratio"] / stats["sharpe_se"], rel=1e-9
    )


@pytest.mark.parametrize("risk_free", [2.0, 4.5])
def test_a_nonzero_risk_free_does_not_blow_up_the_sharpe(risk_free: float) -> None:
    """漏掉 /100 的症状：Sharpe 被推到几百个单位。这里给出量级上界。"""
    rng = np.random.default_rng(9092)
    df = _daily_df(rng.normal(400.0, 11_000.0, PROJECT_N))

    stats = _engine(risk_free).calculate_statistics(df=df, output=False)

    # 真实值在个位数；漏 /100 时是 −180 量级。10 是宽松但足以分辨的界。
    assert abs(stats["sharpe_ratio"]) < 10.0
    assert abs(stats["sharpe_ratio"] / stats["sharpe_se"]) < 60.0


def test_inference_from_daily_df_uses_the_engine_risk_free_convention() -> None:
    """便捷入口与引擎必须同一套单位约定，否则两条路径给出不同的 Sharpe。"""
    rng = np.random.default_rng(9093)
    df = _daily_df(rng.normal(400.0, 11_000.0, PROJECT_N))
    risk_free = 3.0

    stats = _engine(risk_free).calculate_statistics(df=df, output=False)
    direct = inference_from_daily_df(
        df, annual_days=ANNUAL_DAYS, risk_free=risk_free, n_boot=0
    )

    assert direct.sharpe_annual == pytest.approx(stats["sharpe_ratio"], rel=1e-9)


def test_risk_free_shifts_the_sharpe_by_exactly_the_hurdle() -> None:
    """解析核对：``risk_free_period`` 收的是**单期**利率，年化 Sharpe 正好降

        Δ = rf_period × √annual_days / σ_period

    换句话说，一个真年化 4% 的门槛，单期值是 0.04 / annual_days（不是 /√annual_days）。
    """
    rng = np.random.default_rng(9094)
    returns = rng.normal(0.0007, 0.011, PROJECT_N)
    sigma_period = float(returns.std(ddof=1))

    base = sharpe_inference(returns, annual_days=ANNUAL_DAYS, n_boot=0)

    # 真年化 4%：单期 = 0.04 / annual_days
    hurdle_period = 0.04 / ANNUAL_DAYS
    bumped = sharpe_inference(
        returns, annual_days=ANNUAL_DAYS, risk_free_period=hurdle_period, n_boot=0
    )

    expected_drop = hurdle_period * math.sqrt(ANNUAL_DAYS) / sigma_period
    assert base.sharpe_annual - bumped.sharpe_annual == pytest.approx(
        expected_drop, rel=1e-9
    )
    # 等价形式：年化超额 / 年化波动率
    annual_vol = sigma_period * math.sqrt(ANNUAL_DAYS)
    assert expected_drop == pytest.approx(0.04 / annual_vol, rel=1e-9)


def test_engine_risk_free_convention_is_reproduced_exactly() -> None:
    """把上游那条公式逐字重算一遍，确认接线层的 /100 正是它要求的。

    上游：``daily_return`` 与 ``return_std`` 都是 ``df["return"] * 100``，
    而 ``daily_risk_free = risk_free / √annual_days`` 直接与之相减。
    我们喂的是分数刻度的 ``df["return"]``，故同一利率必须再 /100。
    （``/√annual_days`` 这一步是上游自己的口径 —— 它不等于"年化利率/年天数"，
    但面板上的 sharpe_ratio 就是这么算的，推断必须与面板一致，不在这里改口径。）
    """
    rng = np.random.default_rng(9095)
    returns = rng.normal(0.0005, 0.010, PROJECT_N)
    risk_free = 2.0

    daily_return = returns.mean() * 100
    return_std = returns.std(ddof=1) * 100
    daily_risk_free = risk_free / math.sqrt(ANNUAL_DAYS)
    upstream = (daily_return - daily_risk_free) / return_std * math.sqrt(ANNUAL_DAYS)

    ours = sharpe_inference(
        returns,
        annual_days=ANNUAL_DAYS,
        risk_free_period=daily_risk_free / 100.0,
        n_boot=0,
    )
    assert ours.sharpe_annual == pytest.approx(upstream, rel=1e-12)

    # 漏掉 /100 会差两个数量级 —— 这就是被修掉的那个 bug
    wrong = sharpe_inference(
        returns, annual_days=ANNUAL_DAYS, risk_free_period=daily_risk_free, n_boot=0
    )
    assert abs(wrong.sharpe_annual - upstream) > 100.0


# 二、方法有效性：纯随机序列必须被判"不显著"，且拒绝率收敛到名义 5%


def _rejection_rate(
    method: str,
    n_reps: int,
    n: int,
    generator,
    seed: int,
    alpha: float = 0.05,
) -> float:
    rng = np.random.default_rng(seed)
    rejects = 0
    for _ in range(n_reps):
        result = sharpe_inference(
            generator(rng, n), annual_days=ANNUAL_DAYS, method=method, n_boot=0
        )
        rejects += int(result.p_value < alpha)
    return rejects / n_reps


def _white_noise(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.normal(0.0, 0.012, n)


def _student_t(rng: np.random.Generator, n: int) -> np.ndarray:
    """自由度 4 的厚尾：峰度无穷大以外最接近真实日收益的玩具分布。"""
    return rng.standard_t(4, n) * 0.006


def _ar1_noise(rng: np.random.Generator, n: int, rho: float = 0.25) -> np.ndarray:
    """AR(1) 噪声：真实 SR 仍为 0，但 iid 公式会低估 SE。"""
    burn = 200
    eps = rng.normal(0.0, 0.012, n + burn)
    out = np.empty(n + burn)
    out[0] = eps[0]
    for t in range(1, n + burn):
        out[t] = rho * out[t - 1] + eps[t]
    return out[burn:]


@pytest.mark.parametrize("seed", [11, 22, 33, 44, 55])
def test_a_random_series_is_reported_as_not_significant(seed: int) -> None:
    """最直白的一条：纯随机序列，检验必须说"不显著"。多个种子降低偶然性。"""
    rng = np.random.default_rng(seed)
    result = sharpe_inference(_white_noise(rng, PROJECT_N), annual_days=ANNUAL_DAYS)

    assert not result.significant, result.summary_line()
    assert result.p_value > 0.05
    # "不显著"与"置信区间覆盖 0"是同一件事的两种说法
    assert result.ci_low_annual < 0.0 < result.ci_high_annual
    # bootstrap 第二意见必须同向
    assert result.bootstrap_p_value > 0.05


def test_size_converges_to_five_percent_on_iid_normal() -> None:
    """2000 次重复的 Monte-Carlo size test：真实 SR=0 时拒绝率必须 ≈ 5%。

    这是检验"方法本身是否有效"的核心：
      拒绝率远高于 5% → 检验是橡皮图章，会把运气盖章成技巧；
      拒绝率远低于 5% → 检验过度保守，真信号也发现不了。
    n=2000、p=0.05 时的二项标准差 ≈ 0.49pp，故 [3.5%, 6.5%] 是约 ±3σ 的区间。
    """
    rate = _rejection_rate("hac", n_reps=2000, n=PROJECT_N, generator=_white_noise, seed=4242)
    assert 0.035 < rate < 0.065, f"HAC 的经验 size = {rate:.4f}，偏离名义 5%"


def test_size_holds_under_fat_tails() -> None:
    """厚尾下 size 仍需受控 —— 交易策略的日收益从来不是正态的。"""
    rate = _rejection_rate("hac", n_reps=1000, n=PROJECT_N, generator=_student_t, seed=4343)
    assert 0.03 < rate < 0.075, f"厚尾下的经验 size = {rate:.4f}"


def test_hac_fixes_the_over_rejection_that_iid_suffers_under_autocorrelation() -> None:
    """AR(1) 噪声：iid 公式过度拒绝，HAC 必须把 size 拉回来。

    这条是"要不要用修正"的实证依据 —— CTA 持仓跨多日，日 P&L 天然自相关。
    """
    iid_rate = _rejection_rate(
        "iid_normal", n_reps=600, n=PROJECT_N, generator=_ar1_noise, seed=4444
    )
    hac_rate = _rejection_rate(
        "hac", n_reps=600, n=PROJECT_N, generator=_ar1_noise, seed=4444
    )

    assert iid_rate > 0.08, f"iid 在 AR(1) 下没有过度拒绝（{iid_rate:.4f}），测试前提失效"
    assert hac_rate < iid_rate, f"HAC 没有改善 size：{iid_rate:.4f} → {hac_rate:.4f}"
    assert hac_rate < 0.09, f"HAC 修正后仍过度拒绝：{hac_rate:.4f}"


def test_a_real_edge_is_still_detected() -> None:
    """反向对照：检验不能是"永远说不显著"的稻草人。"""
    rng = np.random.default_rng(4545)
    # 年化 SR ≈ 2.0，远在 611 日门槛之上
    returns = rng.normal(2.0 / math.sqrt(ANNUAL_DAYS) * 0.012, 0.012, PROJECT_N)
    result = sharpe_inference(returns, annual_days=ANNUAL_DAYS)

    assert result.significant, result.summary_line()
    assert result.p_value < 0.01
    assert result.ci_low_annual > 0.0


# 三、611 日样本：要多大 Sharpe 才显著


def test_required_sharpe_for_611_days_is_about_one_point_two() -> None:
    """本项目样本量下的门槛数字，写死防回归。"""
    two_sided = required_sharpe(PROJECT_N, annual_days=ANNUAL_DAYS, alpha=0.05)
    one_sided = required_sharpe(
        PROJECT_N, annual_days=ANNUAL_DAYS, alpha=0.05, one_sided=True
    )

    assert two_sided == pytest.approx(1.233, abs=0.01)
    assert one_sided == pytest.approx(1.034, abs=0.01)
    assert one_sided < two_sided


def test_required_sharpe_rises_when_hac_inflates_the_standard_error() -> None:
    """自相关越强，门槛越高 —— 门槛不是常数，随曲线性质变。"""
    plain = required_sharpe(PROJECT_N, annual_days=ANNUAL_DAYS, se_inflation=1.0)
    inflated = required_sharpe(PROJECT_N, annual_days=ANNUAL_DAYS, se_inflation=1.3)

    assert inflated > plain
    # SE 放大 f 倍，门槛近似同比放大 f 倍（高阶矩项只带来微小偏离）
    assert inflated == pytest.approx(plain * 1.3, rel=0.02)


def test_required_sharpe_is_the_actual_break_even_of_the_test() -> None:
    """闭式解与检验实际行为必须一致：恰在门槛上的曲线，p 应正好压在 0.05。

    构造一条样本矩完全受控的序列（去均值后再注入精确漂移），
    使样本 Sharpe 精确等于门槛值，然后看检验怎么判。
    """
    threshold_annual = required_sharpe(PROJECT_N, annual_days=ANNUAL_DAYS, alpha=0.05)
    threshold_period = threshold_annual / math.sqrt(ANNUAL_DAYS)

    rng = np.random.default_rng(4646)
    raw = rng.normal(0.0, 0.012, PROJECT_N)
    centered = raw - raw.mean()
    # 令样本均值 / 样本标准差恰好等于 threshold_period
    returns = centered + threshold_period * centered.std(ddof=1)

    result = sharpe_inference(
        returns, annual_days=ANNUAL_DAYS, method="iid_nonnormal", n_boot=0
    )
    assert result.sharpe_annual == pytest.approx(threshold_annual, rel=1e-6)
    assert result.p_value == pytest.approx(0.05, abs=0.004)


def test_power_at_the_threshold_is_only_about_half() -> None:
    """期望管理：真实 SR 恰好等于门槛时，只有约一半的样本能测出显著。

    直接后果是 —— 真实 SR 略高于 1.23 的策略，611 日里有接近一半的概率
    被自己的显著性检验判为"不显著"。门槛不是"低于它就没用"，
    而是"低于它这段样本无法区分它和运气"。
    """
    threshold_annual = required_sharpe(PROJECT_N, annual_days=ANNUAL_DAYS, alpha=0.05)
    drift = threshold_annual / math.sqrt(ANNUAL_DAYS) * 0.012

    rng = np.random.default_rng(4747)
    rejects = 0
    n_reps = 600
    for _ in range(n_reps):
        returns = rng.normal(drift, 0.012, PROJECT_N)
        rejects += int(
            sharpe_inference(
                returns, annual_days=ANNUAL_DAYS, method="iid_nonnormal", n_boot=0
            ).p_value
            < 0.05
        )
    power = rejects / n_reps

    assert 0.40 < power < 0.60, f"门槛处的功效 = {power:.3f}，应约 50%"


def test_the_reported_134_day_case_is_not_significant() -> None:
    """痛点复现：134 日曲线 sharpe = −1.68、SE ≈ 1.36，连符号都没到 2σ。"""
    n = 134
    rng = np.random.default_rng(4848)
    raw = rng.normal(0.0, 0.015, n)
    centered = raw - raw.mean()
    target_period = -1.68 / math.sqrt(ANNUAL_DAYS)
    returns = centered + target_period * centered.std(ddof=1)

    result = sharpe_inference(returns, annual_days=ANNUAL_DAYS, method="iid_normal", n_boot=0)

    assert result.sharpe_annual == pytest.approx(-1.68, rel=1e-6)
    # SE 的量级必须落在 1.3~1.45（Lo iid 正态公式在 n=134 下的取值）
    assert 1.30 < result.standard_error_annual < 1.45
    assert not result.significant
    assert result.ci_low_annual < 0.0 < result.ci_high_annual
    # 短样本必须自带告警
    assert any("不足一年" in w for w in result.warnings)


def test_shorter_samples_need_larger_sharpe() -> None:
    """样本越短门槛越高 —— 单调性是这条公式最基本的自洽要求。"""
    thresholds = [
        required_sharpe(n, annual_days=ANNUAL_DAYS, alpha=0.05)
        for n in (134, 250, PROJECT_N, 1200, 2400)
    ]
    assert thresholds == sorted(thresholds, reverse=True)
    # 134 日的门槛必须高到离谱，这正是痛点里那条曲线无法判定的原因
    assert thresholds[0] > 2.5
