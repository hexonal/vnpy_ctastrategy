"""`overfitting_audit` 的测试。

分三类，缺任何一类这个模块都不该上线：

  A. 单元正确性 —— 回合计数 / 持仓天数 / 可行性规划 / 有效效率，
     每一条都用**手工构造、答案可心算**的数据钉死。
  B. 方法有效性（最关键）—— 对**纯随机序列**，整条流水线必须给出
     "不显著 / 不可采信"的结论。一个连随机数都判成"稳健"的方法，
     它对真策略的任何"通过"结论都是零信息量。这里同时检验：
        · 随机数据 → 决不给 GO
        · 假阳性率 ≈ 名义 alpha（不是"碰巧没通过"，而是错误率受控）
        · 真实优势 → 能给 GO（否则就是个永远说 NO 的废物闸门）
  C. 本项目实测复现 —— 把 700.SEHK 那次真实运行暴露的"零交易折"病理
     固化成回归测试，防止有人日后把它"优化"掉。

全部测试**不连数据库**：用可控的假 runner 生成 daily_df，
所以在 CI / 无 QuestDB 的机器上照样能跑。
"""

from __future__ import annotations

import math
import sys
import zlib
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from pandas import DataFrame

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vnpy_ctastrategy.overfitting_audit import (  # noqa: E402
    GO,
    NO_GO,
    UNDECIDABLE,
    AuditResult,
    count_round_trips,
    effective_efficiency,
    fold_activity,
    format_audit,
    mean_holding_days,
    plan_walk_forward,
    run_audit,
)

ANNUAL_DAYS = 252
CAPITAL = 1_000_000.0


# ══════════════════════════════════════════════════════════════════════
# 测试脚手架：一个不碰数据库的假 runner
# ══════════════════════════════════════════════════════════════════════

def make_dates(n: int, start: datetime = datetime(2024, 1, 1)) -> list[datetime]:
    """n 个连续"交易日"（工作日近似，跳过周末，够用即可）。"""
    out: list[datetime] = []
    cur = start
    while len(out) < n:
        if cur.weekday() < 5:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def make_daily_df(dates, pnl, pos=None, prev_pos: float = 0.0) -> DataFrame:
    """构造一个具备 audit 所需列的 daily_df（date 索引 / net_pnl / trade_count / pos）。

    `prev_pos` 是窗口第一天的期初仓位。切窗口时必须传真值，否则每个窗口都从"空仓"
    起步，`count_round_trips` 会把一个跨窗口的持仓误判成新开仓。
    """
    pnl = np.asarray(pnl, dtype=float)
    if pos is None:
        pos = np.zeros(len(dates), dtype=float)
    pos = np.asarray(pos, dtype=float)

    start_pos = np.empty_like(pos)
    start_pos[0] = float(prev_pos)
    start_pos[1:] = pos[:-1]
    fills = (start_pos != pos).astype(int)

    return DataFrame(
        {
            "net_pnl": pnl,
            "trade_count": fills,
            "start_pos": start_pos,
            "end_pos": pos,
        },
        index=[d.date() for d in dates],
    )


class FakeRunner:
    """按 (参数, 窗口) 返回 daily_df 的假执行器 —— 行为必须与真回测引擎同构。

    关键约束(**踩过坑**)：**同一组参数在同一个日期上的盈亏必须唯一**。
    第一版实现给每个窗口重新播种 RNG，于是"第 0~62 天"这段序列在训练窗和
    测试窗里被生成了两次且完全相同 —— 样本外变成了样本内的副本，
    纯随机数据的假阳性率被推到 37.5%。那不是方法失效，是**测试脚手架自己造了前视泄漏**。
    现在改成：每组参数在**全区间**上一次性生成一条序列并缓存，窗口只做切片。

    RNG 播种用 crc32 而不是内置 `hash()`：`hash()` 对字符串带进程级随机盐
    （PYTHONHASHSEED），会让测试在不同进程里给出不同结果。
    """

    capital = CAPITAL

    def __init__(self, dates, pnl_fn, pos_fn=None, seed: int = 7):
        self._dates = list(dates)
        self._pnl_fn = pnl_fn
        self._pos_fn = pos_fn
        self._seed = seed
        self._cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def bar_datetimes(self):
        return list(self._dates)

    def _series(self, setting: dict) -> tuple[np.ndarray, np.ndarray]:
        key = tuple(sorted(setting.items()))
        if key not in self._cache:
            seed = self._seed + zlib.crc32(repr(key).encode("utf-8"))
            rng = np.random.default_rng(seed)
            n = len(self._dates)
            pnl = np.array(
                [self._pnl_fn(setting, i, rng) for i in range(n)], dtype=float
            )
            pos = (
                np.array([self._pos_fn(setting, i) for i in range(n)], dtype=float)
                if self._pos_fn is not None
                else np.zeros(n, dtype=float)
            )
            self._cache[key] = (pnl, pos)
        return self._cache[key]

    def __call__(self, setting: dict, start, end) -> DataFrame:
        lo = start.date() if hasattr(start, "date") else start
        hi = end.date() if hasattr(end, "date") else end
        idx = [i for i, d in enumerate(self._dates) if lo <= d.date() <= hi]
        if not idx:
            return DataFrame()

        pnl, pos = self._series(setting)
        first = idx[0]
        prev_pos = float(pos[first - 1]) if first > 0 else 0.0
        window = [self._dates[i] for i in idx]
        return make_daily_df(window, pnl[idx], pos[idx], prev_pos=prev_pos)


def simple_statistics(df: DataFrame) -> dict:
    """最小可用的 statistics 面板，口径与 vnpy 的 sharpe_ratio / annual_return 一致。"""
    if df is None or df.empty:
        return {}
    balance = df["net_pnl"].to_numpy(dtype=float).cumsum() + CAPITAL
    pre = np.empty_like(balance)
    pre[0] = CAPITAL
    pre[1:] = balance[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = balance / pre
        x = np.where(x <= 0, np.nan, x)
        ret = np.nan_to_num(np.log(x), nan=0.0, posinf=0.0, neginf=0.0)
    std = float(np.std(ret, ddof=1)) if ret.size > 1 else 0.0
    sharpe = float(np.mean(ret) / std * math.sqrt(ANNUAL_DAYS)) if std > 0 else 0.0
    total_return = float(balance[-1] / CAPITAL - 1.0) * 100.0
    annual = total_return / max(1, len(df)) * ANNUAL_DAYS
    return {
        "sharpe_ratio": sharpe,
        "annual_return": annual,
        "total_return": total_return,
        "total_trade_count": int(df["trade_count"].sum()),
        "max_ddpercent": 0.0,
    }


def grid(n: int = 12) -> list[dict]:
    return [{"p": i} for i in range(n)]


# ══════════════════════════════════════════════════════════════════════
# A. 单元正确性
# ══════════════════════════════════════════════════════════════════════

def test_count_round_trips_counts_position_cycles_not_fills() -> None:
    """金字塔加仓：4 次买入 + 1 次卖出 = 1 个回合，不是 5/2=2.5 个。

    这正是本模块存在的理由之一 —— fills/2 会把海龟的样本量凭空放大一倍多。
    """
    dates = make_dates(10)
    pos = [0, 100, 200, 300, 400, 400, 400, 0, 0, 0]
    df = make_daily_df(dates, np.zeros(10), pos)

    opened, closed = count_round_trips(df)
    assert (opened, closed) == (1, 1)
    assert int(df["trade_count"].sum()) == 5      # 4 买 + 1 卖
    assert closed * 2 < int(df["trade_count"].sum())   # fills/2 会高估


def test_count_round_trips_handles_two_cycles_and_open_tail() -> None:
    dates = make_dates(12)
    pos = [0, 100, 0, 0, 100, 100, 0, 0, 0, 100, 100, 100]
    df = make_daily_df(dates, np.zeros(12), pos)
    opened, closed = count_round_trips(df)
    assert opened == 3          # 三次开仓
    assert closed == 2          # 末尾那次还没平


def test_count_round_trips_on_empty_or_missing_columns() -> None:
    assert count_round_trips(DataFrame()) == (0, 0)
    assert count_round_trips(DataFrame({"net_pnl": [1.0, 2.0]})) == (0, 0)


def test_mean_holding_days() -> None:
    dates = make_dates(10)
    pos = [0, 100, 100, 100, 0, 0, 100, 100, 0, 0]   # 持仓 3 天 + 2 天，2 个回合
    df = make_daily_df(dates, np.zeros(10), pos)
    assert mean_holding_days(df) == pytest.approx(2.5)
    assert math.isnan(mean_holding_days(DataFrame()))


def test_mean_holding_days_is_nan_without_closed_trip() -> None:
    dates = make_dates(5)
    df = make_daily_df(dates, np.zeros(5), [0, 100, 100, 100, 100])
    assert math.isnan(mean_holding_days(df))


def test_plan_walk_forward_reproduces_the_700_sehk_impasse() -> None:
    """本项目实测：627 根日线 / 约 10 个完整回合 → 折数与每折回合数不可兼得。"""
    plan = plan_walk_forward(n_bars=627, round_trips_total=10, train_bars=252, test_bars=63)

    assert plan.n_folds == 5                              # 与实测的 5 折一致
    assert plan.bars_per_round_trip == pytest.approx(62.7)
    assert plan.expected_round_trips_per_fold < 2         # 每折期望约 1 个回合
    assert plan.per_fold_efficiency_usable is False       # 逐折效率不可报
    assert any("不可兼得" in n for n in plan.notes)


def test_plan_walk_forward_accepts_a_high_frequency_strategy() -> None:
    """同样 627 根样本，若策略每 5 天一个回合，逐折效率就是可报的。"""
    plan = plan_walk_forward(n_bars=627, round_trips_total=125, train_bars=252, test_bars=63)
    assert plan.n_folds == 5
    assert plan.expected_round_trips_per_fold > 5
    assert plan.per_fold_efficiency_usable is True
    assert plan.feasible is True


def test_plan_walk_forward_flags_too_few_folds() -> None:
    plan = plan_walk_forward(n_bars=400, round_trips_total=200, train_bars=252, test_bars=126)
    assert plan.n_folds == 1
    assert plan.per_fold_efficiency_usable is False
    assert any("只切得出" in n for n in plan.notes)


def test_plan_walk_forward_handles_zero_trades() -> None:
    plan = plan_walk_forward(n_bars=627, round_trips_total=0)
    assert plan.expected_round_trips_per_fold == 0.0
    assert plan.feasible is False
    assert any("从未开仓" in n for n in plan.notes)


def test_effective_efficiency_excludes_untraded_folds() -> None:
    """三个零交易折（效率 0）+ 两个真实折（0.9 / 1.1）。

    含零交易折的中位数是 0.0（"严重衰减"），剔除后是 1.0（"没有衰减"）。
    同一批数据两个相反结论 —— 这就是本模块要修的那个采样假象。
    """
    from vnpy_ctastrategy.overfitting import WalkForwardFold, WalkForwardSplit

    dates = make_dates(20)
    folds = []
    specs = [(0.0, False), (0.9, True), (0.0, False), (1.1, True), (0.0, False)]
    for i, (eff, traded) in enumerate(specs):
        pos = [0, 100, 100, 0] * 5 if traded else [0] * 20
        df = make_daily_df(dates, np.zeros(20), pos)
        split = WalkForwardSplit(i, dates[0], dates[9], dates[10], dates[19], 10, 10)
        folds.append(
            WalkForwardFold(
                split=split, chosen_setting={"p": i}, chosen_index=i,
                is_target=1.0, oos_target=1.0, is_annual_return=10.0,
                oos_annual_return=10.0 * eff, efficiency=eff, n_candidates=5,
                oos_daily_df=df, oos_statistics={},
            )
        )

    activity = fold_activity(folds)
    assert [a.traded for a in activity] == [False, True, False, True, False]

    naive_median = float(np.median([a.efficiency for a in activity]))
    eff_med, n_traded = effective_efficiency(activity)
    assert naive_median == pytest.approx(0.0)     # 含零交易折 → "严重衰减"
    assert n_traded == 2
    assert eff_med == pytest.approx(1.0)          # 剔除后 → 没有衰减


def test_effective_efficiency_is_nan_when_nothing_traded() -> None:
    from vnpy_ctastrategy.overfitting import WalkForwardFold, WalkForwardSplit

    dates = make_dates(10)
    df = make_daily_df(dates, np.zeros(10), [0] * 10)
    split = WalkForwardSplit(0, dates[0], dates[4], dates[5], dates[9], 5, 5)
    fold = WalkForwardFold(
        split=split, chosen_setting={}, chosen_index=0, is_target=0.0, oos_target=0.0,
        is_annual_return=0.0, oos_annual_return=0.0, efficiency=0.0, n_candidates=1,
        oos_daily_df=df, oos_statistics={},
    )
    eff, n = effective_efficiency(fold_activity([fold]))
    assert n == 0 and math.isnan(eff)


# ══════════════════════════════════════════════════════════════════════
# B. 方法有效性 —— 这一节是判断本模块可不可信的唯一依据
# ══════════════════════════════════════════════════════════════════════

def _noise_runner(dates, seed: int = 11) -> FakeRunner:
    """纯随机：每组参数的日盈亏都是同分布白噪声，**不存在任何真实优劣**。

    仓位设成周期性进出，保证每折都有交易 —— 这样"不通过"只能来自统计检验本身，
    而不是被"零交易折"这条捷径挡下来的。
    """
    def pnl(setting, i, rng):
        return float(rng.normal(0.0, 0.01) * CAPITAL)

    def pos(setting, i):
        return 100.0 if (i // 5) % 2 == 0 else 0.0

    return FakeRunner(dates, pnl, pos, seed=seed)


def test_random_data_is_never_certified() -> None:
    """核心有效性检验：纯随机序列，整条流水线不得给出 GO。

    随机数据里不存在任何可发现的规律，因此正确的方法必须拒绝背书。
    这条测试失败 = 本模块的所有"通过"结论都不可信。
    """
    dates = make_dates(627)
    runner = _noise_runner(dates)

    audit = run_audit(
        runner, grid(12), dates[0], dates[-1],
        baseline_setting={"p": 0},
        train_bars=252, test_bars=63,
        statistics_func=simple_statistics, capital=CAPITAL,
        n_null_sims=60, n_offsets=3, min_block_obs=40,
    )

    assert audit.decision != GO
    assert audit.decision in (NO_GO, UNDECIDABLE)
    # 样本外 Sharpe 必须够不上显著
    assert audit.wf.significance.p_block_bootstrap > 0.05
    # PBO 不得被判为「显著优于零分布」
    assert audit.pbo is not None
    assert not any("显著优于零分布" in r for r in audit.reasons)


def test_random_data_false_positive_rate_is_controlled() -> None:
    """比"随机数据不通过"更强的要求：**错误率必须接近名义 alpha**。

    仅有单次不通过可能是运气。这里跑 40 组独立随机数据，
    统计"样本外 Sharpe 被判显著"的比例，它应当落在 5% 附近而不是失控。
    只做 Walk-Forward 部分（PBO 零分布模拟太慢，其自身的假阳性率由
    test_overfitting.py::test_null_pvalue_does_not_flag_noise_as_robust 覆盖）。
    """
    dates = make_dates(627)
    n_trials = 40
    false_positives = 0

    for trial in range(n_trials):
        runner = _noise_runner(dates, seed=1000 + trial * 37)
        audit = run_audit(
            runner, grid(8), dates[0], dates[-1],
            train_bars=252, test_bars=63,
            statistics_func=simple_statistics, capital=CAPITAL,
            with_pbo=False, seed=500 + trial,
        )
        sig = audit.wf.significance
        if sig.sharpe > 0 and sig.p_block_bootstrap < 0.05:
            false_positives += 1

    rate = false_positives / n_trials
    # 单侧名义水平 2.5%；40 次试验下允许到 15%，超出说明检验被系统性放宽了
    assert rate <= 0.15, f"假阳性率 {rate:.2%} 远高于名义水平"


def test_a_genuine_edge_is_certified() -> None:
    """反向检验：存在真实且稳定的优势时必须给 GO。

    没有这条，「永远输出 NO-GO」也能通过上面所有测试 —— 那是个废物闸门。

    构造要点（第一版在这里翻过车，记下来）：
      * **必须有共同市场因子**。所有参数组交易的是同一个标的，日收益天然高度相关
        （本项目实测参数组间平均相关 0.853）。第一版让每组参数拿完全独立的噪声，
        相邻配置的排名于是纯靠运气翻转，即便存在真实优势 PBO 也上到 0.58 ——
        那不是方法失灵，是**构造出来的数据比现实难得多**。
      * **漂移不能太大**。盈亏是固定金额、名义本金不复利，漂移一大净值就快速上升，
        对数收益被越来越大的分母压扁 → 序列非平稳 → Sharpe 排名随时间段漂移。
        实测漂移取到 0.0012/日 时，全样本 Sharpe 在 p=8 处见顶后回落，不再单调。
    """
    dates = make_dates(627)
    common = np.random.default_rng(101).normal(0.0, 0.010, len(dates))

    def pnl(setting, i, rng):
        edge = 0.00012 * (setting["p"] + 1)      # 真实且持续的漂移，与参数单调相关
        return float((edge + common[i] + rng.normal(0.0, 0.002)) * CAPITAL)

    def pos(setting, i):
        return 100.0 if (i // 5) % 2 == 0 else 0.0

    runner = FakeRunner(dates, pnl, pos, seed=3)
    audit = run_audit(
        runner, grid(20), dates[0], dates[-1],
        baseline_setting={"p": 19},
        train_bars=252, test_bars=63,
        statistics_func=simple_statistics, capital=CAPITAL,
        n_null_sims=200, n_offsets=3, min_block_obs=40,
    )

    assert audit.decision == GO, audit.blockers
    assert audit.wf.significance.sharpe > 0
    assert audit.wf.significance.p_block_bootstrap < 0.05
    assert audit.pbo is not None
    assert audit.pbo.result.pbo <= 0.10
    assert audit.pbo.result.rank_ic > 0.5          # 样本内排名确实预测样本外排名


def test_overfit_data_is_rejected_as_no_go() -> None:
    """教科书式过拟合：样本内表现完全由噪声决定，样本外无任何延续。

    这里让"样本内最优"这件事在样本外系统性反转（后半段收益取反），
    审查必须给 NO-GO 而不是"无法裁决" —— 因为存在正面反证。
    """
    dates = make_dates(627)
    half = len(dates) // 2

    def pnl(setting, i, rng):
        # 参数越大，前半段漂移越强；后半段这份优势精确反号。
        # 于是「样本内最优」在样本外系统性变成最差 —— 教科书式的过拟合指纹。
        tilt = 0.0025 * (setting["p"] - 5.5) / 5.5
        sign = 1.0 if i < half else -1.0
        return float((sign * tilt + rng.normal(0.0, 0.006)) * CAPITAL)

    def pos(setting, i):
        return 100.0 if (i // 5) % 2 == 0 else 0.0

    runner = FakeRunner(dates, pnl, pos, seed=5)
    audit = run_audit(
        runner, grid(12), dates[0], dates[-1],
        baseline_setting={"p": 0},
        train_bars=252, test_bars=63,
        statistics_func=simple_statistics, capital=CAPITAL,
        n_null_sims=200, n_offsets=3, min_block_obs=40,
    )
    assert audit.decision != GO
    # 排名反转必须被 rank IC 抓到（这是比 PBO 点估计更直接的过拟合指纹）
    assert audit.pbo is not None
    assert audit.pbo.result.rank_ic < 0.0


def test_undecidable_is_distinct_from_no_go() -> None:
    """低功效样本上，「证明不了稳健」必须报「无法裁决」，不能报 NO-GO。

    这是三态设计的核心：把「数据不够格发证」和「策略被证伪」混成一个 NO-GO，
    会让人以为所有策略都被否定了，从而干脆不再看这份报告。

    构造：**微弱但为正**的漂移 + 低交易频率（逐折效率因此不可报）。
    没有任何一条正面反证成立（样本外不是显著为负、参数也稳定），
    但显著性够不上 —— 正确结论是「无法裁决」。
    """
    dates = make_dates(627)

    def pnl(setting, i, rng):
        return float((0.00015 + rng.normal(0.0, 0.012)) * CAPITAL)

    def pos(setting, i):
        # 每 80 根 K 线一个回合 → 63 天的测试窗期望不足 1 个回合 → 逐折效率不可报
        return 100.0 if (i % 80) < 25 else 0.0

    runner = FakeRunner(dates, pnl, pos, seed=9)
    audit = run_audit(
        runner, grid(10), dates[0], dates[-1],
        train_bars=252, test_bars=63,
        statistics_func=simple_statistics, capital=CAPITAL,
        with_pbo=False,
    )
    assert audit.decision == UNDECIDABLE
    # 「无法裁决」的理由里不得出现任何正面反证
    assert not any("显著为负" in b for b in audit.blockers)
    assert not any("负贡献" in b for b in audit.blockers)
    assert not any("严重衰减" in b for b in audit.blockers)
    assert any("不可区分" in b for b in audit.blockers)


# ══════════════════════════════════════════════════════════════════════
# C. 本项目实测复现 —— 零交易折病理的回归测试
# ══════════════════════════════════════════════════════════════════════

def test_zero_trade_folds_do_not_masquerade_as_decay() -> None:
    """复现 700.SEHK 实测：低频策略 + 63 日测试窗 → 大量零交易折。

    钉住两件事：
      1. 逐折活跃度能把零交易折识别出来（traded=False）；
      2. 可行性规划把 per_fold_efficiency_usable 判为 False，
         从而使裁决理由里出现"逐折效率不可报"而不是"严重衰减"。
    """
    dates = make_dates(627)

    def pnl(setting, i, rng):
        # 只在少数几个时段有盈亏，其余时间完全空仓
        return float(rng.normal(0.0, 0.01) * CAPITAL) if (i % 130) < 20 else 0.0

    def pos(setting, i):
        return 100.0 if (i % 130) < 20 else 0.0

    runner = FakeRunner(dates, pnl, pos, seed=13)
    audit = run_audit(
        runner, grid(8), dates[0], dates[-1],
        baseline_setting={"p": 0},
        train_bars=252, test_bars=63,
        statistics_func=simple_statistics, capital=CAPITAL,
        with_pbo=False,
    )

    assert audit.plan.round_trips_total < 20
    assert audit.plan.per_fold_efficiency_usable is False
    assert any("逐折效率不可报" in b for b in audit.blockers)
    assert not any("严重衰减" in b for b in audit.blockers)
    assert any(not a.traded for a in audit.activity)


def test_audit_rejects_when_optimization_loses_to_baseline() -> None:
    """优化打不过默认参数 = 正面反证 = NO-GO。这是本项目实测最锋利的那条判据。"""
    dates = make_dates(627)
    baseline = {"p": -1}

    def pnl(setting, i, rng):
        if setting["p"] == -1:
            return float((0.0016 + rng.normal(0.0, 0.004)) * CAPITAL)   # 默认参数很稳
        # 其余参数：样本内靠噪声碰运气，整体漂移为负
        return float((-0.0004 + rng.normal(0.0, 0.012)) * CAPITAL)

    def pos(setting, i):
        return 100.0 if (i // 5) % 2 == 0 else 0.0

    runner = FakeRunner(dates, pnl, pos, seed=17)
    audit = run_audit(
        runner, grid(10), dates[0], dates[-1],
        baseline_setting=baseline,
        train_bars=252, test_bars=63,
        statistics_func=simple_statistics, capital=CAPITAL,
        with_pbo=False,
    )
    assert audit.decision == NO_GO
    assert any("负贡献" in b for b in audit.blockers)


# ══════════════════════════════════════════════════════════════════════
# D. 接口与报告
# ══════════════════════════════════════════════════════════════════════

def test_run_audit_requires_statistics_func_for_custom_runner() -> None:
    dates = make_dates(400)
    runner = _noise_runner(dates)
    with pytest.raises(ValueError, match="statistics_func"):
        run_audit(runner, grid(4), dates[0], dates[-1], train_bars=252, test_bars=63)


def test_run_audit_rejects_empty_settings() -> None:
    dates = make_dates(400)
    runner = _noise_runner(dates)
    with pytest.raises(ValueError, match="settings"):
        run_audit(
            runner, [], dates[0], dates[-1],
            statistics_func=simple_statistics, capital=CAPITAL,
        )


def test_run_audit_rejects_sample_too_short_to_split() -> None:
    dates = make_dates(100)
    runner = _noise_runner(dates)
    with pytest.raises(ValueError, match="切不出"):
        run_audit(
            runner, grid(4), dates[0], dates[-1],
            train_bars=252, test_bars=63,
            statistics_func=simple_statistics, capital=CAPITAL,
        )


def test_audit_as_dict_is_serialisable_and_text_renders() -> None:
    import json

    dates = make_dates(627)
    runner = _noise_runner(dates)
    audit = run_audit(
        runner, grid(6), dates[0], dates[-1],
        baseline_setting={"p": 0},
        train_bars=252, test_bars=63,
        statistics_func=simple_statistics, capital=CAPITAL,
        with_pbo=False,
    )
    assert isinstance(audit, AuditResult)

    payload = audit.as_dict()
    text = json.dumps(payload, default=float, ensure_ascii=False)
    assert "decision" in payload and payload["decision"] in (GO, NO_GO, UNDECIDABLE)
    assert len(text) > 100

    rendered = format_audit(audit)
    assert "裁决：" in rendered
    assert "切分可行性" in rendered
    assert "逐折成交活跃度" in rendered


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
