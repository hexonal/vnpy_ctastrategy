"""Walk-Forward 效率的符号陷阱 + PBO 样本诊断的回合数口径。

本文件钉住两处**会让判据反向失灵**的性质，它们都不是"函数跑不跑得通"的问题：

1. **WFE 符号陷阱**（`walk_forward_efficiency`）
   效率 = 样本外年化 / 样本内年化。分母为负时裸除会把灾难折伪装成优秀折：
   样本内 −10%、样本外 −20%（亏得更狠）→ 比值 +2.0 → 落进"≥0.5 无衰减"档，
   并把中位数往上抬，足以把 `walk_forward_verdict` 从"不通过"翻成"通过"。
   修正后分母 ≤0 一律返回 nan（无定义），并从效率统计里剔除、在 warnings 里点名。

2. **回合数口径**（`pbo_from_settings(round_trip_counter=...)`）
   小样本诊断里唯一重要的分母是**完整回合数**。用 `trade_count/2`（成交笔数的一半）
   在金字塔加仓策略上会高估一倍以上（海龟 4 买 1 卖 = 5 笔成交 → 估 2.5 回合，实际 1），
   于是 PBO 段的诊断比 `overfitting_audit` 段宽松，同一份报告印出两个矛盾的回合数。

另外复核方法本身在零假设下的表现：**纯随机序列必须被判为不显著**，
且这条结论要在被本次改动碰过的 `run_walk_forward` 代码路径上成立（不只在
`assess_significance` 单函数层面）。
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pytest
from pandas import DataFrame

from vnpy_ctastrategy.overfitting import (
    annualised_sharpe,
    daily_log_returns,
    make_walk_forward_splits,
    pbo_from_settings,
    run_walk_forward,
    walk_forward_efficiency,
)

ANNUAL_DAYS = 252
CAPITAL = 1_000_000.0


# ══════════════════════════════════════════════════════════════════════
# 工具：可控的假 runner（自带成交笔数，便于验回合数口径）
# ══════════════════════════════════════════════════════════════════════

class FakeRunner:
    """把一张 (T, N) 逐日盈亏表当引擎用。

    `fills_per_bar` 控制每根 K 线记多少笔成交 —— 用来模拟金字塔加仓策略
    "一个回合 5 笔成交"的情形，验证 trade_count/2 估回合数会高估。
    """

    def __init__(
        self,
        dates: list[datetime],
        pnl: np.ndarray,
        settings: list[dict],
        fills_per_bar: float = 1.0,
    ) -> None:
        if pnl.shape != (len(dates), len(settings)):
            raise ValueError(f"pnl shape {pnl.shape} 与 dates/settings 不匹配")
        self.dates = dates
        self.pnl = pnl
        self.keys = [self._key(s) for s in settings]
        self.fills_per_bar = fills_per_bar

    @staticmethod
    def _key(setting: dict) -> tuple:
        return tuple(sorted(setting.items()))

    def __call__(self, setting: dict, start: datetime, end: datetime) -> DataFrame:
        col = self.keys.index(self._key(setting))
        rows = [i for i, d in enumerate(self.dates) if start <= d <= end]
        if not rows:
            return DataFrame()
        return DataFrame(
            {
                "net_pnl": self.pnl[rows, col],
                "trade_count": np.full(len(rows), self.fills_per_bar, dtype=float),
            },
            index=[self.dates[i].date() for i in rows],
        )

    @staticmethod
    def statistics(df: DataFrame) -> dict:
        if df is None or df.empty:
            return {"sharpe_ratio": 0.0, "annual_return": 0.0}
        r = daily_log_returns(df["net_pnl"].to_numpy(), CAPITAL)
        total = (float(np.sum(df["net_pnl"].to_numpy())) / CAPITAL) * 100.0
        return {
            "sharpe_ratio": annualised_sharpe(r, ANNUAL_DAYS),
            "annual_return": total / len(df) * ANNUAL_DAYS,
        }


def _dates(n: int) -> list[datetime]:
    base = datetime(2024, 1, 2)
    return [base + timedelta(days=i) for i in range(n)]


def _grid(n: int) -> list[dict]:
    return [{"window": 10 + i} for i in range(n)]


# ══════════════════════════════════════════════════════════════════════
# 1. WFE 符号陷阱 —— 纯函数层
# ══════════════════════════════════════════════════════════════════════

def test_efficiency_is_the_plain_ratio_when_in_sample_is_positive() -> None:
    assert walk_forward_efficiency(20.0, 10.0) == pytest.approx(0.5)
    assert walk_forward_efficiency(20.0, 24.0) == pytest.approx(1.2)


def test_efficiency_is_negative_when_a_winning_fold_flips_out_of_sample() -> None:
    """样本内赚、样本外亏 = 负效率。这个符号是对的，必须保留。"""
    assert walk_forward_efficiency(10.0, -5.0) == pytest.approx(-0.5)


def test_losing_in_sample_fold_does_not_masquerade_as_efficient() -> None:
    """核心回归：IS −10% / OOS −20%（亏得更狠）绝不能报成 +2.0。"""
    eff = walk_forward_efficiency(-10.0, -20.0)
    assert math.isnan(eff), f"分母为负时必须无定义，得到 {eff}"


def test_losing_in_sample_fold_that_improves_is_also_undefined() -> None:
    """IS −10% / OOS +5%：裸除得 −0.5（把变好记成负效率），同样必须无定义。"""
    assert math.isnan(walk_forward_efficiency(-10.0, 5.0))


def test_flat_in_sample_fold_is_undefined_not_infinite() -> None:
    assert math.isnan(walk_forward_efficiency(0.0, 5.0))
    assert math.isnan(walk_forward_efficiency(1e-12, 5.0))


def test_non_finite_inputs_are_undefined() -> None:
    assert math.isnan(walk_forward_efficiency(float("nan"), 1.0))
    assert math.isnan(walk_forward_efficiency(1.0, float("inf")))


# ══════════════════════════════════════════════════════════════════════
# 2. WFE 符号陷阱 —— 端到端（这是判据真正会读到的那个数）
# ══════════════════════════════════════════════════════════════════════

def _pnl_with_one_losing_fold(
    n_days: int, n_cfg: int, train: int, test: int, seed: int = 7
) -> np.ndarray:
    """构造：第 0 折的测试窗之前那段训练窗全亏，且其样本外亏得更多。

    做法是把整段设成温和上涨，再把 [0, train+test) 这一段整体压成下跌。
    这样第 0 折（train=[0,train), test=[train,train+test)）样本内外都为负，
    后面几折仍为正 —— 正好制造"一个折污染中位数"的场景。
    """
    rng = np.random.default_rng(seed)
    pnl = rng.normal(300.0, 2_000.0, size=(n_days, n_cfg))
    pnl[: train + test, :] = rng.normal(-400.0, 2_000.0, size=(train + test, n_cfg))
    # 让第 0 折的样本外比样本内亏得更狠 —— 裸除会得到正效率
    pnl[train: train + test, :] -= 1_500.0
    return pnl


def test_losing_fold_is_excluded_from_the_efficiency_median() -> None:
    train, test, n_cfg = 100, 25, 6
    n_days = train + test * 5
    dates = _dates(n_days)
    settings = _grid(n_cfg)
    runner = FakeRunner(dates, _pnl_with_one_losing_fold(n_days, n_cfg, train, test), settings)
    splits = make_walk_forward_splits(dates, train_bars=train, test_bars=test)

    report = run_walk_forward(
        runner, settings, splits,
        statistics_func=FakeRunner.statistics, annual_days=ANNUAL_DAYS,
        capital=CAPITAL, n_bootstrap=500,
    )

    bad = [f for f in report.folds if f.is_annual_return <= 0]
    assert bad, "构造失败：没有任何一折样本内为负，这个测试就没意义了"
    for f in bad:
        assert math.isnan(f.efficiency)
        # 裸除会给出的那个值必须没有出现在报告里
        naive = f.oos_annual_return / f.is_annual_return
        assert not any(
            math.isclose(x.efficiency, naive, rel_tol=1e-9)
            for x in report.folds if math.isfinite(x.efficiency)
        )

    finite = [f.efficiency for f in report.folds if math.isfinite(f.efficiency)]
    assert report.efficiency_median == pytest.approx(float(np.median(finite)))
    assert any("样本内年化 ≤ 0" in w for w in report.warnings)


def test_all_folds_losing_gives_undefined_median_and_an_honest_verdict() -> None:
    """全折样本内为负 → 中位数 nan，判据必须说"不可计算"，不能印成 nan<0.5。"""
    train, test, n_cfg = 60, 20, 4
    n_days = train + test * 4
    dates = _dates(n_days)
    settings = _grid(n_cfg)
    rng = np.random.default_rng(11)
    pnl = rng.normal(-500.0, 1_500.0, size=(n_days, n_cfg))
    runner = FakeRunner(dates, pnl, settings)
    splits = make_walk_forward_splits(dates, train_bars=train, test_bars=test)

    report = run_walk_forward(
        runner, settings, splits,
        statistics_func=FakeRunner.statistics, annual_days=ANNUAL_DAYS,
        capital=CAPITAL, n_bootstrap=500,
    )

    assert all(f.is_annual_return <= 0 for f in report.folds)
    assert math.isnan(report.efficiency_median)
    assert "不可计算" in report.verdict
    assert "nan" not in report.verdict
    assert report.verdict.startswith("不通过")


# ══════════════════════════════════════════════════════════════════════
# 3. 零假设复核 —— 纯随机序列必须被判"不显著"
# ══════════════════════════════════════════════════════════════════════

def test_walk_forward_on_pure_noise_is_not_significant() -> None:
    """无漂移的随机盈亏：样本外 Sharpe 不能被判显著，判据必须"不通过"。"""
    train, test, n_cfg = 100, 25, 8
    n_days = train + test * 6
    dates = _dates(n_days)
    settings = _grid(n_cfg)
    rng = np.random.default_rng(20260725)
    pnl = rng.normal(0.0, 2_000.0, size=(n_days, n_cfg))     # 漂移严格为 0
    runner = FakeRunner(dates, pnl, settings)
    splits = make_walk_forward_splits(dates, train_bars=train, test_bars=test)

    report = run_walk_forward(
        runner, settings, splits,
        statistics_func=FakeRunner.statistics, annual_days=ANNUAL_DAYS,
        capital=CAPITAL, n_bootstrap=2000,
    )

    assert report.significance.significant is False
    assert report.significance.p_block_bootstrap > 0.05
    assert report.verdict.startswith("不通过")


def test_walk_forward_false_positive_rate_is_near_nominal() -> None:
    """20 条独立噪声路径里，被判显著的不应明显多于 alpha —— 检验的 size 正确。

    20 次 × alpha=0.05 的期望误报数是 1；上限取 4（二项尾部允许，钉住"不会到处报喜"）。
    """
    train, test, n_cfg = 80, 20, 6
    n_days = train + test * 5
    dates = _dates(n_days)
    settings = _grid(n_cfg)
    splits = make_walk_forward_splits(dates, train_bars=train, test_bars=test)

    false_positives = 0
    for seed in range(20):
        rng = np.random.default_rng(1000 + seed)
        pnl = rng.normal(0.0, 2_000.0, size=(n_days, n_cfg))
        runner = FakeRunner(dates, pnl, settings)
        report = run_walk_forward(
            runner, settings, splits,
            statistics_func=FakeRunner.statistics, annual_days=ANNUAL_DAYS,
            capital=CAPITAL, n_bootstrap=800, seed=seed,
        )
        false_positives += int(report.significance.significant)

    assert false_positives <= 4, f"20 条纯噪声里报了 {false_positives} 次显著，检验 size 失控"


# ══════════════════════════════════════════════════════════════════════
# 4. PBO 样本诊断的回合数口径
# ══════════════════════════════════════════════════════════════════════

def _pbo_kwargs() -> dict:
    return {
        "capital": CAPITAL,
        "annual_days": ANNUAL_DAYS,
        "min_block_obs": 20,
        "n_offsets": 2,
        "n_null_sims": 0,       # 零分布对本组断言无关，省时间
    }


def test_round_trip_counter_overrides_the_fills_over_two_estimate() -> None:
    """金字塔加仓：成交笔数远多于回合数。默认口径高估，传 counter 后精确。

    数据里每根 K 线记 1 笔成交（300 根 = 300 笔），而按仓位序列数出来的完整回合
    只有 3 个 —— 这正是本项目 700.SEHK 实测的量级（约 30 笔成交 / 9 个回合）。
    `count_round_trips` 在真实 daily_df 上做的就是这件事，这里用固定返回值代替，
    以免把被测对象（诊断口径）和 FakeRunner 的仓位模拟混在一起。
    """
    n_days, n_cfg = 300, 20
    dates = _dates(n_days)
    settings = _grid(n_cfg)
    rng = np.random.default_rng(3)
    pnl = rng.normal(0.0, 1_500.0, size=(n_days, n_cfg))
    runner = FakeRunner(dates, pnl, settings, fills_per_bar=1.0)

    loose = pbo_from_settings(runner, settings, dates[0], dates[-1], **_pbo_kwargs())
    strict = pbo_from_settings(
        runner, settings, dates[0], dates[-1],
        round_trip_counter=lambda df: 3,
        **_pbo_kwargs(),
    )

    # fills/2 = 150（每侧 75 笔，看着完全够用）；真实回合 = 3（每侧 1.5 笔）
    assert loose.diagnosis.trades_total == 150
    assert strict.diagnosis.trades_total == 3
    assert strict.diagnosis.trades_total < loose.diagnosis.trades_total
    assert any("trade_count/2" in n for n in loose.notes)
    assert not any("trade_count/2" in n for n in strict.notes)
    # PBO 本身与回合数口径无关，两条路径必须给出同一个数
    assert strict.result.pbo == pytest.approx(loose.result.pbo)


def test_round_trip_counter_makes_the_sample_diagnosis_stricter() -> None:
    """口径变严 → 每侧回合数掉到资格线下 → "样本没资格"的警告必须出现。

    宽松口径下每侧 75 笔（看着够用、不会触发警告），严格口径下每侧 1.5 笔。
    这条测试钉住的是：**换对口径会改变结论，不是改变措辞**。
    """
    n_days, n_cfg = 300, 20
    dates = _dates(n_days)
    settings = _grid(n_cfg)
    rng = np.random.default_rng(4)
    pnl = rng.normal(0.0, 1_500.0, size=(n_days, n_cfg))
    runner = FakeRunner(dates, pnl, settings, fills_per_bar=1.0)

    loose = pbo_from_settings(runner, settings, dates[0], dates[-1], **_pbo_kwargs())
    assert not any("经验下限 20 笔" in n for n in loose.diagnosis.notes)

    strict = pbo_from_settings(
        runner, settings, dates[0], dates[-1],
        round_trip_counter=lambda df: 3,
        **_pbo_kwargs(),
    )
    assert strict.diagnosis.trades_per_side == pytest.approx(1.5)
    assert any("经验下限 20 笔" in n for n in strict.diagnosis.notes)
    assert strict.diagnosis.min_detectable_sharpe > 1.5
