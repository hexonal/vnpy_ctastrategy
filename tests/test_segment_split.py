"""三段切分（TRAIN / VALID / TEST）的验证测试。

本文件的重点不是"函数跑不跑得通"，而是**纪律有没有被代码钉死**：

  1. 三段边界互不重叠 —— 没有任何一根 K 线同时属于两段，
     且 TEST 严格晚于 VALID、VALID 严格晚于 TRAIN。
  2. 测试段的统计量能单独取出，且自带"这是样本外"的标记；
     VALID 必须被标成【样本内】（官方第 4 篇口径）。
  3. 在 TEST 上扫参数会被挡住；重复查看 TEST 会被预算挡住；
     重开预算必须留痕。
  4. **构造性反例**：造一组"在 TRAIN 上最差、在 TEST 上最好"的参数，
     它绝不许被选中 —— 这是"TEST 没参与选参"的唯一硬证据，
     光看代码路径不算。
  5. 取数 API 的 segment 参数无默认值（抄官方 predict(dataset, segment)
     的形状），用 inspect 直接钉住签名。
"""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta

import numpy as np
import pytest
from pandas import DataFrame
from vnpy.trader.constant import Interval
from vnpy.trader.optimize import OptimizationSetting

from vnpy_ctastrategy.backtesting import BacktestingEngine
from vnpy_ctastrategy.overfitting import annualised_sharpe, daily_log_returns
from vnpy_ctastrategy.segments import (
    HoldoutReport,
    Segment,
    SegmentBudgetExhaustedError,
    SegmentedRunner,
    SegmentGuardedEngine,
    SegmentLeakError,
    SegmentResult,
    ThreeWaySplit,
    assert_segment_parity,
    in_sample_segments,
    is_out_of_sample,
    make_three_way_split,
    run_holdout,
    split_by_ratio,
)

ANNUAL_DAYS = 252
CAPITAL = 1_000_000.0


# ══════════════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════════════

def _dates(n: int) -> list[datetime]:
    base = datetime(2024, 1, 2)
    return [base + timedelta(days=i) for i in range(n)]


def _grid(n: int) -> list[dict]:
    return [{"window": 10 + i} for i in range(n)]


class FakeRunner:
    """把一张预先生成好的 (T, N) 逐日盈亏表当成回测引擎。

    与 `tests/test_overfitting.py` 里的同名工具同构：同一组参数在同一天的
    盈亏与窗口无关，先生成整段序列再按窗口切片。这样"哪一段选了哪组参数"
    完全可控，构造性反例才立得住。
    """

    def __init__(
        self,
        dates: list[datetime],
        pnl: np.ndarray,
        settings: list[dict],
        trades: np.ndarray | None = None,
    ) -> None:
        if pnl.shape != (len(dates), len(settings)):
            raise ValueError(f"pnl shape {pnl.shape} 与 dates/settings 不匹配")
        self.dates = dates
        self.pnl = pnl
        self.trades = np.ones_like(pnl) if trades is None else trades
        self.keys = [self._key(s) for s in settings]
        self.calls: list[tuple[tuple, datetime, datetime]] = []

    @staticmethod
    def _key(setting: dict) -> tuple:
        return tuple(sorted(setting.items()))

    def __call__(self, setting: dict, start: datetime, end: datetime) -> DataFrame:
        col = self.keys.index(self._key(setting))
        self.calls.append((self._key(setting), start, end))
        mask = [i for i, d in enumerate(self.dates) if start <= d <= end]
        if not mask:
            return DataFrame()
        return DataFrame(
            {
                "net_pnl": self.pnl[mask, col],
                "trade_count": self.trades[mask, col],
            },
            index=[self.dates[i].date() for i in mask],
        )

    @staticmethod
    def statistics(df: DataFrame) -> dict:
        """最小 statistics 面板：只保留本文件用得到的键。"""
        if df is None or df.empty:
            return {"sharpe_ratio": 0.0, "annual_return": 0.0, "total_days": 0}
        r = daily_log_returns(df["net_pnl"].to_numpy(), CAPITAL)
        total = (float(np.sum(df["net_pnl"].to_numpy())) / CAPITAL) * 100.0
        return {
            "sharpe_ratio": annualised_sharpe(r, ANNUAL_DAYS),
            "annual_return": total / len(df) * ANNUAL_DAYS,
            "total_days": len(df),
        }


def _runner(
    dates: list[datetime],
    pnl: np.ndarray,
    settings: list[dict],
    split: ThreeWaySplit,
    trades: np.ndarray | None = None,
    test_budget: int = 1,
) -> SegmentedRunner:
    return SegmentedRunner(
        runner=FakeRunner(dates, pnl, settings, trades),
        split=split,
        statistics_func=FakeRunner.statistics,
        capital=CAPITAL,
        test_budget=test_budget,
    )


def _window_dates(dts: list[datetime], lo: datetime, hi: datetime) -> set[datetime]:
    return {d for d in dts if lo <= d <= hi}


# ══════════════════════════════════════════════════════════════════════
# 1. 枚举与语义
# ══════════════════════════════════════════════════════════════════════

def test_segment_enum_matches_vnpy_alpha() -> None:
    """本地 Segment 与 vnpy.alpha 的枚举必须逐项一致。

    刻意不 import vnpy.alpha（那会把 polars/alphalens 变成本包硬依赖），
    对齐由本测试钉住：上游改了枚举值，这里会红。
    """
    checked = assert_segment_parity()
    if not checked:
        pytest.skip("vnpy.alpha 未安装（optional extra），本次未对比")
    from vnpy.alpha.dataset.utility import Segment as AlphaSegment

    assert {m.name: m.value for m in Segment} == {
        m.name: m.value for m in AlphaSegment
    }


def test_valid_counts_as_in_sample() -> None:
    """官方第 4 篇：只要按验证集表现调参，它就参与了研究决策 → 样本内。"""
    assert is_out_of_sample(Segment.TRAIN) is False
    assert is_out_of_sample(Segment.VALID) is False
    assert is_out_of_sample(Segment.TEST) is True
    assert in_sample_segments() == (Segment.TRAIN, Segment.VALID)


def test_is_out_of_sample_rejects_non_segment() -> None:
    with pytest.raises(TypeError):
        is_out_of_sample("TEST")            # type: ignore[arg-type]


# ══════════════════════════════════════════════════════════════════════
# 2. 切分：边界不重叠
# ══════════════════════════════════════════════════════════════════════

def test_three_way_split_boundaries_do_not_overlap() -> None:
    """三段严格时序相邻，且没有任何一根 K 线同时落进两段。"""
    dts = _dates(300)
    split = make_three_way_split(dts, train_bars=180, valid_bars=60, test_bars=60)

    assert split.train_end < split.valid_start
    assert split.valid_end < split.test_start

    train = _window_dates(dts, *split.as_period(Segment.TRAIN))
    valid = _window_dates(dts, *split.as_period(Segment.VALID))
    test = _window_dates(dts, *split.as_period(Segment.TEST))

    assert len(train) == 180 == split.bars(Segment.TRAIN)
    assert len(valid) == 60 == split.bars(Segment.VALID)
    assert len(test) == 60 == split.bars(Segment.TEST)
    assert not (train & valid)
    assert not (valid & test)
    assert not (train & test)
    assert len(train | valid | test) == 300


def test_anchor_end_puts_test_at_the_newest_bars() -> None:
    """默认 anchor='end'：TEST 贴着样本末尾 —— "把最近这段锁起来"。"""
    dts = _dates(300)
    end_split = make_three_way_split(dts, 100, 40, 40)
    assert end_split.test_end == dts[-1]
    assert end_split.train_start == dts[300 - 180]

    start_split = make_three_way_split(dts, 100, 40, 40, anchor="start")
    assert start_split.train_start == dts[0]
    assert start_split.test_end == dts[179]


def test_split_by_ratio_consumes_every_bar() -> None:
    dts = _dates(251)
    split = split_by_ratio(dts, train=0.6, valid=0.2, test=0.2)
    total = split.train_bars + split.valid_bars + split.test_bars
    assert total == 251
    assert split.train_start == dts[0]
    assert split.test_end == dts[-1]

    train = _window_dates(dts, *split.as_period(Segment.TRAIN))
    valid = _window_dates(dts, *split.as_period(Segment.VALID))
    test = _window_dates(dts, *split.as_period(Segment.TEST))
    assert len(train | valid | test) == 251
    assert not (train & valid) and not (valid & test) and not (train & test)


def test_in_sample_period_spans_train_and_valid() -> None:
    dts = _dates(200)
    split = make_three_way_split(dts, 120, 40, 40, anchor="start")
    lo, hi = split.in_sample_period()
    assert lo == split.train_start
    assert hi == split.valid_end
    assert hi < split.test_start


def test_make_three_way_split_rejects_bad_inputs() -> None:
    dts = _dates(100)
    with pytest.raises(ValueError):
        make_three_way_split(dts, 60, 30, 30)          # 合计 120 > 100
    with pytest.raises(ValueError):
        make_three_way_split(dts, 1, 10, 10)           # train_bars < 2
    with pytest.raises(ValueError):
        make_three_way_split(dts, 50, 0, 10)           # valid_bars < 1
    with pytest.raises(ValueError):
        make_three_way_split(dts, 50, 10, 0)           # test_bars < 1
    with pytest.raises(ValueError):
        make_three_way_split([], 2, 1, 1)              # 空样本
    with pytest.raises(ValueError):
        make_three_way_split(dts, 50, 10, 10, anchor="middle")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        make_three_way_split(list(reversed(dts)), 50, 10, 10)   # 非递增


def test_split_by_ratio_rejects_bad_ratios() -> None:
    dts = _dates(100)
    with pytest.raises(ValueError):
        split_by_ratio(dts, train=0.6, valid=0.2, test=0.3)     # 和不为 1
    with pytest.raises(ValueError):
        split_by_ratio(dts, train=1.0, valid=0.0, test=0.0)     # 比例非正
    with pytest.raises(ValueError):
        split_by_ratio(_dates(4), train=0.6, valid=0.2, test=0.2)  # 段太短


def test_three_way_split_post_init_rejects_overlap() -> None:
    """直接构造出重叠边界也必须被拦下，不只是工厂函数把关。"""
    d = _dates(10)
    with pytest.raises(ValueError, match="重叠"):
        ThreeWaySplit(
            train_start=d[0], train_end=d[5],
            valid_start=d[3], valid_end=d[7],
            test_start=d[8], test_end=d[9],
            train_bars=6, valid_bars=5, test_bars=2,
        )
    with pytest.raises(ValueError, match="重叠"):
        ThreeWaySplit(
            train_start=d[0], train_end=d[2],
            valid_start=d[3], valid_end=d[7],
            test_start=d[5], test_end=d[9],
            train_bars=3, valid_bars=5, test_bars=5,
        )


def test_split_period_and_bars_reject_non_segment() -> None:
    split = make_three_way_split(_dates(100), 60, 20, 20)
    with pytest.raises(TypeError):
        split.as_period("TRAIN")        # type: ignore[arg-type]
    with pytest.raises(TypeError):
        split.bars("TEST")              # type: ignore[arg-type]


def test_split_as_dict_is_plain() -> None:
    split = make_three_way_split(_dates(100), 60, 20, 20)
    d = split.as_dict()
    assert d["train_bars"] == 60 and d["valid_bars"] == 20 and d["test_bars"] == 20
    assert d["train_end"] < d["valid_start"] < d["valid_end"] < d["test_start"]


# ══════════════════════════════════════════════════════════════════════
# 3. 强制显式性 —— segment 一律无默认值
# ══════════════════════════════════════════════════════════════════════

def test_segment_argument_has_no_default_anywhere() -> None:
    """抄官方 `AlphaModel.predict(dataset, segment)` 的形状：把纪律变成签名。

    凡"取某一段结果"的 API，segment 都必须是必填参数 ——
    有默认值就意味着可以不写，不写就意味着看的人不知道这是哪一段。
    """
    targets = [
        ThreeWaySplit.as_period,
        ThreeWaySplit.bars,
        SegmentedRunner.run,
        SegmentedRunner.scan,
        HoldoutReport.result,
        HoldoutReport.statistics,
        HoldoutReport.daily_df,
        HoldoutReport.target,
        is_out_of_sample,
    ]
    for func in targets:
        params = inspect.signature(func).parameters
        assert "segment" in params, f"{func.__qualname__} 没有 segment 参数"
        assert params["segment"].default is inspect.Parameter.empty, (
            f"{func.__qualname__} 的 segment 有默认值 —— 显式性被破坏"
        )


def test_alpha_predict_shape_is_the_reference() -> None:
    """官方 predict(dataset, segment) 本身也是无默认值 —— 我们抄的就是它。"""
    try:
        from vnpy.alpha.model.template import AlphaModel
    except ImportError:
        pytest.skip("vnpy.alpha 未安装（optional extra）")
    params = inspect.signature(AlphaModel.predict).parameters
    assert params["segment"].default is inspect.Parameter.empty


# ══════════════════════════════════════════════════════════════════════
# 4. 分段执行器：取数与标注
# ══════════════════════════════════════════════════════════════════════

def test_segment_result_carries_in_out_of_sample_marker() -> None:
    """每段 statistics 自带 segment 与 is_out_of_sample 两个键。

    这治的是"样本内的 sharpe 与样本外的长得一模一样"：
    面板流到下游时，这两个键跟着一起流。
    """
    dts = _dates(120)
    settings = _grid(3)
    rng = np.random.default_rng(7)
    pnl = rng.normal(400, 6_000, size=(120, 3))
    split = make_three_way_split(dts, 70, 25, 25, anchor="start")
    runner = _runner(dts, pnl, settings, split)

    train = runner.run(settings[0], Segment.TRAIN)
    assert train.statistics["segment"] == "TRAIN"
    assert train.statistics["is_out_of_sample"] is False
    assert train.out_of_sample is False
    assert train.daily_df.attrs["segment"] == "TRAIN"
    assert train.statistics["total_days"] == 70

    valid = runner.run(settings[0], Segment.VALID)
    assert valid.statistics["is_out_of_sample"] is False   # VALID 是样本内
    assert valid.statistics["total_days"] == 25

    test = runner.run(settings[0], Segment.TEST)
    assert test.statistics["segment"] == "TEST"
    assert test.statistics["is_out_of_sample"] is True
    assert test.out_of_sample is True
    assert test.statistics["total_days"] == 25


def test_segment_result_windows_match_the_split() -> None:
    """执行器喂给底层 runner 的起止，必须正是该段的边界。"""
    dts = _dates(100)
    settings = _grid(2)
    split = make_three_way_split(dts, 60, 20, 20, anchor="start")
    fake = FakeRunner(dts, np.zeros((100, 2)), settings)
    runner = SegmentedRunner(
        runner=fake, split=split, statistics_func=FakeRunner.statistics,
        capital=CAPITAL,
    )

    runner.run(settings[0], Segment.TRAIN)
    runner.run(settings[0], Segment.VALID)
    runner.run(settings[0], Segment.TEST)

    windows = [(start, end) for _, start, end in fake.calls]
    assert windows == [
        split.as_period(Segment.TRAIN),
        split.as_period(Segment.VALID),
        split.as_period(Segment.TEST),
    ]


def test_segment_result_target_and_trade_count() -> None:
    dts = _dates(60)
    settings = _grid(1)
    split = make_three_way_split(dts, 30, 15, 15, anchor="start")
    trades = np.full((60, 1), 2.0)
    runner = _runner(dts, np.zeros((60, 1)), settings, split, trades=trades)

    res: SegmentResult = runner.run(settings[0], Segment.TEST)
    assert res.trade_count() == pytest.approx(30.0)      # 15 天 × 2 笔
    assert np.isnan(res.target("no_such_key"))


def test_segmented_runner_requires_statistics_func_for_custom_runner() -> None:
    dts = _dates(60)
    split = make_three_way_split(dts, 30, 15, 15, anchor="start")
    fake = FakeRunner(dts, np.zeros((60, 1)), _grid(1))
    with pytest.raises(ValueError, match="statistics_func"):
        SegmentedRunner(runner=fake, split=split)


def test_segmented_runner_rejects_non_segment_and_bad_budget() -> None:
    dts = _dates(60)
    settings = _grid(1)
    split = make_three_way_split(dts, 30, 15, 15, anchor="start")
    runner = _runner(dts, np.zeros((60, 1)), settings, split)
    with pytest.raises(TypeError):
        runner.run(settings[0], "TEST")         # type: ignore[arg-type]
    with pytest.raises(TypeError):
        runner.scan(settings, "TRAIN")          # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _runner(dts, np.zeros((60, 1)), settings, split, test_budget=0)


# ══════════════════════════════════════════════════════════════════════
# 5. 两道闸：TEST 上不许扫参数、不许反复看
# ══════════════════════════════════════════════════════════════════════

def test_scanning_the_test_segment_is_blocked() -> None:
    """在 TEST 上扫参数网格 = 用测试段做模型选择 —— 代码直接拦，不是告警。"""
    dts = _dates(120)
    settings = _grid(4)
    split = make_three_way_split(dts, 70, 25, 25, anchor="start")
    runner = _runner(dts, np.zeros((120, 4)), settings, split)

    with pytest.raises(SegmentLeakError, match="TEST"):
        runner.scan(settings, Segment.TEST)

    # 被拦住的动作不许留下副作用：预算一次都没被消耗
    assert runner.test_calls == 0
    assert runner.test_audit == []

    # 样本内两段照常放行
    assert len(runner.scan(settings, Segment.TRAIN)) == 4
    assert len(runner.scan(settings, Segment.VALID)) == 4


def test_scan_rejects_empty_settings() -> None:
    dts = _dates(60)
    split = make_three_way_split(dts, 30, 15, 15, anchor="start")
    runner = _runner(dts, np.zeros((60, 1)), _grid(1), split)
    with pytest.raises(ValueError, match="settings"):
        runner.scan([], Segment.TRAIN)


def test_second_peek_at_test_segment_is_blocked() -> None:
    """默认预算 1 次。第二次取数即抛，且审计日志留下两条记录。"""
    dts = _dates(120)
    settings = _grid(3)
    split = make_three_way_split(dts, 70, 25, 25, anchor="start")
    runner = _runner(dts, np.zeros((120, 3)), settings, split)

    runner.run(settings[0], Segment.TEST)
    assert runner.test_calls == 1

    with pytest.raises(SegmentBudgetExhaustedError, match="预算"):
        runner.run(settings[1], Segment.TEST)

    assert runner.test_calls == 2
    assert len(runner.test_audit) == 2
    assert "window" in runner.test_audit[1]      # 被拒的那次也留痕


def test_reset_test_budget_demands_a_reason() -> None:
    """重开测试集是研究决策：必须显式、必须留痕、不许无声发生。"""
    dts = _dates(120)
    settings = _grid(2)
    split = make_three_way_split(dts, 70, 25, 25, anchor="start")
    runner = _runner(dts, np.zeros((120, 2)), settings, split)

    runner.run(settings[0], Segment.TEST)
    with pytest.raises(SegmentBudgetExhaustedError):
        runner.run(settings[0], Segment.TEST)

    with pytest.raises(ValueError, match="reason"):
        runner.reset_test_budget("")
    with pytest.raises(ValueError, match="reason"):
        runner.reset_test_budget("   ")

    runner.reset_test_budget(reason="数据修复后重跑，前一次结果作废")
    assert runner.test_calls == 0
    assert any("RESET" in line for line in runner.test_audit)
    assert any("数据修复" in line for line in runner.test_audit)

    runner.run(settings[0], Segment.TEST)        # 重开后可再看一次
    assert runner.test_calls == 1


def test_larger_budget_allows_that_many_peeks() -> None:
    dts = _dates(120)
    settings = _grid(3)
    split = make_three_way_split(dts, 70, 25, 25, anchor="start")
    runner = _runner(dts, np.zeros((120, 3)), settings, split, test_budget=2)

    runner.run(settings[0], Segment.TEST)
    runner.run(settings[1], Segment.TEST)
    with pytest.raises(SegmentBudgetExhaustedError):
        runner.run(settings[2], Segment.TEST)


# ══════════════════════════════════════════════════════════════════════
# 6. holdout：TEST 不参与选参（构造性反例）
# ══════════════════════════════════════════════════════════════════════

def _staged_pnl(
    dts: list[datetime], split: ThreeWaySplit, per_segment: dict[Segment, list[float]]
) -> np.ndarray:
    """按段给每组参数灌固定的日盈亏 —— 让"哪段谁最好"完全可控。"""
    n_settings = len(next(iter(per_segment.values())))
    pnl = np.zeros((len(dts), n_settings), dtype=float)
    for segment, values in per_segment.items():
        lo, hi = split.as_period(segment)
        rows = [i for i, d in enumerate(dts) if lo <= d <= hi]
        for col, value in enumerate(values):
            pnl[rows, col] = value
    return pnl


def test_holdout_selection_ignores_test_segment() -> None:
    """构造性反例：参数 2 在 TEST 上最好、在 TRAIN 上最差，它绝不许被选中。

    这是"TEST 没参与选参"的硬证据。只看代码路径不算 ——
    必须让一个"如果偷看 TEST 就会被选中"的参数存在，并证明它没被选中。
    """
    dts = _dates(150)
    settings = _grid(3)
    split = make_three_way_split(dts, 90, 30, 30, anchor="start")
    pnl = _staged_pnl(
        dts, split,
        {
            Segment.TRAIN: [3_000.0, 1_000.0, -2_000.0],   # 0 最好，2 最差
            Segment.VALID: [2_000.0, 1_000.0, -1_000.0],
            Segment.TEST: [-5_000.0, 0.0, 9_000.0],        # 2 最好
        },
    )
    runner = _runner(dts, pnl, settings, split)

    report = run_holdout(
        runner, settings, target_name="annual_return",
        significance=True, n_bootstrap=200, annual_days=ANNUAL_DAYS,
    )

    assert report.chosen_index == 0
    assert report.chosen_setting == settings[0]
    # 若曾按 TEST 选参，选中的会是 2；这里显式钉住反面
    assert report.chosen_setting != settings[2]
    assert report.target(Segment.TRAIN) > 0
    assert report.target(Segment.TEST) < 0            # 样本外确实翻车，照实报


def test_holdout_test_statistics_can_be_taken_alone() -> None:
    """测试段的统计量必须能单独取出，并自带样本外标记。"""
    dts = _dates(150)
    settings = _grid(3)
    split = make_three_way_split(dts, 90, 30, 30, anchor="start")
    rng = np.random.default_rng(3)
    pnl = rng.normal(300, 5_000, size=(150, 3))
    runner = _runner(dts, pnl, settings, split)

    report = run_holdout(
        runner, settings, target_name="sharpe_ratio",
        significance=True, n_bootstrap=200, annual_days=ANNUAL_DAYS,
    )

    test_stats = report.statistics(Segment.TEST)
    train_stats = report.statistics(Segment.TRAIN)
    valid_stats = report.statistics(Segment.VALID)

    assert test_stats["is_out_of_sample"] is True
    assert train_stats["is_out_of_sample"] is False
    assert valid_stats["is_out_of_sample"] is False
    assert test_stats["total_days"] == 30
    assert train_stats["total_days"] == 90
    assert valid_stats["total_days"] == 30
    assert test_stats["sharpe_ratio"] != train_stats["sharpe_ratio"]

    test_df = report.daily_df(Segment.TEST)
    assert len(test_df) == 30
    assert test_df.attrs["segment"] == "TEST"

    assert report.test_significance is not None
    assert report.test_significance.n_obs == 30

    summary = report.summary()
    assert "TEST" in summary and "样本外" in summary
    assert report.as_dict()["targets"]["TEST"] == pytest.approx(
        report.target(Segment.TEST)
    )


def test_holdout_consumes_the_test_budget_exactly_once() -> None:
    """holdout 跑完后 TEST 预算正好用掉 1 次 —— 再看一眼就会被挡。"""
    dts = _dates(150)
    settings = _grid(4)
    split = make_three_way_split(dts, 90, 30, 30, anchor="start")
    rng = np.random.default_rng(5)
    pnl = rng.normal(200, 4_000, size=(150, 4))
    runner = _runner(dts, pnl, settings, split)

    run_holdout(runner, settings, significance=False)
    assert runner.test_calls == 1

    with pytest.raises(SegmentBudgetExhaustedError):
        runner.run(settings[0], Segment.TEST)


def test_holdout_only_runs_test_window_for_the_chosen_setting() -> None:
    """底层 runner 在 TEST 窗口上只被调用一次，且用的是被选中的那组参数。"""
    dts = _dates(150)
    settings = _grid(4)
    split = make_three_way_split(dts, 90, 30, 30, anchor="start")
    pnl = _staged_pnl(
        dts, split,
        {
            Segment.TRAIN: [500.0, 4_000.0, 100.0, -100.0],
            Segment.VALID: [500.0, 3_000.0, 100.0, -100.0],
            Segment.TEST: [9_000.0, 10.0, 9_000.0, 9_000.0],
        },
    )
    fake = FakeRunner(dts, pnl, settings)
    runner = SegmentedRunner(
        runner=fake, split=split, statistics_func=FakeRunner.statistics,
        capital=CAPITAL,
    )
    report = run_holdout(
        runner, settings, target_name="annual_return", significance=False
    )

    assert report.chosen_index == 1
    test_window = split.as_period(Segment.TEST)
    test_calls = [c for c in fake.calls if (c[1], c[2]) == test_window]
    assert len(test_calls) == 1
    assert test_calls[0][0] == tuple(sorted(settings[1].items()))


def test_holdout_warns_on_zero_trades_and_short_test() -> None:
    """零成交与测试段过短都必须显式喊出来，不能被读成"样本外持平"。"""
    dts = _dates(60)
    settings = _grid(2)
    split = make_three_way_split(dts, 35, 15, 10, anchor="start")
    trades = np.ones((60, 2))
    test_rows = [
        i for i, d in enumerate(dts)
        if split.test_start <= d <= split.test_end
    ]
    trades[test_rows, :] = 0.0
    pnl = _staged_pnl(
        dts, split,
        {
            Segment.TRAIN: [1_000.0, 500.0],
            Segment.VALID: [800.0, 400.0],
            Segment.TEST: [0.0, 0.0],
        },
    )
    runner = _runner(dts, pnl, settings, split, trades=trades)

    report = run_holdout(
        runner, settings, target_name="annual_return",
        significance=False, min_test_bars=20,
    )
    joined = " | ".join(report.warnings)
    assert "零成交" in joined
    assert "根 K 线" in joined
    assert "⚠" in report.summary()


def test_holdout_warns_when_valid_contradicts_train() -> None:
    """选中参数在 VALID 段翻负 —— 样本内两段已经打架，必须留痕。"""
    dts = _dates(150)
    settings = _grid(2)
    split = make_three_way_split(dts, 90, 30, 30, anchor="start")
    pnl = _staged_pnl(
        dts, split,
        {
            Segment.TRAIN: [4_000.0, 100.0],
            Segment.VALID: [-3_000.0, 50.0],     # 选中的那组在 VALID 上翻负
            Segment.TEST: [1_000.0, 1_000.0],
        },
    )
    runner = _runner(dts, pnl, settings, split)
    report = run_holdout(
        runner, settings, target_name="annual_return", significance=False
    )
    assert report.chosen_index == 0
    assert any("VALID" in w for w in report.warnings)


def test_holdout_rejects_bad_inputs() -> None:
    dts = _dates(150)
    settings = _grid(3)
    split = make_three_way_split(dts, 90, 30, 30, anchor="start")
    runner = _runner(dts, np.zeros((150, 3)), settings, split)

    with pytest.raises(ValueError, match="settings"):
        run_holdout(runner, [], significance=False)

    with pytest.raises(TypeError, match="SegmentedRunner"):
        run_holdout(
            FakeRunner(dts, np.zeros((150, 3)), settings),  # type: ignore[arg-type]
            settings, significance=False,
        )


def test_holdout_rejects_out_of_range_selector() -> None:
    dts = _dates(150)
    settings = _grid(3)
    split = make_three_way_split(dts, 90, 30, 30, anchor="start")
    runner = _runner(dts, np.zeros((150, 3)), settings, split)

    def bad_selector(_settings: object, _scores: object) -> int:
        return 99

    with pytest.raises(ValueError, match="越界"):
        run_holdout(
            runner, settings, selector=bad_selector,   # type: ignore[arg-type]
            significance=False,
        )


def test_report_result_rejects_unknown_segment_key() -> None:
    dts = _dates(150)
    settings = _grid(2)
    split = make_three_way_split(dts, 90, 30, 30, anchor="start")
    runner = _runner(dts, np.zeros((150, 2)), settings, split)
    report = run_holdout(
        runner, settings, target_name="annual_return", significance=False
    )
    with pytest.raises(TypeError):
        report.statistics("TEST")           # type: ignore[arg-type]

    report.results.pop(Segment.TEST)
    with pytest.raises(KeyError):
        report.statistics(Segment.TEST)


# ══════════════════════════════════════════════════════════════════════
# 7. 守卫版引擎：挡住直接拿引擎在 TEST 段上扫参数
# ══════════════════════════════════════════════════════════════════════

ENGINE_KWARGS: dict = {
    "vt_symbol": "700.SEHK",
    "interval": Interval.DAILY,
    "rate": 0.0013,
    "slippage": 0.0,
    "size": 1,
    "pricetick": 0.01,
    "capital": 1_000_000,
    "annual_days": ANNUAL_DAYS,
}


def _guarded_engine() -> tuple[SegmentGuardedEngine, ThreeWaySplit]:
    split = make_three_way_split(_dates(300), 180, 60, 60, anchor="start")
    engine = SegmentGuardedEngine()
    engine.bind_split(split)
    return engine, split


def _empty_optimization_setting() -> OptimizationSetting:
    """空的寻优设置：`check_optimization_setting` 会判 False 并直接返回 []。

    这样"放行分支"不需要数据库也不会真跑回测 —— 本组测试要证明的是
    **守卫拦不拦**，不是引擎跑不跑得动。
    """
    return OptimizationSetting()


def test_guarded_engine_blocks_optimization_on_test_segment() -> None:
    engine, _ = _guarded_engine()
    engine.use_segment(Segment.TEST, **ENGINE_KWARGS)
    with pytest.raises(SegmentLeakError, match="TEST"):
        engine.run_bf_optimization(_empty_optimization_setting())
    with pytest.raises(SegmentLeakError, match="TEST"):
        engine.run_ga_optimization(_empty_optimization_setting())
    with pytest.raises(SegmentLeakError, match="TEST"):
        engine.run_optimization(_empty_optimization_setting())


def test_guarded_engine_allows_optimization_in_sample() -> None:
    """TRAIN / VALID 上照常放行（空设置 → 引擎自己返回 []，不碰数据库）。"""
    engine, _ = _guarded_engine()

    engine.use_segment(Segment.TRAIN, **ENGINE_KWARGS)
    assert engine.run_bf_optimization(_empty_optimization_setting()) == []

    engine.use_segment(Segment.VALID, **ENGINE_KWARGS)
    assert engine.run_ga_optimization(_empty_optimization_setting()) == []

    assert engine.segment_warnings == []


def test_guarded_engine_blocks_full_sample_optimization() -> None:
    """全样本窗口同样被拦：它把 TEST 段一起用掉了。"""
    engine, split = _guarded_engine()
    engine.set_parameters(
        start=split.train_start, end=split.test_end, **ENGINE_KWARGS
    )
    with pytest.raises(SegmentLeakError, match="交集"):
        engine.run_bf_optimization(_empty_optimization_setting())


def test_guarded_engine_blocks_window_touching_test_by_one_bar() -> None:
    """只碰到 TEST 段一根 K 线也算越界 —— 边界是严格的。"""
    engine, split = _guarded_engine()
    engine.set_parameters(
        start=split.train_start, end=split.test_start, **ENGINE_KWARGS
    )
    with pytest.raises(SegmentLeakError):
        engine.run_bf_optimization(_empty_optimization_setting())


def test_guarded_engine_warns_when_split_not_bound() -> None:
    """没绑切分 → 不阻塞（既有用法照跑），但必须留痕，不许静默通过。"""
    engine = SegmentGuardedEngine()
    engine.set_parameters(
        start=datetime(2024, 1, 2), end=datetime(2024, 6, 30), **ENGINE_KWARGS
    )
    assert engine.run_bf_optimization(_empty_optimization_setting()) == []
    assert len(engine.segment_warnings) == 1
    assert "未绑定" in engine.segment_warnings[0]


def test_guarded_engine_warns_when_parameters_not_set() -> None:
    engine, _ = _guarded_engine()
    assert engine.run_bf_optimization(_empty_optimization_setting()) == []
    assert any("set_parameters" in w for w in engine.segment_warnings)


def test_use_segment_refuses_hand_written_dates() -> None:
    """窗口只能由切分决定 —— "换了段却忘了改日期"是这一层要消灭的错误。"""
    engine, split = _guarded_engine()
    with pytest.raises(ValueError, match="start"):
        engine.use_segment(
            Segment.TRAIN, start=split.train_start, **ENGINE_KWARGS
        )
    with pytest.raises(ValueError, match="end"):
        engine.use_segment(Segment.TRAIN, end=split.train_end, **ENGINE_KWARGS)
    with pytest.raises(TypeError):
        engine.use_segment("TRAIN", **ENGINE_KWARGS)   # type: ignore[arg-type]


def test_use_segment_sets_the_window_from_the_split() -> None:
    engine, split = _guarded_engine()
    engine.use_segment(Segment.VALID, **ENGINE_KWARGS)
    assert engine.start == split.valid_start
    assert engine.end.date() == split.valid_end.date()
    assert engine.segment is Segment.VALID
    assert engine.current_segment() is Segment.VALID


def test_current_segment_is_none_for_a_foreign_window() -> None:
    engine, split = _guarded_engine()
    assert engine.current_segment() is None          # 还没 set_parameters
    engine.set_parameters(
        start=split.train_start, end=split.valid_end, **ENGINE_KWARGS
    )
    assert engine.current_segment() is None          # 跨两段，不等于任何一段


def test_bind_split_and_require_split_are_strict() -> None:
    engine = SegmentGuardedEngine()
    with pytest.raises(ValueError, match="bind_split"):
        engine.require_split()
    with pytest.raises(TypeError):
        engine.bind_split("split")                   # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bind_split"):
        engine.use_segment(Segment.TRAIN, **ENGINE_KWARGS)


def test_guard_overrides_do_not_pin_the_parent_signature() -> None:
    """两个寻优覆写必须是纯透传（*args/**kwargs）。

    父类的寻优入口还在长参数（gates / collect_returns / ...）。
    覆写一旦抄死签名，新参数就会被【静默】挡在门外 ——
    调用方传了却不生效，比报错更难查。
    """
    for name in ("run_bf_optimization", "run_ga_optimization"):
        params = inspect.signature(getattr(SegmentGuardedEngine, name)).parameters
        kinds = {p.kind for p in params.values()}
        assert inspect.Parameter.VAR_POSITIONAL in kinds, name
        assert inspect.Parameter.VAR_KEYWORD in kinds, name


def test_guard_forwards_every_argument_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """放行时参数原样到达父类，包括本层从未听说过的新参数。"""
    engine, _ = _guarded_engine()
    engine.use_segment(Segment.TRAIN, **ENGINE_KWARGS)

    seen: dict = {}

    def fake(self: object, *args: object, **kwargs: object) -> list:
        seen["args"] = args
        seen["kwargs"] = kwargs
        return ["forwarded"]

    monkeypatch.setattr(BacktestingEngine, "run_bf_optimization", fake)
    setting = _empty_optimization_setting()
    out = engine.run_bf_optimization(setting, output=False, brand_new_param=42)

    assert out == ["forwarded"]
    assert seen["args"] == (setting,)
    assert seen["kwargs"] == {"output": False, "brand_new_param": 42}


def test_guard_runs_before_forwarding(monkeypatch: pytest.MonkeyPatch) -> None:
    """拦下时父类一次都不许被调到 —— 守卫在转发之前。"""
    engine, _ = _guarded_engine()
    engine.use_segment(Segment.TEST, **ENGINE_KWARGS)

    called: list[int] = []

    def fake(self: object, *args: object, **kwargs: object) -> list:
        called.append(1)
        return []

    monkeypatch.setattr(BacktestingEngine, "run_bf_optimization", fake)
    with pytest.raises(SegmentLeakError):
        engine.run_bf_optimization(_empty_optimization_setting())
    assert called == []


# ══════════════════════════════════════════════════════════════════════
# 接入：闸必须能被 import 到，否则没人会走它
# ══════════════════════════════════════════════════════════════════════

def test_segment_api_is_exported_from_the_package() -> None:
    """样本外纪律必须从包顶层可达。

    在此之前 `SegmentedRunner` 除自身测试外零调用点，也不在 `__all__` 里：
    研究者按常规姿势 `set_parameters(全区间)` + `run_bf_optimization` 就
    绕过了 TEST 段禁扫参与查看次数预算这两道闸。import 不到的闸等于没有闸。
    """
    import vnpy_ctastrategy as pkg

    for name in (
        "Segment",
        "SegmentedRunner",
        "SegmentLeakError",
        "SegmentBudgetExhaustedError",
        "ThreeWaySplit",
        "make_three_way_split",
    ):
        assert name in pkg.__all__, f"{name} 不在 __all__ 里"
        assert hasattr(pkg, name), f"{name} 无法从包顶层取到"


def test_exported_segment_types_are_the_same_objects() -> None:
    """导出的必须是同一批对象，不能是同名的另一份。"""
    import vnpy_ctastrategy as pkg
    from vnpy_ctastrategy import segments as seg_module

    assert pkg.SegmentedRunner is seg_module.SegmentedRunner
    assert pkg.Segment is seg_module.Segment
    assert pkg.SegmentLeakError is seg_module.SegmentLeakError
