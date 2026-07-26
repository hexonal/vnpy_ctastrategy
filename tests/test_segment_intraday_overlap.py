"""三段切分在【日内周期】下必须真的不重叠。

`ThreeWaySplit.__post_init__` 用原始时间戳校验 `train_end < valid_start <
test_start`，这在切分对象上成立。但执行侧把窗口上界撑到了当日 23:59:59
（`overfitting.EngineRunner.__call__` 一次，`BacktestingEngine.set_parameters`
再一次），于是在日内周期下：

    VALID 声明 [10:00, 10:14]  →  引擎实收 [10:00, 23:59:59]

TEST 段当天剩下的 K 线全部落进 VALID 的绩效里。日线路径不受影响（一天一根，
撑到 23:59:59 仍是同一根），所以既有测试全绿也说明不了问题 —— 本文件专门
用 15 分钟线把这条路径钉住。

`SegmentedRunner` 的两道纪律（TEST 段禁扫参、TEST 查看次数预算）保护的是
"什么时候可以看测试段"；这里保护的是"看的时候拿到的是不是那一段"。
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta

import pytest
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData

from vnpy_ctastrategy import backtesting as backtesting_module
from vnpy_ctastrategy import overfitting as overfitting_module
from vnpy_ctastrategy.backtesting import BacktestingEngine
from vnpy_ctastrategy.overfitting import EngineRunner
from vnpy_ctastrategy.segments import (
    Segment,
    SegmentedRunner,
    make_three_way_split,
)
from vnpy_ctastrategy.template import CtaTemplate

SYMBOL = "SEGINTRA"
EXCHANGE = Exchange.SEHK
VT_SYMBOL = f"{SYMBOL}.{EXCHANGE.value}"
CAPITAL = 1_000_000


class Idle(CtaTemplate):
    """交易与否无关紧要 —— 本文件测的是喂进引擎的 K 线范围。"""

    def on_init(self) -> None:
        self.inited = True

    def on_start(self) -> None:
        self.trading = True

    def on_bar(self, bar: BarData) -> None:
        """不下单：本测试只关心 history_data 的范围。"""


def intraday_bars(n_days: int, per_day: int) -> list[BarData]:
    """每天 `per_day` 根 15 分钟线，从 09:30 起。"""
    bars: list[BarData] = []
    day = datetime(2024, 1, 2, 9, 30)
    for d in range(n_days):
        base = day + timedelta(days=d)
        for i in range(per_day):
            moment = base + timedelta(minutes=15 * i)
            price = 100.0 + len(bars) * 0.01
            bars.append(BarData(
                symbol=SYMBOL,
                exchange=EXCHANGE,
                datetime=moment,
                interval=Interval.MINUTE,
                volume=1000.0,
                turnover=1000.0 * price,
                open_interest=0.0,
                open_price=price,
                high_price=price,
                low_price=price,
                close_price=price,
                gateway_name="TEST",
            ))
    return bars


@pytest.fixture()
def bars(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[BarData]]:
    series = intraday_bars(n_days=4, per_day=16)

    def fake_load_bar_data(
        symbol: str, exchange: Exchange, interval: Interval,
        start: datetime, end: datetime,
    ) -> list[BarData]:
        return [bar for bar in series if start <= bar.datetime <= end]

    monkeypatch.setattr(backtesting_module, "load_bar_data", fake_load_bar_data)
    monkeypatch.setattr(overfitting_module, "load_bar_data", fake_load_bar_data, raising=False)
    monkeypatch.setattr(BacktestingEngine, "output", lambda self, msg: None)
    yield series


def make_runner(series: list[BarData]) -> EngineRunner:
    return EngineRunner(
        strategy_class=Idle,
        vt_symbol=VT_SYMBOL,
        interval=Interval.MINUTE,
        rate=0.0,
        slippage=0.0,
        size=1,
        pricetick=0.01,
        capital=CAPITAL,
        start=series[0].datetime,
        end=series[-1].datetime,
        cache_bars=True,
    )


def fed_bars(runner: EngineRunner, start: datetime, end: datetime) -> list[datetime]:
    """引擎在这个窗口里实际收到的 K 线时间戳。"""
    seen: list[datetime] = []
    original = BacktestingEngine.run_backtesting

    def spy(self: BacktestingEngine) -> None:
        seen.extend(bar.datetime for bar in self.history_data)
        original(self)

    BacktestingEngine.run_backtesting = spy          # type: ignore[method-assign]
    try:
        runner({}, start, end)
    finally:
        BacktestingEngine.run_backtesting = original  # type: ignore[method-assign]
    return seen


def test_valid_window_does_not_reach_into_the_test_segment(
    bars: list[BarData],
) -> None:
    """核心断言：VALID 段跑回测时，一根 TEST 段的 K 线都不许进来。"""
    dts = [bar.datetime for bar in bars]
    split = make_three_way_split(dts, train_bars=20, valid_bars=20, test_bars=20, anchor="start")
    runner = make_runner(bars)

    valid_start, valid_end = split.as_period(Segment.VALID)
    test_start, _test_end = split.as_period(Segment.TEST)

    seen = fed_bars(runner, valid_start, valid_end)

    leaked = [dt for dt in seen if dt >= test_start]
    assert leaked == [], f"VALID 窗口吃进了 {len(leaked)} 根 TEST 段 K 线"


def test_train_window_does_not_reach_into_later_segments(
    bars: list[BarData],
) -> None:
    dts = [bar.datetime for bar in bars]
    split = make_three_way_split(dts, train_bars=20, valid_bars=20, test_bars=20, anchor="start")
    runner = make_runner(bars)

    train_start, train_end = split.as_period(Segment.TRAIN)
    valid_start, _ = split.as_period(Segment.VALID)

    seen = fed_bars(runner, train_start, train_end)

    assert [dt for dt in seen if dt >= valid_start] == []


def test_each_segment_gets_exactly_the_bars_it_declares(
    bars: list[BarData],
) -> None:
    """声明多少根就喂多少根 —— 多喂即重叠，少喂即窗口算错。"""
    dts = [bar.datetime for bar in bars]
    split = make_three_way_split(dts, train_bars=20, valid_bars=20, test_bars=20, anchor="start")
    runner = make_runner(bars)

    for segment in (Segment.TRAIN, Segment.VALID, Segment.TEST):
        start, end = split.as_period(segment)
        seen = fed_bars(runner, start, end)
        assert len(seen) == split.bars(segment), (
            f"{segment.name} 声明 {split.bars(segment)} 根，实收 {len(seen)} 根"
        )


def test_the_three_segments_share_no_bar(bars: list[BarData]) -> None:
    """三段实收 K 线的交集必须为空 —— 这正是 ThreeWaySplit 承诺的那句话。"""
    dts = [bar.datetime for bar in bars]
    split = make_three_way_split(dts, train_bars=20, valid_bars=20, test_bars=20, anchor="start")
    runner = make_runner(bars)

    seen_by_segment: dict[Segment, set[datetime]] = {}
    for segment in (Segment.TRAIN, Segment.VALID, Segment.TEST):
        start, end = split.as_period(segment)
        seen_by_segment[segment] = set(fed_bars(runner, start, end))

    train, valid, test = (
        seen_by_segment[Segment.TRAIN],
        seen_by_segment[Segment.VALID],
        seen_by_segment[Segment.TEST],
    )
    assert train & valid == set()
    assert valid & test == set()
    assert train & test == set()


def test_daily_segments_are_unaffected(monkeypatch: pytest.MonkeyPatch) -> None:
    """对照组：日线路径本来就是干净的，修复不得改变它。"""
    series: list[BarData] = []
    for i in range(120):
        moment = datetime(2024, 1, 1) + timedelta(days=i)
        price = 100.0 + i
        series.append(BarData(
            symbol=SYMBOL, exchange=EXCHANGE, datetime=moment,
            interval=Interval.DAILY, volume=1000.0, turnover=1000.0 * price,
            open_interest=0.0, open_price=price, high_price=price,
            low_price=price, close_price=price, gateway_name="TEST",
        ))

    def fake_load_bar_data(
        symbol: str, exchange: Exchange, interval: Interval,
        start: datetime, end: datetime,
    ) -> list[BarData]:
        return [bar for bar in series if start <= bar.datetime <= end]

    monkeypatch.setattr(backtesting_module, "load_bar_data", fake_load_bar_data)
    monkeypatch.setattr(BacktestingEngine, "output", lambda self, msg: None)

    dts = [bar.datetime for bar in series]
    split = make_three_way_split(dts, train_bars=60, valid_bars=30, test_bars=30)
    runner = EngineRunner(
        strategy_class=Idle, vt_symbol=VT_SYMBOL, interval=Interval.DAILY,
        rate=0.0, slippage=0.0, size=1, pricetick=0.01, capital=CAPITAL,
        start=series[0].datetime, end=series[-1].datetime, cache_bars=True,
    )

    for segment in (Segment.TRAIN, Segment.VALID, Segment.TEST):
        start, end = split.as_period(segment)
        seen = fed_bars(runner, start, end)
        assert len(seen) == split.bars(segment)


def test_segmented_runner_reports_the_declared_bar_count(
    bars: list[BarData],
) -> None:
    """经 SegmentedRunner 走一遍：statistics 的 total_days 不得含别段的数据。"""
    dts = [bar.datetime for bar in bars]
    split = make_three_way_split(dts, train_bars=20, valid_bars=20, test_bars=20, anchor="start")
    seg_runner = SegmentedRunner(
        runner=make_runner(bars), split=split, capital=float(CAPITAL)
    )

    valid = seg_runner.run({}, Segment.VALID)
    test_start, _ = split.as_period(Segment.TEST)

    assert valid.end < test_start
    assert not valid.daily_df.empty
