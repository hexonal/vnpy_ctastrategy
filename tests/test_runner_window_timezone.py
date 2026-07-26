"""`EngineRunner` 的窗口切片必须和数据库查询指向同一段交易日。

病灶
----
`EngineRunner` 有两条取数路径：`cache_bars=True` 时整段 K 线只查一次、之后
按窗口在内存里切片；切不出东西才回退到 `engine.load_data()`。查询那一侧走
`backtesting.load_bar_data`，它用 `query_window.localize_bound` 把裸边界读成
**交易所墙钟**；而切片那一侧原先用 `replace(tzinfo=None)` 把两边都剥成裸值再比。

剥时区不是"忽略时区"，是"把 DB_TZ 的 K 线时间戳当成交易所墙钟读"。
本项目 `database.timezone = UTC`，港股日线因此存成前一日 16:00Z：

    港股 2024-01-26 这根  →  数据库交回 2024-01-25 16:00:00+00:00

于是 `start=2024-01-26` 的窗口，查询给 11 根，切片只留 10 根 ——
**低端丢掉窗口的第一个交易日**。同一个错位在高端反向发作：窗口 `end` 之后
那一根（次日 00:00 HKT = 当日 16:00Z）被切片当成"还在今天"收了进来。
Walk-Forward 的每一折因此整体后移一个交易日，训练窗吃掉测试窗的第一根 =
前视泄漏。

这些断言在裸边界（用户手打的日期）与带时区边界（`bar_datetimes()` 交回的
真实 K 线时间戳）上都必须成立，所以两种形状各测一遍。
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from pandas import DataFrame
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import DB_TZ, convert_tz
from vnpy.trader.object import BarData

from vnpy_ctastrategy import backtesting as backtesting_module
from vnpy_ctastrategy.backtesting import BacktestingEngine, load_bar_data
from vnpy_ctastrategy.overfitting import EngineRunner
from vnpy_ctastrategy.template import CtaTemplate

HK_TZ = ZoneInfo("Asia/Hong_Kong")
SYMBOL = "TZWIN"
EXCHANGE = Exchange.SEHK
VT_SYMBOL = f"{SYMBOL}.{EXCHANGE.value}"
CAPITAL = 1_000_000

FIRST_SESSION = datetime(2024, 1, 2, tzinfo=HK_TZ)
BAR_COUNT = 60

# 窗口两端都挑在周内：低端 2024-01-26 是周五，它的前一根是 01-25（周四），
# 高端 02-07 是周三，它的后一根是 02-08（周四）—— 相邻两根隔一个自然日，
# 正是"剥时区后 16:00Z 到 23:59:59 之间还塞得下一根"的那种排列。
WINDOW_START = datetime(2024, 1, 26)
WINDOW_END = datetime(2024, 2, 7)


class Idle(CtaTemplate):
    """不下单。本文件测的是喂进引擎的 K 线范围，与策略行为无关。"""

    def on_init(self) -> None:
        self.inited = True

    def on_start(self) -> None:
        self.trading = True

    def on_bar(self, bar: BarData) -> None:
        """不下单。"""


def build_bars() -> list[BarData]:
    """一交易日一根，时间戳 = 交易所墙钟的 00:00（gateway 写库时的形状）。"""
    bars: list[BarData] = []
    moment: datetime = FIRST_SESSION
    while len(bars) < BAR_COUNT:
        if moment.weekday() < 5:
            price: float = 100.0 + len(bars)
            bars.append(BarData(
                symbol=SYMBOL,
                exchange=EXCHANGE,
                datetime=moment,
                interval=Interval.DAILY,
                volume=1000.0,
                turnover=1000.0 * price,
                open_interest=0.0,
                open_price=price,
                high_price=price,
                low_price=price,
                close_price=price,
                gateway_name="TEST",
            ))
        moment += timedelta(days=1)
    return bars


class FakeDatabase:
    """`BaseDatabase.load_bar_data` 里被用到的那一片。

    关键在**交回**的形状：真实驱动把时间戳存成 `convert_tz` 后的裸 DB_TZ 值，
    读出来再贴回 DB_TZ。所以港股 00:00 HKT 的那根，交回时是前一日 16:00Z。
    这个转换就是本文件要钉的错位的来源，双写在这里而不是用 HKT 直接交回。
    """

    def __init__(self, bars: list[BarData]) -> None:
        self.bars = bars

    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> list[BarData]:
        low: datetime = convert_tz(start)
        high: datetime = convert_tz(end)
        out: list[BarData] = []
        for bar in self.bars:
            if bar.symbol != symbol or bar.exchange != exchange:
                continue
            if bar.interval != interval:
                continue
            stored: datetime = convert_tz(bar.datetime)
            if low <= stored <= high:
                row = BarData(
                    symbol=bar.symbol,
                    exchange=bar.exchange,
                    datetime=stored.replace(tzinfo=DB_TZ),
                    interval=bar.interval,
                    volume=bar.volume,
                    turnover=bar.turnover,
                    open_interest=bar.open_interest,
                    open_price=bar.open_price,
                    high_price=bar.high_price,
                    low_price=bar.low_price,
                    close_price=bar.close_price,
                    gateway_name=bar.gateway_name,
                )
                out.append(row)
        return out


@pytest.fixture
def bars(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[BarData]]:
    """装上数据库替身。`load_bar_data` 有 lru_cache，进出都要清。"""
    series: list[BarData] = build_bars()
    monkeypatch.setattr(backtesting_module, "get_database", lambda: FakeDatabase(series))
    monkeypatch.setattr(BacktestingEngine, "output", lambda self, msg: None)
    load_bar_data.cache_clear()
    yield series
    load_bar_data.cache_clear()


def make_runner(
    start: datetime,
    end: datetime,
    cache_bars: bool = True,
    warmup_bars: int = 0,
) -> EngineRunner:
    return EngineRunner(
        strategy_class=Idle,
        vt_symbol=VT_SYMBOL,
        interval=Interval.DAILY,
        rate=0.0,
        slippage=0.0,
        size=1,
        pricetick=0.01,
        capital=CAPITAL,
        start=start,
        end=end,
        cache_bars=cache_bars,
        warmup_bars=warmup_bars,
    )


def fed_bars(runner: EngineRunner, start: datetime, end: datetime) -> list[datetime]:
    """引擎在这个窗口里实际收到的 K 线时间戳。"""
    seen: list[datetime] = []
    original = BacktestingEngine.run_backtesting

    def spy(self: BacktestingEngine) -> None:
        seen.extend(bar.datetime for bar in self.history_data)
        original(self)

    BacktestingEngine.run_backtesting = spy       # type: ignore[method-assign]
    try:
        runner({}, start, end)
    finally:
        BacktestingEngine.run_backtesting = original    # type: ignore[method-assign]
    return seen


def queried(start: datetime, end: datetime) -> list[datetime]:
    """同一个窗口，数据库那一侧的答案（`load_bar_data` 的口径）。"""
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=VT_SYMBOL, interval=Interval.DAILY, start=start, end=end,
        rate=0.0, slippage=0.0, size=1, pricetick=0.01, capital=CAPITAL,
    )
    return [
        bar.datetime
        for bar in load_bar_data(SYMBOL, EXCHANGE, Interval.DAILY, engine.start, engine.end)
    ]


# ── 低端：窗口的第一个交易日不许被切掉 ────────────────────────────────

def test_cached_slice_keeps_every_bar_the_query_returns(bars: list[BarData]) -> None:
    """整段缓存 + 切片，必须和"直接查这个窗口"给出同一批 K 线。"""
    runner = make_runner(WINDOW_START, WINDOW_END)

    assert fed_bars(runner, WINDOW_START, WINDOW_END) == queried(WINDOW_START, WINDOW_END)


def test_first_session_of_the_window_is_fed(bars: list[BarData]) -> None:
    """窗口低端那一天的 K 线必须进引擎（它就是 start 指名的那一天）。"""
    runner = make_runner(WINDOW_START, WINDOW_END)
    first: datetime = fed_bars(runner, WINDOW_START, WINDOW_END)[0]

    assert first.astimezone(HK_TZ).date() == WINDOW_START.date()


# ── 高端：窗口结束之后那一根不许被收进来（Walk-Forward 的前视泄漏）────

def test_window_does_not_eat_the_session_after_end(bars: list[BarData]) -> None:
    """缓存里有 `end` 之后的 K 线时，切片也不许把它算进本窗口。"""
    runner = make_runner(bars[0].datetime, bars[-1].datetime)   # 缓存覆盖全样本
    last: datetime = fed_bars(runner, WINDOW_START, WINDOW_END)[-1]

    assert last.astimezone(HK_TZ).date() == WINDOW_END.date()


def test_adjacent_windows_do_not_overlap(bars: list[BarData]) -> None:
    """相邻两折的 K 线集合必须无交集 —— 折边界重叠 = 训练窗看到测试窗。"""
    runner = make_runner(bars[0].datetime, bars[-1].datetime)
    dts: list[datetime] = runner.bar_datetimes()

    train = set(fed_bars(runner, dts[5], dts[14]))
    test = set(fed_bars(runner, dts[15], dts[24]))

    assert train & test == set()
    assert len(train) == 10
    assert len(test) == 10


# ── 两条取数路径必须交出同一段 ────────────────────────────────────────

def test_cache_and_load_data_paths_agree(bars: list[BarData]) -> None:
    """`cache_bars` 只是快慢开关，不许改变绩效。"""
    cached = make_runner(WINDOW_START, WINDOW_END, cache_bars=True)
    direct = make_runner(WINDOW_START, WINDOW_END, cache_bars=False)

    assert fed_bars(cached, WINDOW_START, WINDOW_END) == fed_bars(
        direct, WINDOW_START, WINDOW_END
    )


# ── 预热段的剔除也要落在同一个时钟上 ──────────────────────────────────

def test_warmup_does_not_change_the_reported_days(bars: list[BarData]) -> None:
    """预热跑完后剔除的只能是预热段，窗口自己的第一天必须留下。

    `daily_df` 的索引是 `bar.datetime.date()`，即 **K 线自己那个时钟**的日期；
    按调用方墙钟的 `start.date()` 去裁，会把窗口首日一起裁掉。所以断言写成
    "开不开预热，报出来的日子一模一样" —— 这个不变量不依赖任何一个时钟。
    """
    warmed: DataFrame = make_runner(
        bars[0].datetime, bars[-1].datetime, warmup_bars=5
    )({}, WINDOW_START, WINDOW_END)
    plain: DataFrame = make_runner(
        bars[0].datetime, bars[-1].datetime, warmup_bars=0
    )({}, WINDOW_START, WINDOW_END)

    assert list(warmed.index) == list(plain.index)
    assert len(plain.index) == len(queried(WINDOW_START, WINDOW_END))


def test_warmup_actually_prepends_bars(bars: list[BarData]) -> None:
    """预热真的多喂了 K 线（否则上一个断言可能是"预热没生效"蒙对的）。"""
    warmed = make_runner(bars[0].datetime, bars[-1].datetime, warmup_bars=5)
    plain = make_runner(bars[0].datetime, bars[-1].datetime, warmup_bars=0)

    assert len(fed_bars(warmed, WINDOW_START, WINDOW_END)) == len(
        fed_bars(plain, WINDOW_START, WINDOW_END)
    ) + 5
