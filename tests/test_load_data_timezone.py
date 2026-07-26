"""``BacktestingEngine.load_data`` must return every stored bar, on any machine.

Three independent defects made it return fewer bars than the database holds,
and made the chunked loader disagree with a single query over the same window:

1. **Naive bounds were read as machine-local time.** Every database driver
   funnels query bounds through ``vnpy.trader.database.convert_tz``, which
   calls ``datetime.astimezone()``; on a naive datetime that method reads the
   value as the *host's* local zone. A window typed as HK dates therefore
   slides by the machine's UTC offset, and the bars sitting on the window
   edges fall out. Measured against this project's QuestDB (host in US
   Eastern, ``database.timezone`` = UTC) on a 700-bar daily series: 699 bars
   from one query, 693 through ``load_data``, and **0** from a naive
   single-day window over a day that has a bar.

2. **The chunk loop skipped the gap between chunks.** Chunk queries are
   inclusive at both ends, and the loop advanced with
   ``start = end + interval_delta``. Every bar whose timestamp fell strictly
   inside ``(end, end + interval_delta)`` — i.e. every bar not labelled
   exactly on a chunk boundary — was queried by neither chunk. This one is
   pure arithmetic and bites even with perfectly tz-aware bounds.

3. **Windows shorter than a calendar day raised ZeroDivisionError**, because
   ``progress_days / total_days`` never guarded ``total_days == 0``.

The tests below run against an in-memory database double rather than a live
server, so they fail on any host and need no QuestDB. The double is faithful
where it matters: it filters with the very ``convert_tz`` the real drivers
call, so defect 1 reproduces through the same code path the drivers use.
``TZ`` is pinned to US Pacific so the host offset is a known non-zero value
instead of whatever the machine or CI runner happens to be set to.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import DB_TZ, convert_tz
from vnpy.trader.object import BarData
from vnpy_gatewaykit.query_window import localize_bound, query_tz

from vnpy_ctastrategy import backtesting as backtesting_module
from vnpy_ctastrategy.backtesting import BacktestingEngine, load_bar_data

HK_TZ = ZoneInfo("Asia/Hong_Kong")
SYMBOL = "TZBUG"
EXCHANGE = Exchange.SEHK
VT_SYMBOL = f"{SYMBOL}.{EXCHANGE.value}"
INTERVAL = Interval.DAILY

# Enough bars for load_data to split the window into its usual ten chunks, so
# the seam arithmetic gets exercised nine times rather than once.
BAR_COUNT = 700
FIRST_SESSION = datetime(2023, 1, 2, tzinfo=HK_TZ)

pytestmark = pytest.mark.skipif(
    not hasattr(time, "tzset"),
    reason="test pins the host timezone via TZ, which needs time.tzset (POSIX)",
)


class FakeDatabase:
    """The slice of ``BaseDatabase`` ``load_bar_data`` touches.

    Filtering happens in ``convert_tz`` space — the same normalisation every
    real driver applies to both the query bounds and the stored timestamps —
    so a bound this double accepts is a bound QuestDB/SQLite/MySQL accept too.
    Bounds are inclusive at both ends, matching the drivers' SQL.
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
        return [
            bar for bar in self.bars
            if bar.symbol == symbol
            and bar.exchange == exchange
            and bar.interval == interval
            and low <= convert_tz(bar.datetime) <= high
        ]


def build_bars() -> list[BarData]:
    """One bar per weekday, stamped 00:00 in the exchange's own timezone.

    That is the shape gateways write: ``vnpy_gatewaykit.market_clock`` attaches
    the market zone to the feed's wall-clock string, so an HK daily bar is
    midnight Hong Kong — 16:00 UTC the previous day, which is exactly why a
    UTC-or-machine-local reading of a naive bound lands on the wrong side of it.
    """
    bars: list[BarData] = []
    moment: datetime = FIRST_SESSION
    while len(bars) < BAR_COUNT:
        if moment.weekday() < 5:
            price: float = 100.0 + len(bars)
            bars.append(BarData(
                symbol=SYMBOL,
                exchange=EXCHANGE,
                datetime=moment,
                interval=INTERVAL,
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


@pytest.fixture
def bars(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[BarData]]:
    """Install the double, and pin the host zone away from the market's.

    ``load_bar_data`` is ``lru_cache``d, so the cache has to be dropped on both
    sides of the test: entering, so a previous test's double cannot answer;
    leaving, so this double's rows cannot outlive it.
    """
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()

    series: list[BarData] = build_bars()
    monkeypatch.setattr(backtesting_module, "get_database", lambda: FakeDatabase(series))
    load_bar_data.cache_clear()
    yield series
    load_bar_data.cache_clear()
    monkeypatch.undo()
    time.tzset()


def make_engine(start: datetime, end: datetime) -> BacktestingEngine:
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=VT_SYMBOL,
        interval=INTERVAL,
        start=start,
        end=end,
        rate=0.0,
        slippage=0.0,
        size=1,
        pricetick=0.01,
        capital=100_000,
    )
    engine.output = lambda msg: None            # type: ignore[method-assign]
    return engine


def chunked(start: datetime, end: datetime) -> list[BarData]:
    engine: BacktestingEngine = make_engine(start, end)
    engine.load_data()
    return list(engine.history_data)


def single(start: datetime, end: datetime) -> list[BarData]:
    return load_bar_data(SYMBOL, EXCHANGE, INTERVAL, start, end)


def both_ways(start: datetime, end: datetime) -> tuple[list[BarData], list[BarData]]:
    """The chunked and the single answer to *the same* question.

    ``set_parameters`` stretches ``end`` to 23:59:59 of the given day, so the
    single query has to be asked about ``engine.end`` rather than the caller's
    ``end`` — otherwise the two are being asked about different windows and any
    difference between them says nothing about the chunk loop.
    """
    engine: BacktestingEngine = make_engine(start, end)
    engine.load_data()
    return list(engine.history_data), single(engine.start, engine.end)


# ── the host really is on a different clock than the market ──────────────────

def test_host_timezone_differs_from_market(bars: list[BarData]) -> None:
    """Guard the premise: with host == market zone every assertion below is vacuous."""
    host_offset = datetime.now().astimezone().utcoffset()
    market_offset = FIRST_SESSION.utcoffset()
    assert host_offset != market_offset


# ── defect 1: naive bounds must mean the market's wall clock ─────────────────

def test_single_query_with_naive_bounds_returns_every_bar(bars: list[BarData]) -> None:
    """A naive bound is the market's wall clock, not the host's."""
    start: datetime = bars[0].datetime.replace(tzinfo=None)
    end: datetime = bars[-1].datetime.replace(tzinfo=None)

    assert len(single(start, end)) == BAR_COUNT


def test_naive_single_day_window_returns_that_days_bar(bars: list[BarData]) -> None:
    """The tight query that used to return nothing at all."""
    session: datetime = bars[BAR_COUNT // 2].datetime.replace(tzinfo=None)

    found: list[BarData] = single(session, session.replace(hour=23, minute=59))

    assert [bar.datetime for bar in found] == [bars[BAR_COUNT // 2].datetime]


def test_load_data_over_a_single_day_does_not_divide_by_zero(
    bars: list[BarData],
) -> None:
    """``total_days`` is 0 for a same-day window; the progress maths must survive."""
    session: datetime = bars[BAR_COUNT // 2].datetime.replace(tzinfo=None)

    assert chunked(session, session) == [bars[BAR_COUNT // 2]]


# ── defect 2: chunked and single must agree, and both must be complete ───────

def test_chunked_matches_single_and_the_database_naive_bounds(
    bars: list[BarData],
) -> None:
    """The headline invariant, over the full stored range."""
    start: datetime = bars[0].datetime.replace(tzinfo=None)
    end: datetime = bars[-1].datetime.replace(tzinfo=None)

    from_chunks, from_single = both_ways(start, end)

    assert len(from_chunks) == len(from_single) == BAR_COUNT
    assert [bar.datetime for bar in from_chunks] == [bar.datetime for bar in bars]


def test_chunked_matches_single_when_bounds_straddle_bar_timestamps(
    bars: list[BarData],
) -> None:
    """Chunk seams must not fall between two bars.

    UTC midnight sits 8 hours after the HK-midnight bar label, so no chunk
    boundary coincides with a bar. Every seam is then a chance to drop the bar
    that follows it — which is what ``start = end + interval_delta`` did. This
    case is pure arithmetic: the bounds are explicit instants, so it fails even
    with the timezone reading already correct.
    """
    utc = ZoneInfo("UTC")
    start: datetime = bars[0].datetime.astimezone(utc).replace(hour=0, minute=0)
    end: datetime = bars[-1].datetime.astimezone(utc).replace(hour=0, minute=0)

    from_chunks, from_single = both_ways(start, end)

    assert [bar.datetime for bar in from_chunks] == [bar.datetime for bar in from_single]
    assert len(from_single) > BAR_COUNT * 0.9      # the window really is nearly full


def test_chunk_seams_produce_no_duplicates(bars: list[BarData]) -> None:
    """Overlapping inclusive chunks must not double-count the bar on the seam."""
    start: datetime = bars[0].datetime.replace(tzinfo=None)
    end: datetime = bars[-1].datetime.replace(tzinfo=None)

    stamps: list[datetime] = [bar.datetime for bar in chunked(start, end)]

    assert len(stamps) == len(set(stamps))
    assert stamps == sorted(stamps)


def test_aware_bounds_are_left_alone(bars: list[BarData]) -> None:
    """An explicit instant is already unambiguous; nothing may reinterpret it."""
    start: datetime = bars[0].datetime
    end: datetime = bars[-1].datetime

    assert len(single(start, end)) == BAR_COUNT
    assert len(chunked(start, end)) == BAR_COUNT


# ── the bound-localising helper itself ───────────────────────────────────────

def test_localize_bound_attaches_the_market_zone() -> None:
    moment: datetime = datetime(2024, 1, 26, 9, 30)

    localized: datetime = localize_bound(moment, Exchange.SEHK)

    assert localized.utcoffset() == HK_TZ.utcoffset(moment)
    assert localized.replace(tzinfo=None) == moment


def test_localize_bound_is_idempotent_on_aware_input() -> None:
    moment: datetime = datetime(2024, 1, 26, 9, 30, tzinfo=ZoneInfo("UTC"))

    assert localize_bound(moment, Exchange.SEHK) is moment


def test_unmapped_exchange_falls_back_to_the_configured_database_zone() -> None:
    """Upstream markets ``market_clock`` does not map still get a declared zone."""
    assert query_tz(Exchange.CFFEX) is DB_TZ
    assert localize_bound(datetime(2024, 1, 26), Exchange.CFFEX).tzinfo is DB_TZ
