"""审查窗口的交易日筛选必须和取数走同一个时钟。

`run_audit` 拿 `[d for d in runner.bar_datetimes() if _within(d, start, end)]`
决定"这次审查用哪些交易日"，Walk-Forward 的每一折边界都从这份清单上切。
`_within` 原先把两侧都 `replace(tzinfo=None)` 再比 —— 与
`overfitting.EngineRunner` 那个已修的病灶同源：K 线时间戳带 DB_TZ，
调用方的 `start` / `end` 是裸的交易所墙钟，剥时区等于把前者当后者读。

本项目 `database.timezone = UTC`，港股日线因此存成前一日 16:00Z：

    港股 2024-01-26 这根  →  数据库交回 2024-01-25 16:00:00+00:00

于是 `start=2024-01-26` 的审查窗口漏掉自己的第一个交易日，而 `end` 之后
那一根（次日 00:00 HKT = 当日 16:00Z）反被算作"还在 end 当天"收进来 ——
审查区间整体后移一个交易日，且越过了调用方声明的终点。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from vnpy.trader.constant import Exchange
from vnpy.trader.database import DB_TZ, convert_tz

from vnpy_ctastrategy.overfitting_audit import _within

HK_TZ = ZoneInfo("Asia/Hong_Kong")
EXCHANGE = Exchange.SEHK

WINDOW_START = datetime(2024, 1, 26)
WINDOW_END = datetime(2024, 2, 7)


def stored(session: datetime) -> datetime:
    """交易日的 00:00（交易所墙钟）→ 数据库交回来的样子（DB_TZ）。"""
    return convert_tz(session.replace(tzinfo=HK_TZ)).replace(tzinfo=DB_TZ)


def sessions(first: datetime, count: int) -> list[datetime]:
    """连续 `count` 个工作日的 K 线时间戳，按数据库交回的形状。"""
    out: list[datetime] = []
    moment = first
    while len(out) < count:
        if moment.weekday() < 5:
            out.append(stored(moment))
        moment += timedelta(days=1)
    return out


# ── 低端：窗口第一天必须算在窗口里 ────────────────────────────────────

def test_first_session_of_the_window_is_inside() -> None:
    assert _within(stored(WINDOW_START), WINDOW_START, WINDOW_END, EXCHANGE)


def test_session_before_the_window_stays_outside() -> None:
    assert not _within(stored(datetime(2024, 1, 25)), WINDOW_START, WINDOW_END, EXCHANGE)


# ── 高端：end 之后那一根不许被收进来 ──────────────────────────────────

def test_last_session_of_the_window_is_inside() -> None:
    assert _within(stored(WINDOW_END), WINDOW_START, WINDOW_END, EXCHANGE)


def test_session_after_the_window_stays_outside() -> None:
    assert not _within(stored(datetime(2024, 2, 8)), WINDOW_START, WINDOW_END, EXCHANGE)


# ── 整段筛选：根数与端点都要对 ────────────────────────────────────────

def test_selected_sessions_match_the_declared_window() -> None:
    """2024-01-26 ~ 02-07 之间共 9 个工作日（含两端）。"""
    every = sessions(datetime(2024, 1, 15), 30)
    picked = [d for d in every if _within(d, WINDOW_START, WINDOW_END, EXCHANGE)]

    assert [d.astimezone(HK_TZ).date() for d in picked] == [
        datetime(2024, 1, day).date()
        for day in (26, 29, 30, 31)
    ] + [
        datetime(2024, 2, day).date()
        for day in (1, 2, 5, 6, 7)
    ]


# ── 拿不到交易所时退回旧口径（既有假 runner 的形状）──────────────────

def test_without_exchange_falls_back_to_naive_comparison() -> None:
    """K 线时间戳与边界同钟（都是裸值）时，不传交易所也必须正确。"""
    naive = [datetime(2024, 1, 20) + timedelta(days=i) for i in range(30)]
    picked = [d for d in naive if _within(d, WINDOW_START, WINDOW_END)]

    assert picked[0] == WINDOW_START
    assert picked[-1] == WINDOW_END
