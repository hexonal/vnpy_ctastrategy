"""TurtleSignalStrategy 的 2N 申报止损。

自带策略里唯一在实盘跑的那条（`~/.vntrader/cta_strategy_setting.json` 里的
`aa`，NBIS.SMART）。它本来就有 2N 止损的概念，只是只在 `on_trade` 之后才算
出来 —— 而风控闸要在下单**之前**看到止损。这里钉住「下单前用报价代入同一
个公式」这条等价关系，以及 ATR 还没算出来时的诚实拒绝。
"""

from __future__ import annotations

from vnpy.trader.constant import Direction

from vnpy_ctastrategy.strategies.turtle_signal_strategy import TurtleSignalStrategy


def _strategy(atr: float) -> TurtleSignalStrategy:
    strategy = TurtleSignalStrategy.__new__(TurtleSignalStrategy)
    strategy.atr_value = atr
    return strategy


def test_long_stop_is_two_atr_below_the_order_price() -> None:
    strategy = _strategy(1.5)

    assert strategy.get_stop_price("NBIS.SMART", Direction.LONG, 42.5) == 39.5


def test_short_stop_is_two_atr_above_the_order_price() -> None:
    strategy = _strategy(1.5)

    assert strategy.get_stop_price("NBIS.SMART", Direction.SHORT, 42.5) == 45.5


def test_it_matches_the_stop_on_trade_records_when_the_fill_is_at_the_quote() -> None:
    """`on_trade` 写的是 `long_entry - 2 * atr_value`；成交价等于报价时两者必须一致，
    否则风控核算用的止损与事后记录的止损是两个数。"""
    strategy = _strategy(1.5)
    price: float = 42.5

    declared = strategy.get_stop_price("NBIS.SMART", Direction.LONG, price)
    recorded = price - 2 * strategy.atr_value

    assert declared == recorded


def test_no_stop_is_declared_before_atr_has_a_value() -> None:
    """ArrayManager 未 inited（或刚清仓、ATR 尚未重算）时 N 是 0，止损会等于
    入场价 —— 那不是止损，`check_stop_side` 也会以「不提供保护」拒掉。
    如实返回 None，让委托被拒得有理由可查。"""
    assert _strategy(0.0).get_stop_price("NBIS.SMART", Direction.LONG, 42.5) is None
    assert _strategy(-1.0).get_stop_price("NBIS.SMART", Direction.LONG, 42.5) is None


def test_the_declared_stop_is_on_the_protective_side_of_the_price() -> None:
    """`check_stop_side` 会拒掉「多单止损不低于委托价」的申报。"""
    strategy = _strategy(1.5)
    price: float = 42.5

    long_stop = strategy.get_stop_price("NBIS.SMART", Direction.LONG, price)
    short_stop = strategy.get_stop_price("NBIS.SMART", Direction.SHORT, price)

    assert long_stop is not None and long_stop < price
    assert short_stop is not None and short_stop > price
