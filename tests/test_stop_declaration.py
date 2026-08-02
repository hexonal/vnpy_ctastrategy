"""CTA 委托的申报止损 —— 钉住「开仓单被静默拒绝」这个回归。

改动之前：`send_server_order` 造的 reference 是 `CtaStrategy_<名字>`，不带
`|stop=` 后缀；挂在 `MainEngine.send_order` 上的「强制止损检查」据此判定
「增敞口且无止损」并拒单，而 `if not vt_orderid: continue` 把这次拒绝吃掉。
结果是策略在界面上「运行中」、日志安静、`pos` 恒为 0，实际一单没发出去。

这份用例分两层守：编码层（reference 里到底有没有止损）与可见层（拒绝时
到底有没有留下能查的日志）。两层都塌了才会重现那个现象，所以两层都要钉。
"""

from __future__ import annotations

from vnpy.trader.constant import Direction
from vnpy_gatewaykit.order_stop import extract_stop

from vnpy_ctastrategy.stop_declaration import (
    declared_stop,
    explain_rejection,
    with_declared_stop,
)


class StubStrategy:
    """只提供 `get_stop_price` 的最小策略替身。"""

    def __init__(self, stop: object) -> None:
        self._stop = stop
        self.strategy_name = "aa"

    def get_stop_price(
        self, vt_symbol: str, direction: Direction, price: float
    ) -> object:
        return self._stop


class SilentStrategy:
    """连 `get_stop_price` 都没有 —— 上游模板拷出去的老策略就是这样。"""

    strategy_name = "old"


# ---------------------------------------------------------------------------
# declared_stop：策略说了什么
# ---------------------------------------------------------------------------
def test_a_declared_stop_is_read_back() -> None:
    assert declared_stop(
        StubStrategy(40.2), "NBIS.SMART", Direction.LONG, 42.5
    ) == 40.2


def test_a_strategy_without_the_hook_declares_nothing() -> None:
    assert declared_stop(SilentStrategy(), "NBIS.SMART", Direction.LONG, 42.5) is None


def test_an_explicit_none_declares_nothing() -> None:
    assert declared_stop(StubStrategy(None), "NBIS.SMART", Direction.LONG, 42.5) is None


def test_a_nan_stop_is_treated_as_not_declared_rather_than_raising() -> None:
    """`attach_stop` 对 NaN 会抛 ValueError，而下单路径上抛异常会中断整轮委托。

    判为未声明只让这一笔被闸拒掉，其余标的照常下单 —— 一个坏止损是它自己的
    问题，不该连坐。
    """
    assert declared_stop(
        StubStrategy(float("nan")), "NBIS.SMART", Direction.LONG, 42.5
    ) is None


def test_a_non_positive_stop_is_treated_as_not_declared() -> None:
    for bad in (0.0, -1.0):
        assert declared_stop(StubStrategy(bad), "NBIS.SMART", Direction.LONG, 42.5) is None


def test_a_non_numeric_stop_is_treated_as_not_declared() -> None:
    assert declared_stop(
        StubStrategy("便宜"), "NBIS.SMART", Direction.LONG, 42.5
    ) is None


# ---------------------------------------------------------------------------
# with_declared_stop：reference 里到底带没带
# ---------------------------------------------------------------------------
def test_reference_carries_the_stop_so_the_gate_can_see_it() -> None:
    reference: str = with_declared_stop(
        "CtaStrategy_aa", StubStrategy(40.2), "NBIS.SMART", Direction.LONG, 42.5
    )

    assert extract_stop(reference) == 40.2
    # 策略名仍然可读 —— reference 同时是给人看的溯源信息。
    assert reference.startswith("CtaStrategy_aa")


def test_reference_is_untouched_when_nothing_is_declared() -> None:
    """原样返回不是兜底成功：这笔委托仍会被闸拒掉，只是拒得有据可查。"""
    reference: str = with_declared_stop(
        "CtaStrategy_old", SilentStrategy(), "NBIS.SMART", Direction.LONG, 42.5
    )

    assert reference == "CtaStrategy_old"
    assert extract_stop(reference) is None


def test_a_strategy_named_like_the_marker_does_not_fake_a_stop() -> None:
    """策略名是用户输入的自由文本，不能被当成已声明止损。"""
    reference: str = with_declared_stop(
        "CtaStrategy_my|stop=999", SilentStrategy(), "NBIS.SMART", Direction.LONG, 42.5
    )

    assert extract_stop(reference) is None


# ---------------------------------------------------------------------------
# explain_rejection：拒绝时留下的那句话
# ---------------------------------------------------------------------------
def test_rejection_without_a_stop_names_the_hook_to_implement() -> None:
    """排查者从「委托没发出去」出发，几乎不可能自己想到 reference 后缀。"""
    message: str = explain_rejection("aa", "NBIS.SMART", "CtaStrategy_aa")

    assert "未声明止损价" in message
    assert "get_stop_price" in message
    assert "aa" in message and "NBIS.SMART" in message


def test_rejection_with_a_stop_points_at_the_other_gates() -> None:
    """带了止损还被拒，原因就在委托规模/单笔风险/重复委托那几条上。"""
    message: str = explain_rejection("aa", "NBIS.SMART", "CtaStrategy_aa|stop=40.2")

    assert "未声明止损价" not in message
    assert "风控日志" in message
