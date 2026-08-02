"""引擎级证据：`send_server_order` 真的把止损带进了 OrderRequest，拒单真的喊了出来。

`test_stop_declaration.py` 钉的是编解码函数本身；这一份钉的是**接线** ——
`engine.py` 到底有没有调用它、reference 到底有没有落到真的 `OrderRequest` 上。
两者缺一：函数写对了但没接上去，现象和完全没修一模一样。

不构造真的 `CtaEngine`：它的 `__init__` 会 `get_database()` 并连 QuestDB，
而这条链路上一行数据库代码都不碰。`__new__` + 手工装配它真正用到的四个属性，
用例因此在没有数据库的机器上也能跑（同目录的
`test_local_stop_survives_rejection.py` 走的是真构造，所以它需要 QuestDB）。
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vnpy.trader.constant import Direction, Exchange, Offset, OrderType, Product
from vnpy.trader.object import ContractData, OrderRequest
from vnpy_gatewaykit.order_stop import extract_stop

from vnpy_ctastrategy.engine import CtaEngine

VT_SYMBOL = "NBIS.SMART"


@dataclass
class _FakeMainEngine:
    """只提供 `send_server_order` 真正用到的三个方法。

    `accept` 为 False 时 `send_order` 返回空字符串 —— 这正是风控闸拒单后
    `MainEngine.send_order` 的返回值，也是本文件要钉住的那个分支。
    """

    accept: bool = True
    sent: list[OrderRequest] = field(default_factory=list)

    def convert_order_request(
        self, req: OrderRequest, gateway_name: str, lock: bool, net: bool
    ) -> list[OrderRequest]:
        return [req]

    def send_order(self, req: OrderRequest, gateway_name: str) -> str:
        self.sent.append(req)
        return "FAKE.1" if self.accept else ""

    def update_order_request(
        self, req: OrderRequest, vt_orderid: str, gateway_name: str
    ) -> None:
        return None


class _Strategy:
    def __init__(self, name: str, stop: float | None) -> None:
        self.strategy_name = name
        self._stop = stop

    def get_stop_price(
        self, vt_symbol: str, direction: Direction, price: float
    ) -> float | None:
        return self._stop


def _contract() -> ContractData:
    symbol, exchange = VT_SYMBOL.split(".")
    return ContractData(
        symbol=symbol,
        exchange=Exchange(exchange),
        name=symbol,
        product=Product.EQUITY,
        size=1,
        pricetick=0.01,
        gateway_name="FAKE",
    )


def _engine(main: _FakeMainEngine) -> CtaEngine:
    engine = CtaEngine.__new__(CtaEngine)
    engine.main_engine = main  # type: ignore[assignment]
    engine.orderid_strategy_map = {}
    engine.strategy_orderid_map = defaultdict(set)
    engine.logs = []  # type: ignore[attr-defined]
    engine.write_log = lambda msg, strategy=None: engine.logs.append(msg)  # type: ignore[assignment,attr-defined]
    return engine


def _send(engine: CtaEngine, strategy: _Strategy) -> list:
    return engine.send_server_order(
        strategy,  # type: ignore[arg-type]
        _contract(),
        Direction.LONG,
        Offset.OPEN,
        42.5,
        100,
        OrderType.LIMIT,
        False,
        False,
    )


# ---------------------------------------------------------------------------
# 接线：止损进没进 OrderRequest
# ---------------------------------------------------------------------------
def test_the_request_that_reaches_send_order_carries_the_stop() -> None:
    """载荷断言而不是返回值断言 —— 要证明的是「发出去的那张委托」长什么样。"""
    main = _FakeMainEngine()
    engine = _engine(main)

    _send(engine, _Strategy("aa", 40.2))

    assert len(main.sent) == 1
    assert extract_stop(main.sent[0].reference) == 40.2
    assert main.sent[0].reference.startswith("CtaStrategy_aa")


def test_a_strategy_that_declares_nothing_still_sends_a_stopless_request() -> None:
    """引擎不替策略编造止损；这张单会被闸拒掉，那是风控策略本身。"""
    main = _FakeMainEngine()
    engine = _engine(main)

    _send(engine, _Strategy("old", None))

    assert extract_stop(main.sent[0].reference) is None


def test_a_strategy_name_cannot_forge_a_stop() -> None:
    """`strategy_name` 是界面上自由输入的文本，而 `|stop=` 后缀锚定在末尾 ——
    一个叫 `my|stop=999` 的策略曾能让 reference 天然读成「已声明止损 999」，
    用改名就绕过强制止损闸。"""
    main = _FakeMainEngine()
    engine = _engine(main)

    _send(engine, _Strategy("my|stop=999", None))

    assert extract_stop(main.sent[0].reference) is None


# ---------------------------------------------------------------------------
# 可见性：拒单有没有喊出来
# ---------------------------------------------------------------------------
def test_a_refused_order_is_logged_instead_of_silently_skipped() -> None:
    """回归本体：原来这里只有 `continue`，返回空 list，策略侧毫无察觉。"""
    main = _FakeMainEngine(accept=False)
    engine = _engine(main)

    vt_orderids = _send(engine, _Strategy("aa", None))

    assert vt_orderids == []
    logs = engine.logs  # type: ignore[attr-defined]
    assert logs, "委托被拒却没有任何日志 —— 这正是原来的静默失败"
    assert "get_stop_price" in logs[0]
    assert "aa" in logs[0] and VT_SYMBOL in logs[0]


def test_a_refused_order_that_did_declare_a_stop_points_elsewhere() -> None:
    """带了止损还被拒，就不该再让人去查 get_stop_price。"""
    main = _FakeMainEngine(accept=False)
    engine = _engine(main)

    _send(engine, _Strategy("aa", 40.2))

    logs = engine.logs  # type: ignore[attr-defined]
    assert "get_stop_price" not in logs[0]
    assert "风控日志" in logs[0]


def test_an_accepted_order_logs_nothing_and_is_tracked() -> None:
    main = _FakeMainEngine()
    engine = _engine(main)

    vt_orderids = _send(engine, _Strategy("aa", 40.2))

    assert vt_orderids == ["FAKE.1"]
    assert engine.logs == []  # type: ignore[attr-defined]
    assert engine.orderid_strategy_map["FAKE.1"].strategy_name == "aa"
