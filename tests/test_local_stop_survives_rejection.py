"""本地停止单在委托被拒时必须保持武装，不能静默消失。

为什么值得单独一个文件：这是一条会让仓位裸奔的路径，而且**看起来是正常工作的**。

链路：
    check_stop_order 触发 -> send_limit_order(...) -> 网关拒单
    -> RejectOrderMixin._reject 返回**非空** vt_orderid（BaseGateway 契约要求
       send_order 必须返回一个单号，哪怕是拒单）
    -> engine.py 的 `if vt_orderids:` 判定「下单成功」
    -> 停止单被 pop、状态标 TRIGGERED、on_stop_order 通知策略

结果：保护性委托根本没进场，策略却收到「止损已触发」，仓位从此无保护，
且没有任何一处报错。策略若据此认为已离场，还会在下一根 K 线重复开仓。

这条在 futu 路径上不是理论风险。CTA 触发后取价是
`tick.limit_up or tick.ask_price_5`（LONG），而 vnpy_futu 从不设置 limit_up
（港美股本就没有涨跌停），于是永远退到 ask_price_5；盘口不足五档时它是 0.0，
正好撞上 vnpy_gatewaykit.reject 的「限价/止损单价须 > 0」闸。

正确行为：委托没被受理 = 止损没生效 = 停止单必须**留在原地继续武装**，
并把失败大声说出来。宁可下一个 tick 再试一次，也不能假装已经离场。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vnpy.trader.constant import Direction, Exchange, Offset, Product, Status
from vnpy.trader.object import ContractData, OrderData, TickData

from vnpy_ctastrategy.base import StopOrderStatus

VT_SYMBOL = "700.SEHK"


@dataclass
class _FakeMainEngine:
    """只提供 CtaEngine 在这条链路上真正用到的东西。

    reject_next 为 True 时，send_order 模拟 RejectOrderMixin._reject：
    先推一个 REJECTED 的 OrderData 进 OMS，再返回它**非空**的单号 —— 这正是
    真网关的行为，也正是本用例要钉住的前提。
    """

    reject_next: bool = True
    sent: list = field(default_factory=list)
    pushed_orders: list = field(default_factory=list)
    logs: list = field(default_factory=list)

    def get_contract(self, vt_symbol: str) -> ContractData:
        symbol, exchange = vt_symbol.split(".")
        return ContractData(
            symbol=symbol,
            exchange=Exchange(exchange),
            name="腾讯",
            product=Product.EQUITY,
            size=1,
            pricetick=0.2,
            gateway_name="FAKE",
        )

    def send_order(self, req, gateway_name: str) -> str:
        self.sent.append(req)
        n = len(self.sent)
        order: OrderData = req.create_order_data(f"local-reject-{n}", gateway_name)
        if self.reject_next:
            order.status = Status.REJECTED
        else:
            order.status = Status.NOTTRADED
        self.pushed_orders.append(order)
        return order.vt_orderid

    def get_order(self, vt_orderid: str) -> OrderData | None:
        for order in self.pushed_orders:
            if order.vt_orderid == vt_orderid:
                return order
        return None

    def convert_order_request(
        self, req, gateway_name: str, lock: bool, net: bool
    ) -> list:
        """股票是净持仓，OffsetConverter 原样放行（对应 net_position 那条路径）。"""
        return [req]

    def update_order_request(self, req, vt_orderid: str, gateway_name: str) -> None:
        """OffsetConverter 的冻结登记，本用例不关心。"""

    # CtaEngine 在这条链路上还会碰到的零星调用
    def get_tick(self, vt_symbol: str):
        return None

    def cancel_order(self, req, gateway_name: str) -> None:
        pass


class _FakeStrategy:
    def __init__(self) -> None:
        self.strategy_name = "S1"
        self.vt_symbol = VT_SYMBOL
        self.inited = True
        self.trading = True
        self.stop_order_calls: list = []

    def on_stop_order(self, stop_order) -> None:
        self.stop_order_calls.append(stop_order)


def _engine_with_armed_stop(*, reject: bool):
    """建一个已武装本地停止单的 CtaEngine，并把它的依赖换成假的。"""
    from vnpy.event import EventEngine

    from vnpy_ctastrategy.engine import CtaEngine

    main = _FakeMainEngine(reject_next=reject)
    engine = CtaEngine(main, EventEngine())          # type: ignore[arg-type]
    engine.write_log = lambda msg, strategy=None: main.logs.append(str(msg))  # type: ignore[assignment]

    strategy = _FakeStrategy()
    engine.strategies[strategy.strategy_name] = strategy      # type: ignore[assignment]
    engine.strategy_orderid_map[strategy.strategy_name] = set()

    stop_orderid = engine.send_local_stop_order(
        strategy,                                              # type: ignore[arg-type]
        Direction.SHORT, Offset.CLOSE, 420.0, 100, False, False,
    )[0]
    assert stop_orderid in engine.stop_orders, "前置条件不成立：停止单没挂上"
    return engine, main, strategy, stop_orderid


def _triggering_tick(*, bid5: float) -> TickData:
    """一个会触发 SHORT 停止单（跌破 420）的 tick。

    bid5=0.0 复现真实盘面：港美股无涨跌停 -> limit_down 恒 0 -> 退到
    bid_price_5；五档不满时它就是 0.0，随后被 reject 闸挡下。
    """
    from datetime import datetime

    symbol, exchange = VT_SYMBOL.split(".")
    return TickData(
        symbol=symbol, exchange=Exchange(exchange), datetime=datetime.now(),
        name="腾讯", last_price=415.0, bid_price_5=bid5, gateway_name="FAKE",
    )


def test_rejected_protective_order_leaves_the_stop_armed() -> None:
    """委托被拒 -> 停止单必须仍在 stop_orders 里，状态不得变 TRIGGERED。

    修复前实测：停止单被 pop，状态 TRIGGERED，仓位裸奔且策略被告知已离场。
    """
    engine, main, strategy, stop_orderid = _engine_with_armed_stop(reject=True)

    engine.check_stop_order(_triggering_tick(bid5=0.0))

    assert main.sent, "根本没尝试下保护单，前提就不对"
    assert main.pushed_orders[-1].status is Status.REJECTED, "替身没在模拟拒单"

    assert stop_orderid in engine.stop_orders, (
        "保护单被拒，停止单却被撤掉了 —— 仓位从此无保护，且没有任何报错"
    )
    assert engine.stop_orders[stop_orderid].status is not StopOrderStatus.TRIGGERED, (
        "保护单没进场，却把停止单标成已触发 —— 策略会以为自己已经离场"
    )


def test_the_failure_is_loud() -> None:
    """静默是这条 bug 最贵的部分：必须留下能看见的日志。"""
    engine, main, _s, _id = _engine_with_armed_stop(reject=True)

    engine.check_stop_order(_triggering_tick(bid5=0.0))

    assert any("止损" in line or "停止单" in line for line in main.logs), (
        f"保护单被拒却没留下日志: {main.logs}"
    )


def test_strategy_is_not_told_it_exited() -> None:
    """on_stop_order 不能带着 TRIGGERED 通知策略 —— 那会让策略更新持仓状态。"""
    engine, main, strategy, _id = _engine_with_armed_stop(reject=True)

    engine.check_stop_order(_triggering_tick(bid5=0.0))

    triggered = [so for so in strategy.stop_order_calls
                 if so.status is StopOrderStatus.TRIGGERED]
    assert not triggered, "策略收到了『止损已触发』，但保护单其实被拒了"


def test_accepted_order_still_retires_the_stop() -> None:
    """反向守卫：委托真被受理时，行为必须和从前完全一致。

    没有这一条，上面三条可以靠『永远不撤停止单』作弊通过。
    """
    engine, main, strategy, stop_orderid = _engine_with_armed_stop(reject=False)

    engine.check_stop_order(_triggering_tick(bid5=414.0))

    assert stop_orderid not in engine.stop_orders, "正常受理时停止单应当退场"
    assert any(so.status is StopOrderStatus.TRIGGERED
               for so in strategy.stop_order_calls), "正常受理时应通知策略已触发"


def test_a_partially_rejected_batch_keeps_the_stop_armed() -> None:
    """拆单场景：只要没有任何一条腿被受理，就等于没有保护。

    convert_order_request 可能把一笔拆成今仓/昨仓两腿；全被拒和单腿被拒
    是两回事，这里钉的是「全被拒」必须视为失败。
    """
    engine, main, _s, stop_orderid = _engine_with_armed_stop(reject=True)

    engine.check_stop_order(_triggering_tick(bid5=0.0))

    assert all(o.status is Status.REJECTED for o in main.pushed_orders)
    assert stop_orderid in engine.stop_orders


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
