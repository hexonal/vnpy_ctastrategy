"""让 CTA 委托带上申报止损 —— 修「开仓单 100% 被静默拒绝」。

━━━ 现象 ━━━

界面上策略「运行中」，日志安静，`pos` 恒为 0，`on_order` / `on_trade` 从不触发，
也没有任何异常。实际是**一单都没发出去**。

━━━ 根因 ━━━

三条自研风控闸挂在 `MainEngine.send_order` 上（`vnpy_alphakit.rules`，由
`run_gui.py` / `run.py` 在 `add_app(RiskManagerApp)` 之后 `install_gate_rules()`
装上），其中「强制止损检查」要求**任何增敞口的委托必须声明止损价**。

止损价的载体不是独立字段，而是 `OrderRequest.reference` 末尾的 `|stop=<价>`
后缀。而 `engine.py` 里 reference 一直是 `f"{APP_NAME}_{strategy_name}"`
—— 不带后缀 → 闸判为「增敞口且无止损」→ 拒单，`send_order` 返回空字符串。

`send_server_order` 拿到空 `vt_orderid` 只做 `continue`：返回空 list，
`vt_orderids` 为空，策略侧没有任何回调、没有异常、没有日志。三处静默叠在一起，
就成了「看着在跑，其实一单没发」。

装载顺序让这条路径必然被打到：`run_gui.py` 先 `install_gate_rules()`
再 `add_app(CtaStrategyApp)`，所以 GUI 一起来，闸就已经在 CTA 之前就位。

━━━ 改法 ━━━

给策略一个声明止损的入口，而不是给 CTA 开一道后门。

* `CtaTemplate.get_stop_price()` 默认返回 None（见 `template.py`）；
* 策略重写它来声明止损，本模块把返回值编码进 reference；
* 策略不声明 → 委托仍然被拒（**这正是风控策略本身，不该绕过**），
  但改由 `engine.py` 明确写日志说清是谁、哪一笔、为什么。

即：把「静默失败」变成「可见失败」，而不是把闸拆掉。

━━━ 为什么不给引擎加一个默认止损百分比 ━━━

考虑过、否掉了。引擎按「入场价 × (1−pct)」猜一个止损，能让现有策略立刻恢复
下单，但那个价位不是策略交易逻辑得出的：它会进入风控的风险核算
（`|entry-stop| × volume × size`），于是仓位大小被一个没人决定过的数字决定。
海龟策略的 2N 止损和一个拍脑袋的 5% 是完全不同的东西，而报表上看不出区别。

━━━ 一条诚实边界 ━━━

这里的止损是**申报值**，用于风控核算与过闸，不是券商侧的止损单 —— Futu 与
uSMART 都不置 `ContractData.stop_supported`。真正的保护性委托仍然要靠策略自己
挂（`CtaTemplate` 的本地停止单）或人工处置。本模块不改变这一点。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vnpy.trader.constant import Direction
from vnpy_gatewaykit.order_stop import attach_stop, extract_stop, is_finite, strip_stop

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .template import CtaTemplate


def declared_stop(
    strategy: CtaTemplate,
    vt_symbol: str,
    direction: Direction,
    price: float,
) -> float | None:
    """问策略要这一笔的申报止损；策略没声明或声明得不可用时返回 None。

    用 `getattr` 而不是直接调用，是为了兼容不继承本 fork `CtaTemplate` 的
    策略对象（测试替身、以及历史上从上游模板拷出去的类）。与
    `AlphaLiveEngine.resolve_stop` 用的是同一套鸭子类型约定和同一个签名，
    这样同一个策略类在两个引擎下声明止损的方式完全一致。

    非有限值与非正值在这里就地丢弃并当作「没声明」：`attach_stop` 对它们会
    抛 ValueError，而下单路径上抛异常会中断整轮委托；判为未声明则只让这一笔
    被闸拒掉，其余委托照常。
    """
    hook = getattr(strategy, "get_stop_price", None)
    if not callable(hook):
        return None

    stop = hook(vt_symbol, direction, price)
    if stop is None:
        return None

    try:
        value = float(stop)
    except (TypeError, ValueError):
        return None

    if not is_finite(value) or value <= 0:
        return None
    return value


def with_declared_stop(
    reference: str,
    strategy: CtaTemplate,
    vt_symbol: str,
    direction: Direction,
    price: float,
) -> str:
    """把策略声明的止损编码进 reference；没声明则返回一个**确定不带**止损的 reference。

    没声明时之所以要 `strip_stop` 而不是原样返回：reference 是
    `f"{APP_NAME}_{strategy_name}"`，而 `strategy_name` 是用户在界面上自由
    输入的文本。一个叫 `my|stop=999` 的策略会让 reference 天然以合法的止损
    后缀结尾，闸把它读成「已声明止损 999」并放行 —— 用策略名就能绕过强制
    止损检查。`attach_stop` 内部本来就会先 strip 再拼，所以只有「未声明」
    这条分支有这个洞；补上之后两条分支都以「reference 里的止损只可能来自
    `get_stop_price`」收口。

    未声明时返回不带止损的 reference 不是兜底成功 —— 它意味着这笔委托会被
    「强制止损检查」拒掉。调用方负责把那次拒绝喊出来。
    """
    stop: float | None = declared_stop(strategy, vt_symbol, direction, price)
    if stop is None:
        return strip_stop(reference)
    return attach_stop(reference, stop)


def explain_rejection(strategy_name: str, vt_symbol: str, reference: str) -> str:
    """委托没能拿到 vt_orderid 时写进日志的那句话。

    刻意点名 `get_stop_price`：这条路径上最常见的原因就是策略没声明止损，
    而排查者从「委托没发出去」这个现象出发，几乎不可能想到 reference 后缀。
    """
    if extract_stop(reference) is None:
        hint: str = (
            "该委托未声明止损价 —— 风控的「强制止损检查」要求增敞口委托必须带止损。"
            "请在策略里实现 get_stop_price(vt_symbol, direction, price) 返回止损价"
        )
    else:
        hint = "已带止损价, 拒绝原因见风控日志(委托规模/单笔风险/重复委托等)"
    return f"委托未被接受: {strategy_name} {vt_symbol} reference={reference!r} —— {hint}"
