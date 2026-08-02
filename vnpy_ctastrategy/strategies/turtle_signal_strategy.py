from vnpy_ctastrategy import (
    ArrayManager,
    BarData,
    BarGenerator,
    CtaTemplate,
    Direction,
    OrderData,
    StopOrder,
    TickData,
    TradeData,
)


class TurtleSignalStrategy(CtaTemplate):
    """"""
    author = "用Python的交易员"

    entry_window: int = 20
    exit_window: int = 10
    atr_window: int = 20
    fixed_size: int = 1

    entry_up: float = 0
    entry_down: float = 0
    exit_up: float = 0
    exit_down: float = 0
    atr_value: float = 0
    long_entry: float = 0
    short_entry: float = 0
    long_stop: float = 0
    short_stop: float = 0

    parameters = ["entry_window", "exit_window", "atr_window", "fixed_size"]
    variables = ["entry_up", "entry_down", "exit_up", "exit_down", "atr_value"]

    def on_init(self) -> None:
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")

        self.bg: BarGenerator = BarGenerator(self.on_bar)
        self.am: ArrayManager = ArrayManager()

        self.load_bar(20)

    def on_start(self) -> None:
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")

    def on_stop(self) -> None:
        """
        Callback when strategy is stopped.
        """
        self.write_log("策略停止")

    def on_tick(self, tick: TickData) -> None:
        """
        Callback of new tick data update.
        """
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData) -> None:
        """
        Callback of new bar data update.
        """
        self.cancel_all()

        self.am.update_bar(bar)
        if not self.am.inited:
            return

        # Only calculates new entry channel when no position holding
        if not self.pos:
            self.entry_up, self.entry_down = self.am.donchian(
                self.entry_window
            )

        self.exit_up, self.exit_down = self.am.donchian(self.exit_window)

        if not self.pos:
            self.atr_value = self.am.atr(self.atr_window)

            self.long_entry = 0
            self.short_entry = 0
            self.long_stop = 0
            self.short_stop = 0

            self.send_buy_orders(self.entry_up)
            self.send_short_orders(self.entry_down)
        elif self.pos > 0:
            self.send_buy_orders(self.entry_up)

            sell_price: float = max(self.long_stop, self.exit_down)
            self.sell(sell_price, abs(self.pos), True)

        elif self.pos < 0:
            self.send_short_orders(self.entry_down)

            cover_price: float = min(self.short_stop, self.exit_up)
            self.cover(cover_price, abs(self.pos), True)

        self.put_event()

    def on_trade(self, trade: TradeData) -> None:
        """
        Callback of new trade data update.
        """
        if trade.direction == Direction.LONG:
            self.long_entry = trade.price
            self.long_stop = self.long_entry - 2 * self.atr_value
        else:
            self.short_entry = trade.price
            self.short_stop = self.short_entry + 2 * self.atr_value

    def get_stop_price(
        self, vt_symbol: str, direction: Direction, price: float
    ) -> float | None:
        """海龟的 2N 止损，按**本笔委托的报价**算，而不是按已成交的入场价。

        `on_trade` 里的 `long_stop = long_entry - 2 * atr_value` 是同一条规则，
        但它在成交之后才有值；风控闸要在下单之前就看到止损，所以这里用即将
        报出去的 `price` 代入同一个公式。两者在成交价等于报价时完全一致。

        `atr_value` 为 0 时返回 None 而不是返回 `price`：ArrayManager 还没
        `inited`（或刚清过仓、ATR 尚未重算）时 N 是 0，止损等于入场价 ——
        那不是止损，`check_stop_side` 也会以「该止损不提供保护」拒掉它。
        如实说「现在声明不了」，让委托被拒得有理由可查。
        """
        if self.atr_value <= 0:
            return None
        if direction == Direction.LONG:
            return price - 2 * self.atr_value
        return price + 2 * self.atr_value

    def on_order(self, order: OrderData) -> None:
        """
        Callback of new order data update.
        """
        pass

    def on_stop_order(self, stop_order: StopOrder) -> None:
        """
        Callback of stop order update.
        """
        pass

    def send_buy_orders(self, price: float) -> None:
        """"""
        t: float = self.pos / self.fixed_size

        if t < 1:
            self.buy(price, self.fixed_size, True)

        if t < 2:
            self.buy(price + self.atr_value * 0.5, self.fixed_size, True)

        if t < 3:
            self.buy(price + self.atr_value, self.fixed_size, True)

        if t < 4:
            self.buy(price + self.atr_value * 1.5, self.fixed_size, True)

    def send_short_orders(self, price: float) -> None:
        """"""
        t: float = self.pos / self.fixed_size

        if t > -1:
            self.short(price, self.fixed_size, True)

        if t > -2:
            self.short(price - self.atr_value * 0.5, self.fixed_size, True)

        if t > -3:
            self.short(price - self.atr_value, self.fixed_size, True)

        if t > -4:
            self.short(price - self.atr_value * 1.5, self.fixed_size, True)
