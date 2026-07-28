from vnpy.event import Event, EventEngine
from vnpy.trader.constant import Exchange
from vnpy.trader.engine import MainEngine
from vnpy.trader.ui import QtCore, QtGui, QtWidgets
from vnpy.trader.ui.widget import BaseCell, BaseMonitor, EnumCell, MsgCell, TimeCell

from ..base import APP_NAME, EVENT_CTA_LOG, EVENT_CTA_STOPORDER, EVENT_CTA_STRATEGY
from ..engine import CtaEngine
from ..locale import _
from .rollover import RolloverTool


class CtaManager(QtWidgets.QWidget):
    """"""

    signal_log: QtCore.Signal = QtCore.Signal(Event)
    signal_strategy: QtCore.Signal = QtCore.Signal(Event)

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        """"""
        super().__init__()

        self.main_engine: MainEngine = main_engine
        self.event_engine: EventEngine = event_engine
        self.cta_engine: CtaEngine = main_engine.get_engine(APP_NAME)  # type: ignore

        self.managers: dict[str, StrategyManager] = {}

        self.init_ui()
        self.register_event()
        self.cta_engine.init_engine()
        self.update_class_combo()

    def init_ui(self) -> None:
        """"""
        self.setWindowTitle(_("CTA策略"))

        # Create widgets
        self.class_combo: QtWidgets.QComboBox = QtWidgets.QComboBox()

        add_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("添加策略"))
        add_button.clicked.connect(self.add_strategy)

        init_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("全部初始化"))
        init_button.clicked.connect(self.cta_engine.init_all_strategies)

        start_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("全部启动"))
        start_button.clicked.connect(self.cta_engine.start_all_strategies)

        stop_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("全部停止"))
        stop_button.clicked.connect(self.cta_engine.stop_all_strategies)

        clear_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("清空日志"))
        clear_button.clicked.connect(self.clear_log)

        roll_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("移仓助手"))
        roll_button.clicked.connect(self.roll)

        self.scroll_layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        self.scroll_layout.addStretch()

        scroll_widget: QtWidgets.QWidget = QtWidgets.QWidget()
        scroll_widget.setLayout(self.scroll_layout)

        self.scroll_area: QtWidgets.QScrollArea = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(scroll_widget)

        self.log_monitor: LogMonitor = LogMonitor(self.main_engine, self.event_engine)

        self.stop_order_monitor: StopOrderMonitor = StopOrderMonitor(
            self.main_engine, self.event_engine
        )

        self.strategy_combo = QtWidgets.QComboBox()
        self.strategy_combo.setMinimumWidth(200)
        find_button = QtWidgets.QPushButton(_("查找"))
        find_button.clicked.connect(self.find_strategy)

        # Set layout
        hbox1: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        hbox1.addWidget(self.class_combo)
        hbox1.addWidget(add_button)
        hbox1.addStretch()
        hbox1.addWidget(self.strategy_combo)
        hbox1.addWidget(find_button)
        hbox1.addStretch()
        hbox1.addWidget(init_button)
        hbox1.addWidget(start_button)
        hbox1.addWidget(stop_button)
        hbox1.addWidget(clear_button)
        hbox1.addWidget(roll_button)

        grid: QtWidgets.QGridLayout = QtWidgets.QGridLayout()
        grid.addWidget(self.scroll_area, 0, 0, 2, 1)
        grid.addWidget(self.stop_order_monitor, 0, 1)
        grid.addWidget(self.log_monitor, 1, 1)

        vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(grid)

        self.setLayout(vbox)

    def update_class_combo(self) -> None:
        """"""
        names = self.cta_engine.get_all_strategy_class_names()
        names.sort()
        for name in names:
            shown, hint = describe_strategy(name)
            self.class_combo.addItem(shown, userData=name)
            self.class_combo.setItemData(
                self.class_combo.count() - 1, hint, QtCore.Qt.ItemDataRole.ToolTipRole
            )

    def update_strategy_combo(self) -> None:
        """"""
        names = list(self.managers.keys())
        names.sort()

        self.strategy_combo.clear()
        self.strategy_combo.addItems(names)

    def register_event(self) -> None:
        """"""
        self.signal_strategy.connect(self.process_strategy_event)

        self.event_engine.register(
            EVENT_CTA_STRATEGY, self.signal_strategy.emit
        )

    def process_strategy_event(self, event: Event) -> None:
        """
        Update strategy status onto its monitor.
        """
        data = event.data
        strategy_name: str = data["strategy_name"]

        if strategy_name in self.managers:
            manager: StrategyManager = self.managers[strategy_name]
            manager.update_data(data)
        else:
            manager = StrategyManager(self, self.cta_engine, data)
            self.scroll_layout.insertWidget(0, manager)
            self.managers[strategy_name] = manager

            self.update_strategy_combo()

    def remove_strategy(self, strategy_name: str) -> None:
        """"""
        manager: StrategyManager = self.managers.pop(strategy_name)
        manager.deleteLater()

        self.update_strategy_combo()

    def add_strategy(self) -> None:
        """"""
        # 读 userData 而不是 currentText：显示文本带了中文说明
        # （`AtrRsiStrategy · ATR+RSI 波动突破`），类名只在 userData 里。
        class_name: str = str(self.class_combo.currentData())
        if not class_name:
            return

        parameters: dict = self.cta_engine.get_strategy_class_parameters(class_name)
        editor: SettingEditor = SettingEditor(
            parameters, class_name=class_name, vt_symbols=self._known_vt_symbols()
        )
        n: int = editor.exec_()

        if n == editor.DialogCode.Accepted:
            setting: dict = editor.get_setting()
            vt_symbol: str = setting.pop("vt_symbol")
            strategy_name: str = setting.pop("strategy_name")

            problem: str = _bad_vt_symbol(vt_symbol)
            if problem:
                # 引擎侧同样会拒（engine.py:679-686），但它只写一行日志，
                # 而对话框此时已经关了 —— 用户看到的是"什么都没发生"。
                # 在这里当场弹出来，说清楚哪儿不对。
                QtWidgets.QMessageBox.warning(self, _("添加策略"), problem)
                return

            self.cta_engine.add_strategy(
                class_name, strategy_name, vt_symbol, setting
            )

    def _known_vt_symbols(self) -> list[str]:
        """本地已知合约的本地代码，带上名称便于辨认（取值时只取第一段）。"""
        contracts = self.cta_engine.main_engine.get_all_contracts()
        return sorted(
            f"{c.symbol}.{c.exchange.value} {c.name}".rstrip() for c in contracts
        )

    def find_strategy(self) -> None:
        """"""
        strategy_name = self.strategy_combo.currentText()
        if strategy_name:
            manager = self.managers[strategy_name]
            self.scroll_area.ensureWidgetVisible(manager)

    def clear_log(self) -> None:
        """"""
        self.log_monitor.setRowCount(0)

    def show(self) -> None:
        """"""
        self.showMaximized()

    def roll(self) -> None:
        """"""
        dialog: RolloverTool = RolloverTool(self)
        dialog.exec_()


class StrategyManager(QtWidgets.QFrame):
    """
    Manager for a strategy
    """

    def __init__(
        self, cta_manager: CtaManager, cta_engine: CtaEngine, data: dict
    ) -> None:
        """"""
        super().__init__()

        self.cta_manager: CtaManager = cta_manager
        self.cta_engine: CtaEngine = cta_engine

        self.strategy_name: str = data["strategy_name"]
        self._data: dict = data

        self.init_ui()

    def init_ui(self) -> None:
        """"""
        self.setFixedHeight(300)
        self.setFrameShape(self.Shape.Box)
        self.setLineWidth(1)

        self.init_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("初始化"))
        self.init_button.clicked.connect(self.init_strategy)

        self.start_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("启动"))
        self.start_button.clicked.connect(self.start_strategy)
        self.start_button.setEnabled(False)

        self.stop_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("停止"))
        self.stop_button.clicked.connect(self.stop_strategy)
        self.stop_button.setEnabled(False)

        self.edit_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("编辑"))
        self.edit_button.clicked.connect(self.edit_strategy)

        self.remove_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("移除"))
        self.remove_button.clicked.connect(self.remove_strategy)

        strategy_name: str = self._data["strategy_name"]
        vt_symbol: str = self._data["vt_symbol"]
        class_name: str = self._data["class_name"]
        author: str = self._data["author"]

        label_text: str = (
            f"{strategy_name}  -  {vt_symbol}  ({class_name} by {author})"
        )
        label: QtWidgets.QLabel = QtWidgets.QLabel(label_text)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.parameters_monitor: DataMonitor = DataMonitor(self._data["parameters"])
        self.variables_monitor: DataMonitor = DataMonitor(self._data["variables"])

        hbox: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.init_button)
        hbox.addWidget(self.start_button)
        hbox.addWidget(self.stop_button)
        hbox.addWidget(self.edit_button)
        hbox.addWidget(self.remove_button)

        vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        vbox.addWidget(label)
        vbox.addLayout(hbox)
        vbox.addWidget(self.parameters_monitor)
        vbox.addWidget(self.variables_monitor)
        self.setLayout(vbox)

    def update_data(self, data: dict) -> None:
        """"""
        self._data = data

        self.parameters_monitor.update_data(data["parameters"])
        self.variables_monitor.update_data(data["variables"])

        # Update button status
        variables: dict = data["variables"]
        inited: bool = variables["inited"]
        trading: bool = variables["trading"]

        if not inited:
            return
        self.init_button.setEnabled(False)

        if trading:
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.edit_button.setEnabled(False)
            self.remove_button.setEnabled(False)
        else:
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.edit_button.setEnabled(True)
            self.remove_button.setEnabled(True)

    def init_strategy(self) -> None:
        """"""
        self.cta_engine.init_strategy(self.strategy_name)

    def start_strategy(self) -> None:
        """"""
        self.cta_engine.start_strategy(self.strategy_name)

    def stop_strategy(self) -> None:
        """"""
        self.cta_engine.stop_strategy(self.strategy_name)

    def edit_strategy(self) -> None:
        """"""
        strategy_name: str = self._data["strategy_name"]

        parameters: dict = self.cta_engine.get_strategy_parameters(strategy_name)
        editor: SettingEditor = SettingEditor(parameters, strategy_name=strategy_name)
        n: int = editor.exec_()

        if n == editor.DialogCode.Accepted:
            setting: dict = editor.get_setting()
            self.cta_engine.edit_strategy(strategy_name, setting)

    def remove_strategy(self) -> None:
        """"""
        result: bool = self.cta_engine.remove_strategy(self.strategy_name)

        # Only remove strategy gui manager if it has been removed from engine
        if result:
            self.cta_manager.remove_strategy(self.strategy_name)


def _display(value: object) -> str:
    """单元格里显示的文本。

    浮点数按 4 位小数截断：ATR 这类算出来的值 str() 后是
    `1.1276457926121324`，把列撑爆还得靠横向滚动才看得全，而后面十几位
    对看盘没有任何意义。完整值放进 tooltip，一位都不丢。

    布尔值转"是/否"：这两栏（已初始化/允许交易）是给人看的状态，
    True/False 是程序员的写法。
    """
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        # 截到 4 位后变成 0 但本身不是 0 —— 直接印 "0" 就成了谎报"没有值"。
        # 这种量级改用科学计数法，宁可难看也不能说错。
        if value and float(text or 0) == 0:
            return f"{value:.4g}"
        return text or "0"
    return str(value)


class DataMonitor(QtWidgets.QTableWidget):
    """
    Table monitor for parameters and variables.
    """

    def __init__(self, data: dict) -> None:
        """"""
        super().__init__()

        self._data: dict = data
        self.cells: dict = {}

        self.init_ui()

    def init_ui(self) -> None:
        """"""
        self.setColumnCount(len(self._data))
        self.setRowCount(1)
        self.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(self.EditTrigger.NoEditTriggers)

        for column, name in enumerate(self._data.keys()):
            value = self._data[name]
            shown, tip = describe_parameter(name, type(value))

            # 表头用中文标签，原字段名进 tooltip：这两张表原先只印
            # entry_up / atr_value 这种原名，对着界面看盘的人认不出是什么。
            header: QtWidgets.QTableWidgetItem = QtWidgets.QTableWidgetItem(shown)
            header.setToolTip(tip)
            self.setHorizontalHeaderItem(column, header)

            cell: QtWidgets.QTableWidgetItem = QtWidgets.QTableWidgetItem(_display(value))
            cell.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            cell.setToolTip(str(value))

            self.setItem(0, column, cell)
            self.cells[name] = cell

    def update_data(self, data: dict) -> None:
        """"""
        for name, value in data.items():
            cell: QtWidgets.QTableWidgetItem = self.cells[name]
            cell.setText(_display(value))
            cell.setToolTip(str(value))


class StopOrderMonitor(BaseMonitor):
    """
    Monitor for local stop order.
    """

    event_type: str = EVENT_CTA_STOPORDER
    data_key: str = "stop_orderid"
    sorting: bool = True

    headers: dict = {
        "stop_orderid": {
            "display": _("停止委托号"),
            "cell": BaseCell,
            "update": False,
        },
        "vt_orderids": {"display": _("限价委托号"), "cell": BaseCell, "update": True},
        "vt_symbol": {"display": _("本地代码"), "cell": BaseCell, "update": False},
        "direction": {"display": _("方向"), "cell": EnumCell, "update": False},
        "offset": {"display": _("开平"), "cell": EnumCell, "update": False},
        "price": {"display": _("价格"), "cell": BaseCell, "update": False},
        "volume": {"display": _("数量"), "cell": BaseCell, "update": False},
        "status": {"display": _("状态"), "cell": EnumCell, "update": True},
        "datetime": {"display": _("时间"), "cell": TimeCell, "update": False},
        "lock": {"display": _("锁仓"), "cell": BaseCell, "update": False},
        "net": {"display": _("净仓"), "cell": BaseCell, "update": False},
        "strategy_name": {"display": _("策略名"), "cell": BaseCell, "update": False},
    }

    def __del__(self) -> None:
        """"""
        pass


class LogMonitor(BaseMonitor):
    """
    Monitor for log data.
    """

    event_type: str = EVENT_CTA_LOG
    data_key: str = ""
    sorting: bool = False

    headers: dict = {
        "time": {"display": _("时间"), "cell": TimeCell, "update": False},
        "msg": {"display": _("信息"), "cell": MsgCell, "update": False},
    }

    def init_ui(self) -> None:
        """
        Stretch last column.
        """
        super().init_ui()

        self.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch
        )

    def insert_new_row(self, data: dict) -> None:
        """
        Insert a new row at the top of table.
        """
        super().insert_new_row(data)
        self.resizeRowToContents(0)




# 随包示例策略的中文名与一句话说明。
#
# 下拉框里原本只有类名（AtrRsiStrategy / DualThrustStrategy …）。对读过源码
# 的人够用，但对着界面选策略的人看不出它们分别用什么方法 —— 而选错策略再
# 回测，浪费的是时间和对结果的信任。
#
# 说明取自各策略 on_bar 的实际逻辑，不是照名字猜的。查不到就只显示类名。
_STRATEGY_LABELS: dict[str, tuple[str, str]] = {
    "AtrRsiStrategy": (
        "ATR+RSI 波动突破",
        "ATR 高于其均线（波动放大）时，按 RSI 突破 50±阈值 进场；持仓后按百分比移动止损",
    ),
    "BollChannelStrategy": (
        "布林通道 + CCI 过滤",
        "价格突破布林带上下轨进场，CCI 决定允许的方向，ATR 做移动止损",
    ),
    "DoubleMaStrategy": (
        "双均线交叉",
        "快慢 SMA 金叉做多、死叉做空。最经典的入门策略，用来跑通流程",
    ),
    "DualThrustStrategy": (
        "Dual Thrust 开盘区间突破",
        "用前一日振幅算上下轨（开盘价 ± K×前日Range），日内突破进场、收盘前平",
    ),
    "KingKeltnerStrategy": (
        "肯特纳通道突破",
        "中轨 ± ATR 构成通道，突破进场。Chester Keltner 的经典通道方法",
    ),
    "MultiSignalStrategy": (
        "多信号投票",
        "RSI / CCI / 均线 三个子信号各投一票，方向一致才进场",
    ),
    "MultiTimeframeStrategy": (
        "多周期共振",
        "15 分钟定趋势方向，5 分钟找具体入场点",
    ),
    "TestStrategy": (
        "测试骨架（不交易）",
        "on_bar 是 pass，不产生任何委托。只用来确认事件回调是否跑通，别拿它回测",
    ),
    "TurtleSignalStrategy": (
        "海龟法则（信号版）",
        "唐奇安通道突破进场，用 ATR（N 值）定止损距离与加仓间距",
    ),
}


def describe_strategy(class_name: str) -> tuple[str, str]:
    """策略类名 -> (显示文本, 悬停说明)。查不到只显示类名，不编中文。"""
    label, hint = _STRATEGY_LABELS.get(class_name, ("", ""))
    shown = f"{class_name} · {label}" if label else class_name
    return shown, (hint or class_name)


# 参数的中文标签与说明。
#
# SettingEditor 是靠反射生成的，原先直接把 `f"{name} {type_}"` 当标签 ——
# 屏幕上就是 `atr_length <class 'int'>` 这种。参数名对写策略的人够用，但
# 对着界面填参数的人看不出 rsi_entry 到底是"阈值"还是"周期"，而
# `<class 'int'>` 是 Python 的 repr 泄漏到了 UI 上。
#
# 这里只覆盖框架自带的通用字段与随包发布的示例策略。自定义策略可以在类上
# 声明 `parameter_labels: dict[str, tuple[str, str]]`（标签, 说明）来接管，
# 查不到就回落到参数名本身 —— 宁可显示原名，也不猜一个可能是错的中文。
_PARAM_LABELS: dict[str, tuple[str, str]] = {
    # 每个策略实例都有的两个
    "strategy_name": ("策略实例名", "自己起的名字，用来在列表里区分同一策略的多个实例"),
    "vt_symbol": ("交易标的", "格式为 代码.交易所，如 700.SEHK、NVDA.SMART"),
    # AtrRsiStrategy（随包示例）
    "atr_length": ("ATR 周期", "计算 ATR（真实波幅）用多少根 K 线"),
    "atr_ma_length": ("ATR 均线周期", "对 ATR 再取均值，用来判断波动是否在放大"),
    "rsi_length": ("RSI 周期", "计算 RSI 用多少根 K 线"),
    "rsi_entry": ("RSI 进场阈值", "多头需 RSI>50+该值，空头需 RSI<50−该值。填 16 即 66/34"),
    "trailing_percent": ("移动止损 %", "多头止损 = 持仓期最高价 ×(1 − 该值/100)"),
    "fixed_size": ("每次下单数量", "港股须为每手股数的整数倍"),
    # 其它随包示例策略的参数
    "boll_window": ("布林带周期", "计算布林带中轨用多少根 K 线"),
    "boll_dev": ("布林带标准差倍数", "上下轨 = 中轨 ± 该倍数 × 标准差"),
    "cci_window": ("CCI 周期", "计算 CCI（顺势指标）用多少根 K 线"),
    "cci_level": ("CCI 进场阈值", "CCI 超过 +该值转多、跌破 −该值转空"),
    "fast_window": ("快线周期", "快速均线用多少根 K 线，对价格更敏感"),
    "slow_window": ("慢线周期", "慢速均线用多少根 K 线，代表较长期方向"),
    "atr_window": ("ATR 周期", "计算 ATR（真实波幅）用多少根 K 线"),
    "rsi_window": ("RSI 周期", "计算 RSI 用多少根 K 线"),
    "rsi_level": ("RSI 进场阈值", "多头需 RSI≥50+该值，空头需 RSI≤50−该值"),
    "rsi_signal": ("RSI 信号周期", "RSI 子信号用多少根 K 线"),
    "entry_window": ("进场通道周期", "唐奇安通道：突破近 N 根 K 线的最高/最低价即进场"),
    "exit_window": ("离场通道周期", "跌破近 N 根 K 线的最低/最高价即离场，一般短于进场周期"),
    "kk_length": ("肯特纳通道周期", "计算中轨与波幅用多少根 K 线"),
    "kk_dev": ("肯特纳通道倍数", "上下轨 = 中轨 ± 该倍数 × 平均波幅"),
    "k1": ("上轨系数 K1", "上轨 = 开盘价 + K1 × 昨日振幅，突破即做多"),
    "k2": ("下轨系数 K2", "下轨 = 开盘价 − K2 × 昨日振幅，跌破即做空"),
    "sl_multiplier": ("止损 ATR 倍数", "止损距持仓期最高/最低价 = 该倍数 × ATR"),
    "test_trigger": ("测试触发间隔", "TestStrategy 专用：每收到多少个 tick 触发一次动作"),
}

# 运行时变量的中文标签。变量表是只读的实时状态，看不懂就等于没有。
_VARIABLE_LABELS: dict[str, tuple[str, str]] = {
    # 每个策略都有的三个（CtaTemplate 内置）
    "inited": ("已初始化", "历史数据是否加载完毕。False 时策略不会动作"),
    "trading": ("允许交易", "是否已启动。False 时只算指标、不发单"),
    "pos": ("当前持仓", "正数为多头、负数为空头、0 为空仓（单位：手/股）"),
    # 通用指标值
    "atr_value": ("ATR 当前值", "平均真实波幅，衡量当前波动大小"),
    "atr_ma": ("ATR 均线值", "ATR 自身的均值，用来判断波动是在放大还是收敛"),
    "rsi_value": ("RSI 当前值", "0-100，>50 偏多、<50 偏空"),
    "cci_value": ("CCI 当前值", "顺势指标当前读数"),
    # 通道 / 均线上下轨
    "entry_up": ("进场上轨", "近 N 根 K 线的最高价，向上突破它即做多"),
    "entry_down": ("进场下轨", "近 N 根 K 线的最低价，向下跌破它即做空"),
    "exit_up": ("离场上轨", "空头持仓时，涨破它即平仓"),
    "exit_down": ("离场下轨", "多头持仓时，跌破它即平仓"),
    "boll_up": ("布林带上轨", ""),
    "boll_down": ("布林带下轨", ""),
    "kk_up": ("肯特纳上轨", ""),
    "kk_down": ("肯特纳下轨", ""),
    # 均线
    "fast_ma": ("快线当前值", ""),
    "slow_ma": ("慢线当前值", ""),
    "fast_ma0": ("快线当前值", "本根 K 线的快线值"),
    "fast_ma1": ("快线前值", "上一根 K 线的快线值，与当前值比较判断金叉/死叉"),
    "slow_ma0": ("慢线当前值", "本根 K 线的慢线值"),
    "slow_ma1": ("慢线前值", "上一根 K 线的慢线值"),
    "ma_trend": ("均线方向", "1 为金叉（快线上穿慢线），−1 为死叉"),
    # 进出场价位
    "long_entry": ("多头进场价", "触及该价即开多"),
    "short_entry": ("空头进场价", "触及该价即开空"),
    "long_stop": ("多头止损价", "多头持仓跌破该价即平仓"),
    "short_stop": ("空头止损价", "空头持仓涨破该价即平仓"),
    "intra_trade_high": ("持仓期最高价", "本次持仓以来的最高价，移动止损以它为锚"),
    "intra_trade_low": ("持仓期最低价", "本次持仓以来的最低价，空头移动止损以它为锚"),
    # RSI 派生阈值
    "rsi_buy": ("RSI 做多线", "= 50 + RSI 进场阈值，RSI 高于它才做多"),
    "rsi_sell": ("RSI 做空线", "= 50 − RSI 进场阈值，RSI 低于它才做空"),
    "rsi_long": ("RSI 做多线", "= 50 + RSI 进场阈值"),
    "rsi_short": ("RSI 做空线", "= 50 − RSI 进场阈值"),
    # 其它
    "day_range": ("昨日振幅", "昨日最高价 − 最低价，用来推算今日上下轨"),
    "tick_count": ("已收 tick 数", "TestStrategy 专用计数器"),
    "test_all_done": ("测试是否完成", "TestStrategy 专用标志"),
}

_TYPE_NAMES: dict[type, str] = {str: "文本", int: "整数", float: "小数", bool: "是/否"}


def describe_parameter(name: str, type_: type) -> tuple[str, str]:
    """字段名 -> (显示标签, 悬停说明)。查不到就用原名，不编中文。

    参数与运行时变量共用这一个入口：两张表长得一样、坐在同一个面板里，
    对着看的人不区分"这是我填的"还是"这是策略算的"，只想知道它是什么。
    """
    label, hint = _PARAM_LABELS.get(name) or _VARIABLE_LABELS.get(name, ("", ""))
    shown = f"{label}（{name}）" if label else name
    type_name = _TYPE_NAMES.get(type_, type_.__name__)
    tip = f"{name} · {type_name}"
    if hint:
        tip = f"{tip}\n{hint}"
    return shown, tip


def _bad_vt_symbol(vt_symbol: str) -> str:
    """本地代码不合法时的说明；合法则返回空串。

    与引擎的判据保持一致（engine.py:679-686）：必须是 `代码.交易所`，
    且后缀得是 Exchange 里真实存在的那些。这里先拦一道，是为了让失败
    出现在用户还看着对话框的时候，而不是事后在日志里躺一行。
    """
    if not vt_symbol:
        return _("请填写交易标的，格式为 代码.交易所，如 700.SEHK、NVDA.SMART")
    if "." not in vt_symbol:
        return _("交易标的缺少交易所后缀：{}。正确写法形如 700.SEHK、NVDA.SMART").format(
            vt_symbol
        )
    exchange_str = vt_symbol.rsplit(".", 1)[1]
    if exchange_str not in Exchange.__members__:
        return _("交易所后缀 {} 不是有效交易所。美股用 SMART，港股用 SEHK。").format(
            exchange_str
        )
    return ""


class SettingEditor(QtWidgets.QDialog):
    """
    For creating new strategy and editing strategy parameters.
    """

    def __init__(
        self,
        parameters: dict,
        strategy_name: str = "",
        class_name: str = "",
        vt_symbols: list[str] | None = None,
    ) -> None:
        """vt_symbols 给出时，交易标的一栏变成可搜索下拉。

        这一栏要的是 `代码.交易所`（700.SEHK）。格式只写在 tooltip 里，
        得悬停才看得见，而输入框本身是个空白 —— 用户填了裸代码，对话框照常
        关闭，失败只落在日志面板一行"本地代码缺失交易所后缀"，很容易漏看。
        给出本地已知合约让人直接挑，格式问题就不会发生。
        """
        super().__init__()

        self.parameters: dict = parameters
        self.strategy_name: str = strategy_name
        self.class_name: str = class_name
        self.vt_symbols: list[str] = vt_symbols or []

        self.edits: dict = {}

        self.init_ui()

    def init_ui(self) -> None:
        """"""
        form: QtWidgets.QFormLayout = QtWidgets.QFormLayout()

        # Add vt_symbol and name edit if add new strategy
        if self.class_name:
            self.setWindowTitle(_("添加策略：{}").format(self.class_name))
            button_text: str = _("添加")
            parameters: dict = {"strategy_name": "", "vt_symbol": ""}
            parameters.update(self.parameters)
        else:
            self.setWindowTitle(_("参数编辑：{}").format(self.strategy_name))
            button_text = _("确定")
            parameters = self.parameters

        for name, value in parameters.items():
            type_: type = type(value)

            edit: QtWidgets.QWidget
            if name == "vt_symbol" and self.vt_symbols:
                edit = self._symbol_box()
            else:
                edit = QtWidgets.QLineEdit(str(value))
            if type_ is int:
                int_validator: QtGui.QIntValidator = QtGui.QIntValidator()
                edit.setValidator(int_validator)
            elif type_ is float:
                double_validator: QtGui.QDoubleValidator = QtGui.QDoubleValidator()
                edit.setValidator(double_validator)

            shown, tip = describe_parameter(name, type_)
            edit.setToolTip(tip)
            label = QtWidgets.QLabel(shown)
            label.setToolTip(tip)
            form.addRow(label, edit)

            self.edits[name] = (edit, type_)

        button: QtWidgets.QPushButton = QtWidgets.QPushButton(button_text)
        button.clicked.connect(self.accept)
        form.addRow(button)

        widget: QtWidgets.QWidget = QtWidgets.QWidget()
        widget.setLayout(form)

        scroll: QtWidgets.QScrollArea = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)

        vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        vbox.addWidget(scroll)
        self.setLayout(vbox)

    def _symbol_box(self) -> QtWidgets.QComboBox:
        """本地已知合约的可搜索下拉；不在列表里的代码仍可手输。

        用原生 QComboBox 而不是别处那个 SearchableComboBox：后者住在 vnpy_app,
        本包不依赖它，反向 import 会把依赖方向倒过来。
        """
        box: QtWidgets.QComboBox = QtWidgets.QComboBox()
        box.setEditable(True)
        box.addItem("")                             # 留一个空项，默认不预选任何标的
        box.addItems(self.vt_symbols)
        box.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)

        completer: QtWidgets.QCompleter = QtWidgets.QCompleter(self.vt_symbols, box)
        completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        box.setCompleter(completer)
        return box

    def get_setting(self) -> dict:
        """"""
        setting: dict = {}

        if self.class_name:
            setting["class_name"] = self.class_name

        for name, tp in self.edits.items():
            edit, type_ = tp
            # 下拉与输入框取值方式不同；下拉的显示文本可能带名称，取第一段。
            if isinstance(edit, QtWidgets.QComboBox):
                value_text = str(edit.currentText()).strip().split(" ", 1)[0]
            else:
                value_text = edit.text()

            # bool("False") 是 True，所以布尔项不能走 type_(value_text)
            value = (value_text == "True") if type_ is bool else type_(value_text)

            setting[name] = value

        return setting
