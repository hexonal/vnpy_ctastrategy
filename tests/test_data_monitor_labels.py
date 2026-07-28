"""策略面板的参数表与变量表要看得懂。

用户报障：面板上两张表印的是 entry_window / entry_up / atr_value 这类原始
字段名，加上一个 `1.1276457926121324` —— 对着界面看盘的人认不出这是什么，
也读不完那串小数。

标签只覆盖框架自带字段与随包示例策略，且每一条都对着策略源码核过语义
（k1/k2 来自 dual_thrust、entry_up/exit_up 来自 turtle_signal 等）。
查不到的字段一律原样显示 —— 宁可显示原名，也不猜一个可能是错的中文。
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from vnpy.trader.ui import QtWidgets, create_qapp

from vnpy_ctastrategy.ui.widget import DataMonitor, _display, describe_parameter


@pytest.fixture(scope="module")
def qapp() -> QtWidgets.QApplication:
    existing = QtWidgets.QApplication.instance()
    if existing is not None:
        return existing                                     # type: ignore[return-value]
    return create_qapp()


TURTLE_VARIABLES = {
    "inited": True,
    "trading": False,
    "pos": 0,
    "entry_up": 220.9999,
    "entry_down": 216.11,
    "exit_up": 219.7499,
    "exit_down": 216.11,
    "atr_value": 1.1276457926121324,
}


def _headers(monitor: DataMonitor) -> list[str]:
    return [monitor.horizontalHeaderItem(i).text() for i in range(monitor.columnCount())]


def _cells(monitor: DataMonitor) -> list[str]:
    return [monitor.item(0, i).text() for i in range(monitor.columnCount())]


# ── 表头：中文标签 + 原名 ────────────────────────────────────────────

def test_variable_headers_are_readable(qapp: QtWidgets.QApplication) -> None:
    """用户截图里那一排。"""
    headers = _headers(DataMonitor(TURTLE_VARIABLES))
    assert "进场上轨（entry_up）" in headers
    assert "ATR 当前值（atr_value）" in headers
    assert "当前持仓（pos）" in headers


def test_parameter_headers_are_readable(qapp: QtWidgets.QApplication) -> None:
    headers = _headers(DataMonitor({"entry_window": 20, "exit_window": 10, "fixed_size": 1}))
    assert "进场通道周期（entry_window）" in headers
    assert "离场通道周期（exit_window）" in headers


def test_the_original_field_name_is_always_kept(qapp: QtWidgets.QApplication) -> None:
    """原名必须留着 —— 写策略的人按原名找代码，只给中文等于换了套黑话。"""
    for header in _headers(DataMonitor(TURTLE_VARIABLES)):
        assert "（" in header and header.endswith("）")


def test_unknown_fields_are_shown_verbatim(qapp: QtWidgets.QApplication) -> None:
    """自定义策略的字段查不到标签，就原样显示，不猜中文。"""
    monitor = DataMonitor({"my_custom_thing": 3})
    assert _headers(monitor) == ["my_custom_thing"]


def test_headers_carry_an_explanation(qapp: QtWidgets.QApplication) -> None:
    monitor = DataMonitor(TURTLE_VARIABLES)
    tip = monitor.horizontalHeaderItem(_headers(monitor).index("ATR 当前值（atr_value）")).toolTip()
    assert "atr_value" in tip and "波幅" in tip


def test_labels_cover_every_bundled_field(qapp: QtWidgets.QApplication) -> None:
    """随包策略的字段一个都不该漏 —— 漏一个就在界面上露出一个原名。

    直接从策略类的 parameters/variables 声明里取，加了新策略也会被这条挡住。
    """
    import importlib
    import inspect
    import pkgutil

    import vnpy_ctastrategy.strategies as pkg
    from vnpy_ctastrategy.template import CtaTemplate

    missing: list[str] = []
    for module_info in pkgutil.iter_modules(pkg.__path__):
        module = importlib.import_module(f"{pkg.__name__}.{module_info.name}")
        for _name, cls in inspect.getmembers(module, inspect.isclass):
            if not issubclass(cls, CtaTemplate) or cls is CtaTemplate:
                continue
            if cls.__module__ != module.__name__:
                continue
            for field in list(cls.parameters) + list(cls.variables):
                shown, _tip = describe_parameter(field, int)
                if shown == field:
                    missing.append(f"{cls.__name__}.{field}")

    assert not missing, f"这些字段还没有中文标签：{sorted(set(missing))}"


# ── 取值显示 ────────────────────────────────────────────────────────

def test_long_floats_are_trimmed(qapp: QtWidgets.QApplication) -> None:
    """1.1276457926121324 撑爆列宽，后面十几位对看盘没有意义。"""
    assert _display(1.1276457926121324) == "1.1276"


def test_full_precision_survives_in_the_tooltip(qapp: QtWidgets.QApplication) -> None:
    """截的只是显示，值一位都不能丢。"""
    monitor = DataMonitor(TURTLE_VARIABLES)
    column = _headers(monitor).index("ATR 当前值（atr_value）")
    assert monitor.item(0, column).toolTip() == str(1.1276457926121324)


def test_tiny_values_are_not_shown_as_zero(qapp: QtWidgets.QApplication) -> None:
    """截到 4 位后变成 0 但本身不是 0 —— 印 "0" 就成了谎报"没有值"。"""
    assert _display(0.00001234) != "0"
    assert _display(1e-12) != "0"


def test_real_zero_is_still_zero(qapp: QtWidgets.QApplication) -> None:
    assert _display(0.0) == "0"


def test_whole_floats_lose_the_trailing_zeros(qapp: QtWidgets.QApplication) -> None:
    assert _display(1.0) == "1"
    assert _display(216.11) == "216.11"


def test_booleans_read_as_yes_no(qapp: QtWidgets.QApplication) -> None:
    """已初始化/允许交易是给人看的状态，True/False 是程序员的写法。"""
    assert _display(True) == "是"
    assert _display(False) == "否"


def test_updates_keep_the_same_formatting(qapp: QtWidgets.QApplication) -> None:
    """实时刷新走的是另一条路（update_data），格式必须一致。"""
    monitor = DataMonitor(TURTLE_VARIABLES)
    monitor.update_data({**TURTLE_VARIABLES, "atr_value": 2.987654321, "trading": True})

    cells = _cells(monitor)
    assert "2.9877" in cells
    assert "是" in cells
