"""参数对话框的标签映射。

SettingEditor 靠反射生成界面，原先标签是 `f"{name} {type_}"` —— 屏幕上就是
`atr_length <class 'int'>`。参数名对写策略的人够用，但对着界面填参数的人
看不出 rsi_entry 是"阈值"还是"周期"，而 `<class 'int'>` 是 Python 的 repr
泄漏到了 UI 上。

这里钉住三件事：标签是给人看的、类型信息不丢（进提示）、查不到时回落原名
而不是编一个可能是错的中文。
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _describe(name: str, type_: type) -> tuple[str, str]:
    from vnpy_ctastrategy.ui.widget import describe_parameter

    return describe_parameter(name, type_)


@pytest.mark.parametrize(
    ("name", "type_", "expect_in_label"),
    [
        ("vt_symbol", str, "交易标的"),
        ("strategy_name", str, "策略实例名"),
        ("rsi_entry", int, "RSI 进场阈值"),
        ("trailing_percent", float, "移动止损"),
        ("fixed_size", int, "每次下单数量"),
    ],
)
def test_known_parameters_get_a_readable_label(
    name: str, type_: type, expect_in_label: str
) -> None:
    label, _tip = _describe(name, type_)
    assert expect_in_label in label


def test_label_keeps_the_original_name_visible() -> None:
    """中文旁边要保留参数名 —— 看文档、翻源码、问人时用的都是它。"""
    label, _tip = _describe("atr_length", int)
    assert "atr_length" in label and "ATR 周期" in label


def test_type_moves_into_the_tooltip_not_the_label() -> None:
    """类型信息不该丢，但也不该占着标签。"""
    label, tip = _describe("atr_length", int)
    assert "class" not in label, "Python 的 repr 又泄漏到标签上了"
    assert "整数" in tip and "atr_length" in tip


def test_unknown_parameter_falls_back_to_its_own_name() -> None:
    """查不到就显示原名 —— 宁可不翻译，也不猜一个可能是错的中文。

    自定义策略的参数名千奇百怪，映射表不可能穷举。猜错的中文比不翻译更糟：
    它会让人按错误的理解去填参数，而参数是要用真钱下单的。
    """
    label, tip = _describe("my_custom_knob", float)
    assert label == "my_custom_knob"
    assert "小数" in tip


def test_every_atr_rsi_parameter_is_covered() -> None:
    """随包示例策略的参数必须全部有标签 —— 它是新用户第一个打开的对话框。"""
    from vnpy_ctastrategy.strategies.atr_rsi_strategy import AtrRsiStrategy

    for name in AtrRsiStrategy.parameters:
        label, _tip = _describe(name, int)
        assert label != name, f"{name} 没有中文标签，用户看到的还是原始参数名"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
