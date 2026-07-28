"""添加策略时，交易标的一栏要能选，填错要当场看得见。

用户报障：日志面板出现"创建策略失败，本地代码缺失交易所后缀"。这一栏要的是
`代码.交易所`（700.SEHK），但格式只写在 tooltip 里 —— 得悬停才看得见，
输入框本身是个空白。填了裸代码后对话框照常关闭，失败只落在日志里一行，
用户看到的是"点了添加，什么都没发生"。

两件事分别对应下面两组用例：
- 给出本地已知合约让人直接挑（格式问题不会发生）；
- 真填错时在对话框还开着的时候就说清楚（引擎侧那两句 write_log 太靠后）。
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from vnpy.trader.ui import QtWidgets, create_qapp

from vnpy_ctastrategy.ui.widget import SettingEditor, _bad_vt_symbol


@pytest.fixture(scope="module")
def qapp() -> QtWidgets.QApplication:
    existing = QtWidgets.QApplication.instance()
    if existing is not None:
        return existing                                     # type: ignore[return-value]
    return create_qapp()


SYMBOLS = ["700.SEHK 腾讯控股", "NVDA.SMART NVIDIA CORP", "600519.SSE 贵州茅台"]


def _editor(vt_symbols: list[str] | None) -> SettingEditor:
    return SettingEditor({"fixed_size": 1}, class_name="DemoStrategy", vt_symbols=vt_symbols)


# ── 校验：填错要说清楚哪儿错了 ───────────────────────────────────────

def test_missing_suffix_is_named(qapp: QtWidgets.QApplication) -> None:
    """用户实际撞上的那一个。"""
    problem = _bad_vt_symbol("NEBIUS")
    assert "NEBIUS" in problem
    assert "SEHK" in problem or "SMART" in problem, "只说缺后缀不够，得给个正确的样子"


def test_empty_input_is_named(qapp: QtWidgets.QApplication) -> None:
    assert _bad_vt_symbol("") != ""


def test_bogus_exchange_is_named(qapp: QtWidgets.QApplication) -> None:
    problem = _bad_vt_symbol("NVDA.NASDAQQ")
    assert "NASDAQQ" in problem
    assert "SMART" in problem, "美股其实要用 SMART，得指出来"


def test_valid_symbols_pass(qapp: QtWidgets.QApplication) -> None:
    for good in ("700.SEHK", "NVDA.SMART", "600519.SSE", "IF88.CFFEX"):
        assert _bad_vt_symbol(good) == "", good


def test_validation_matches_the_engine(qapp: QtWidgets.QApplication) -> None:
    """判据必须和引擎一致，否则会出现"这里放行、引擎再拒"的两套标准。

    引擎的规则见 engine.py:679-686：必须含 '.'，且后缀在 Exchange.__members__ 里。
    """
    from vnpy.trader.constant import Exchange

    for name in list(Exchange.__members__)[:8]:
        assert _bad_vt_symbol(f"X.{name}") == ""
    assert _bad_vt_symbol("X.NOT_AN_EXCHANGE") != ""


# ── 下拉：能选，且取值是纯代码不是显示文本 ───────────────────────────

def test_symbol_field_is_a_dropdown_when_contracts_are_known(
    qapp: QtWidgets.QApplication,
) -> None:
    editor = _editor(SYMBOLS)
    edit, _type = editor.edits["vt_symbol"]
    assert isinstance(edit, QtWidgets.QComboBox)
    assert edit.isEditable(), "列表外的代码仍要能手输"


def test_falls_back_to_a_plain_field_without_contracts(qapp: QtWidgets.QApplication) -> None:
    """没连网关时也得能用 —— 手输是唯一的路，别把它挡掉。"""
    editor = _editor(None)
    edit, _type = editor.edits["vt_symbol"]
    assert isinstance(edit, QtWidgets.QLineEdit)


def test_picking_from_the_list_yields_the_bare_code(qapp: QtWidgets.QApplication) -> None:
    """显示带名称，取值必须只取第一段。

    这是"改显示就得改读取"的配对：不取第一段的话，
    "700.SEHK 腾讯控股" 整串会被当成本地代码传给引擎。
    """
    editor = _editor(SYMBOLS)
    edit, _type = editor.edits["vt_symbol"]
    edit.setCurrentIndex(edit.findText("700.SEHK 腾讯控股"))

    assert editor.get_setting()["vt_symbol"] == "700.SEHK"


def test_typed_free_text_is_kept(qapp: QtWidgets.QApplication) -> None:
    """列表里没有的代码照样能提交（刚连上、合约还没查完时的正当用法）。"""
    editor = _editor(SYMBOLS)
    edit, _type = editor.edits["vt_symbol"]
    edit.setEditText("AAPL.SMART")

    assert editor.get_setting()["vt_symbol"] == "AAPL.SMART"


def test_default_selection_is_blank(qapp: QtWidgets.QApplication) -> None:
    """别默认预选第一个合约 —— 那会让人不看就点添加，建到一个没打算交易的标的上。"""
    editor = _editor(SYMBOLS)
    assert editor.get_setting()["vt_symbol"] == ""


def test_other_parameters_are_untouched(qapp: QtWidgets.QApplication) -> None:
    """只有 vt_symbol 换成了下拉，别的字段仍是输入框、类型转换不变。"""
    editor = _editor(SYMBOLS)
    edit, _type = editor.edits["fixed_size"]
    assert isinstance(edit, QtWidgets.QLineEdit)

    edit.setText("25")
    assert editor.get_setting()["fixed_size"] == 25
