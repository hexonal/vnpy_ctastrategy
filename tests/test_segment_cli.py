"""三段回测命令行入口的验证测试。

命令行是三段流程的**唯一执行入口**（图形界面刻意不提供，理由见
`vnpy_app/fluent_ui/backtester_segments.py` 的模块文档）。既然是唯一入口，
纪律必须长在入口本身，而不是"记得加那个参数"：

  1. **默认档 `select` 碰不到 TEST 段** —— 随手敲一条命令不该烧掉测试集。
     看测试段必须显式 `--stage final`。
  2. `final` 档消耗账本额度，额度用尽后**拒绝执行**，除非显式
     `--reset-test-budget "理由"`（空理由不算理由）。
  3. 选参只看 TRAIN —— 用构造性反例钉住（一组"TRAIN 最差、TEST 最好"的
     参数不许被选中）。
  4. 打印文本必须写明哪些数字是样本内的。
  5. `python -m vnpy_ctastrategy.segment_cli` 真的能作为模块跑起来。
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from pandas import DataFrame

from vnpy_ctastrategy.overfitting import annualised_sharpe, daily_log_returns
from vnpy_ctastrategy.segment_cli import (
    FINAL,
    SELECT,
    authorize_final,
    build_parser,
    plan_split,
    run_stage,
)
from vnpy_ctastrategy.segment_record import (
    RECORD_FILENAME,
    SegmentBudgetRefused,
    SegmentPeek,
    open_record,
    record_peek,
    save_record,
)
from vnpy_ctastrategy.segments import (
    Segment,
    SegmentedRunner,
    make_three_way_split,
)

ANNUAL_DAYS = 252
CAPITAL = 1_000_000.0


# ══════════════════════════════════════════════════════════════════════
# 工具（与 tests/test_segment_split.py 的 FakeRunner 同构）
# ══════════════════════════════════════════════════════════════════════

def _dates(n: int) -> list[datetime]:
    base = datetime(2024, 1, 2)
    return [base + timedelta(days=i) for i in range(n)]


class FakeRunner:
    """一张预先生成好的 (T, N) 逐日盈亏表当回测引擎用。"""

    def __init__(
        self, dates: list[datetime], pnl: np.ndarray, settings: list[dict]
    ) -> None:
        self.dates = dates
        self.pnl = pnl
        self.keys = [tuple(sorted(s.items())) for s in settings]

    def __call__(self, setting: dict, start: datetime, end: datetime) -> DataFrame:
        col = self.keys.index(tuple(sorted(setting.items())))
        mask = [i for i, d in enumerate(self.dates) if start <= d <= end]
        if not mask:
            return DataFrame()
        return DataFrame(
            {
                "net_pnl": self.pnl[mask, col],
                "trade_count": np.ones(len(mask)),
            },
            index=[self.dates[i].date() for i in mask],
        )

    @staticmethod
    def statistics(df: DataFrame) -> dict:
        if df is None or df.empty:
            return {"sharpe_ratio": 0.0, "annual_return": 0.0, "total_days": 0}
        r = daily_log_returns(df["net_pnl"].to_numpy(), CAPITAL)
        total = (float(np.sum(df["net_pnl"].to_numpy())) / CAPITAL) * 100.0
        return {
            "sharpe_ratio": annualised_sharpe(r, ANNUAL_DAYS),
            "annual_return": total / len(df) * ANNUAL_DAYS,
            "total_days": len(df),
        }


def _staged_pnl(dts, split, per_segment: dict) -> np.ndarray:
    """按段给每组参数一个恒定日盈亏，段外为 0。"""
    n_settings = len(next(iter(per_segment.values())))
    pnl = np.zeros((len(dts), n_settings))
    for segment, values in per_segment.items():
        lo, hi = split.as_period(segment)
        rows = [i for i, d in enumerate(dts) if lo <= d <= hi]
        for col, value in enumerate(values):
            pnl[rows, col] = value
    return pnl


def _fixture(per_segment: dict, test_budget: int = 1):
    dts = _dates(150)
    settings = [{"window": 10 + i} for i in range(len(next(iter(per_segment.values()))))]
    split = make_three_way_split(dts, 90, 30, 30, anchor="start")
    runner = SegmentedRunner(
        runner=FakeRunner(dts, _staged_pnl(dts, split, per_segment), settings),
        split=split,
        statistics_func=FakeRunner.statistics,
        capital=CAPITAL,
        test_budget=test_budget,
    )
    return runner, settings, split


def _open(path: Path, split, **kwargs: object):
    defaults: dict = {
        "vt_symbol": "700.SEHK",
        "strategy_class": "LongOnlyTurtleStrategy",
        "interval": "d",
        "target_name": "annual_return",
        "test_budget": 1,
    }
    defaults.update(kwargs)
    return open_record(split, path=path, **defaults)   # type: ignore[arg-type]


# ══════════════════════════════════════════════════════════════════════
# 1. 默认档不烧测试集
# ══════════════════════════════════════════════════════════════════════

def test_default_stage_is_select() -> None:
    """随手敲一条命令不该看掉测试集 —— 看它必须显式要求。"""
    args = build_parser().parse_args([])
    assert args.stage == SELECT


def test_select_stage_never_touches_the_test_segment() -> None:
    runner, settings, _ = _fixture(
        {
            Segment.TRAIN: [3_000.0, 1_000.0],
            Segment.VALID: [2_000.0, 1_000.0],
            Segment.TEST: [-5_000.0, 9_000.0],
        }
    )
    outcome = run_stage(
        runner, settings, stage=SELECT, target_name="annual_return"
    )

    assert outcome.test is None
    assert outcome.report is None
    assert runner.test_calls == 0, "select 档不许消耗测试段预算"
    assert runner.test_audit == []


def test_select_stage_picks_the_train_argmax_not_the_test_argmax() -> None:
    """构造性反例：参数 1 在 TEST 上最好、TRAIN 上最差，不许被选中。"""
    runner, settings, _ = _fixture(
        {
            Segment.TRAIN: [3_000.0, -2_000.0],
            Segment.VALID: [2_000.0, -1_000.0],
            Segment.TEST: [-5_000.0, 9_000.0],
        }
    )
    outcome = run_stage(
        runner, settings, stage=SELECT, target_name="annual_return"
    )
    assert outcome.chosen_index == 0
    assert outcome.chosen_setting == settings[0]


def test_select_warns_when_valid_contradicts_train() -> None:
    """VALID 段存在的意义就是这一条：选中的参数在另一段样本内数据上翻车了，
    必须当场说出来。判据与 `run_holdout` 逐字相同（VALID ≤ 0 < TRAIN）。"""
    runner, settings, _ = _fixture(
        {
            Segment.TRAIN: [3_000.0, 1_000.0],
            Segment.VALID: [-4_000.0, -1_000.0],
            Segment.TEST: [1.0, 1.0],
        }
    )
    outcome = run_stage(
        runner, settings, stage=SELECT, target_name="annual_return"
    )
    assert outcome.warnings, "TRAIN 正 VALID 负却一声不吭 = VALID 白跑了"
    assert any("VALID" in note for note in outcome.warnings)
    assert "⚠" in outcome.text()


def test_select_is_quiet_when_train_and_valid_agree() -> None:
    runner, settings, _ = _fixture(
        {
            Segment.TRAIN: [3_000.0, 1_000.0],
            Segment.VALID: [2_000.0, 1_000.0],
            Segment.TEST: [1.0, 1.0],
        }
    )
    outcome = run_stage(
        runner, settings, stage=SELECT, target_name="annual_return"
    )
    assert outcome.warnings == []


def test_select_text_says_the_numbers_are_in_sample() -> None:
    runner, settings, _ = _fixture(
        {
            Segment.TRAIN: [3_000.0, 1_000.0],
            Segment.VALID: [2_000.0, 1_000.0],
            Segment.TEST: [1.0, 1.0],
        }
    )
    text = run_stage(
        runner, settings, stage=SELECT, target_name="annual_return"
    ).text()
    assert "[样本内]" in text
    assert "TRAIN" in text and "VALID" in text
    # 每一行统计量都带 [样本内]/[样本外] 标记；select 档一行样本外都不该有。
    assert "[样本外]" not in text, "select 档没有任何样本外数字，不许印出这个标记"
    assert "[未查看]" in text, "必须写明测试段是【按纪律没看】，不是【算不出来】"


# ══════════════════════════════════════════════════════════════════════
# 2. final 档与账本额度
# ══════════════════════════════════════════════════════════════════════

def test_final_stage_looks_at_the_test_segment_exactly_once() -> None:
    runner, settings, _ = _fixture(
        {
            Segment.TRAIN: [3_000.0, 1_000.0],
            Segment.VALID: [2_000.0, 1_000.0],
            Segment.TEST: [500.0, 400.0],
        }
    )
    outcome = run_stage(
        runner, settings, stage=FINAL, target_name="annual_return",
        significance=False,
    )
    assert outcome.test is not None
    assert outcome.test.out_of_sample is True
    assert runner.test_calls == 1
    assert "[样本外]" in outcome.text()


def test_final_is_refused_once_the_ledger_budget_is_gone(tmp_path: Path) -> None:
    path = tmp_path / RECORD_FILENAME
    _, _, split = _fixture(
        {Segment.TRAIN: [1.0], Segment.VALID: [1.0], Segment.TEST: [1.0]}
    )
    spent = record_peek(
        _open(path, split),
        SegmentPeek(datetime(2026, 7, 26), {"window": 10}, "annual_return", 1.0),
    )
    save_record(spent, path=path)

    reopened = _open(path, split)
    assert reopened.remaining_test_budget() == 0
    with pytest.raises(SegmentBudgetRefused, match="reset"):
        authorize_final(reopened, reset_reason=None)


def test_final_is_allowed_again_after_an_explicit_reset(tmp_path: Path) -> None:
    path = tmp_path / RECORD_FILENAME
    _, _, split = _fixture(
        {Segment.TRAIN: [1.0], Segment.VALID: [1.0], Segment.TEST: [1.0]}
    )
    spent = record_peek(
        _open(path, split),
        SegmentPeek(datetime(2026, 7, 26), {"window": 10}, "annual_return", 1.0),
    )
    save_record(spent, path=path)

    reopened = authorize_final(
        _open(path, split), reset_reason="换了成本假设，旧结论作废"
    )
    assert reopened.remaining_test_budget() == 1
    assert any("换了成本假设" in line for line in reopened.resets)


def test_reset_with_a_blank_reason_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / RECORD_FILENAME
    _, _, split = _fixture(
        {Segment.TRAIN: [1.0], Segment.VALID: [1.0], Segment.TEST: [1.0]}
    )
    with pytest.raises(ValueError, match="reason"):
        authorize_final(_open(path, split), reset_reason="   ")


def test_authorize_passes_through_when_budget_remains(tmp_path: Path) -> None:
    path = tmp_path / RECORD_FILENAME
    _, _, split = _fixture(
        {Segment.TRAIN: [1.0], Segment.VALID: [1.0], Segment.TEST: [1.0]}
    )
    record = _open(path, split)
    assert authorize_final(record, reset_reason=None) is record


# ══════════════════════════════════════════════════════════════════════
# 3. 切分规划
# ══════════════════════════════════════════════════════════════════════

def test_plan_split_by_bars_anchors_test_at_the_newest_data() -> None:
    dts = _dates(300)
    args = build_parser().parse_args(
        ["--train-bars", "180", "--valid-bars", "60", "--test-bars", "40"]
    )
    split = plan_split(dts, args)
    assert (split.train_bars, split.valid_bars, split.test_bars) == (180, 60, 40)
    assert split.test_end == dts[-1], "anchor=end：TEST 必须贴着最新的数据"


def test_plan_split_by_ratio_uses_every_bar() -> None:
    dts = _dates(300)
    args = build_parser().parse_args(["--ratio", "0.6", "0.2", "0.2"])
    split = plan_split(dts, args)
    assert split.train_bars + split.valid_bars + split.test_bars == 300


def test_plan_split_rejects_a_sample_too_short_for_three_segments() -> None:
    args = build_parser().parse_args(
        ["--train-bars", "180", "--valid-bars", "60", "--test-bars", "60"]
    )
    with pytest.raises(ValueError, match="样本"):
        plan_split(_dates(100), args)


def test_stage_choices_are_only_select_and_final() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--stage", "peek-a-bit"])


# ══════════════════════════════════════════════════════════════════════
# 4. 真的能作为模块跑
# ══════════════════════════════════════════════════════════════════════

def test_module_is_runnable_and_help_names_the_stages() -> None:
    """`python -m vnpy_ctastrategy.segment_cli --help` —— GUI 提示里印的就是它。"""
    result = subprocess.run(
        [sys.executable, "-m", "vnpy_ctastrategy.segment_cli", "--help"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "--stage" in result.stdout
    assert "final" in result.stdout
