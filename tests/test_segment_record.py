"""三段切分记录（落盘 + 跨进程测试段预算）的验证测试。

`segments.SegmentedRunner` 的测试段预算是**进程内计数器**：一个
`SegmentedRunner` 对象活着的时候才拦得住第二次查看。命令行是"跑一次退出"
的形态，进程内计数器在第二次调用时归零 —— 预算等于没有。

`segment_record` 把预算搬到磁盘上，本文件钉住这件事真的成立：

  1. 记录能原样往返 JSON（GUI 与 CLI 是两个进程，只能靠文件对话）。
  2. 文件缺失 / 损坏 → 返回 None 而不是抛（GUI 不能因为一个研究文件起不来）。
  3. **同一份切分**再次打开时，已用掉的查看次数**留着**——这是跨进程预算。
  4. **切分边界变了**就是另一个测试集，次数重新计。
  5. 重开预算必须给理由，且理由落盘（与 `SegmentedRunner.reset_test_budget`
     同一口径）。
  6. `hits_test_segment` 与 `SegmentGuardedEngine.guard_optimization`
     **在同样的输入上给同样的答案** —— GUI 那一侧的拦截不能是另一套语义。
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from vnpy.trader.constant import Interval
from vnpy.trader.optimize import OptimizationSetting

from vnpy_ctastrategy.segment_record import (
    RECORD_FILENAME,
    SegmentPeek,
    SplitRecord,
    hits_test_segment,
    load_record,
    open_record,
    record_path,
    record_peek,
    reset_test_budget,
    save_record,
)
from vnpy_ctastrategy.segments import (
    Segment,
    SegmentGuardedEngine,
    SegmentLeakError,
    make_three_way_split,
)


def _dates(n: int) -> list[datetime]:
    base = datetime(2024, 1, 2)
    return [base + timedelta(days=i) for i in range(n)]


def _split(train: int = 180, valid: int = 60, test: int = 60):
    return make_three_way_split(_dates(300), train, valid, test, anchor="start")


def _open(path: Path, **kwargs: object) -> SplitRecord:
    defaults: dict = {
        "vt_symbol": "700.SEHK",
        "strategy_class": "LongOnlyTurtleStrategy",
        "interval": "d",
        "target_name": "sharpe_ratio",
        "test_budget": 1,
    }
    defaults.update(kwargs)
    split = defaults.pop("split", None) or _split()
    return open_record(split, path=path, **defaults)   # type: ignore[arg-type]


# ══════════════════════════════════════════════════════════════════════
# 1. 往返与容错
# ══════════════════════════════════════════════════════════════════════

def test_record_round_trips_through_json(tmp_path: Path) -> None:
    path = tmp_path / RECORD_FILENAME
    record = _open(path)
    record = record_peek(
        record,
        SegmentPeek(
            at=datetime(2026, 7, 26, 9, 30),
            setting={"entry_window": 20},
            target_name="sharpe_ratio",
            target_value=0.42,
            note="首次也是唯一一次查看",
        ),
    )
    save_record(record, path=path)

    loaded = load_record(path=path)
    assert loaded is not None
    assert loaded.split == record.split
    assert loaded.vt_symbol == "700.SEHK"
    assert loaded.strategy_class == "LongOnlyTurtleStrategy"
    assert len(loaded.peeks) == 1
    assert loaded.peeks[0].setting == {"entry_window": 20}
    assert loaded.peeks[0].target_value == pytest.approx(0.42)


def test_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_record(path=tmp_path / "nope.json") is None


def test_corrupt_file_returns_none_instead_of_raising(tmp_path: Path) -> None:
    """GUI 会在启动路径上读它 —— 一个坏文件不许让终端起不来。"""
    path = tmp_path / RECORD_FILENAME
    path.write_text("{ this is not json", encoding="utf-8")
    assert load_record(path=path) is None

    path.write_text(json.dumps({"split": {"train_start": "不是时间"}}), encoding="utf-8")
    assert load_record(path=path) is None


def test_default_path_lives_in_the_trader_dir() -> None:
    from vnpy.trader.utility import get_file_path

    assert record_path() == get_file_path(RECORD_FILENAME)


def test_save_creates_missing_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / RECORD_FILENAME
    save_record(_open(path), path=path)
    assert path.exists()


# ══════════════════════════════════════════════════════════════════════
# 2. 跨进程预算
# ══════════════════════════════════════════════════════════════════════

def test_reopening_the_same_split_keeps_the_views_already_spent(tmp_path: Path) -> None:
    """这就是整个模块存在的理由：换一个进程再来，预算不许回满。"""
    path = tmp_path / RECORD_FILENAME
    first = _open(path)
    assert first.remaining_test_budget() == 1

    spent = record_peek(
        first,
        SegmentPeek(datetime(2026, 7, 26), {"entry_window": 20}, "sharpe_ratio", 0.4),
    )
    save_record(spent, path=path)
    assert spent.remaining_test_budget() == 0

    # 新进程：同样的切分再打开一次
    again = _open(path)
    assert len(again.peeks) == 1
    assert again.remaining_test_budget() == 0


def test_a_different_split_starts_a_fresh_budget(tmp_path: Path) -> None:
    """边界变了就是另一个测试集，旧的查看次数不该压在新集上。"""
    path = tmp_path / RECORD_FILENAME
    spent = record_peek(
        _open(path),
        SegmentPeek(datetime(2026, 7, 26), {"entry_window": 20}, "sharpe_ratio", 0.4),
    )
    save_record(spent, path=path)

    other = _open(path, split=_split(train=150, valid=60, test=90))
    assert other.peeks == ()
    assert other.remaining_test_budget() == 1


def test_a_different_symbol_starts_a_fresh_budget(tmp_path: Path) -> None:
    path = tmp_path / RECORD_FILENAME
    spent = record_peek(
        _open(path),
        SegmentPeek(datetime(2026, 7, 26), {"entry_window": 20}, "sharpe_ratio", 0.4),
    )
    save_record(spent, path=path)

    other = _open(path, vt_symbol="9988.SEHK")
    assert other.peeks == ()


def test_budget_can_go_past_zero_and_stays_clamped(tmp_path: Path) -> None:
    path = tmp_path / RECORD_FILENAME
    record = _open(path)
    for i in range(3):
        record = record_peek(
            record,
            SegmentPeek(datetime(2026, 7, 26), {"i": i}, "sharpe_ratio", float(i)),
        )
    assert record.remaining_test_budget() == 0


def test_reset_demands_a_reason_and_leaves_a_trail(tmp_path: Path) -> None:
    path = tmp_path / RECORD_FILENAME
    record = record_peek(
        _open(path),
        SegmentPeek(datetime(2026, 7, 26), {"entry_window": 20}, "sharpe_ratio", 0.4),
    )
    assert record.remaining_test_budget() == 0

    for bad in ("", "   "):
        with pytest.raises(ValueError, match="reason"):
            reset_test_budget(record, bad)

    reopened = reset_test_budget(record, "换了成本假设，旧的样本外结论作废")
    assert reopened.remaining_test_budget() == 1
    assert reopened.peeks == ()
    assert any("换了成本假设" in line for line in reopened.resets)

    save_record(reopened, path=path)
    loaded = load_record(path=path)
    assert loaded is not None
    assert any("换了成本假设" in line for line in loaded.resets)


def test_record_is_immutable(tmp_path: Path) -> None:
    """记录是 frozen 的：每次变更返回新对象，不给"就地改一下"的余地。"""
    record = _open(tmp_path / RECORD_FILENAME)
    with pytest.raises(FrozenInstanceError):
        record.test_budget = 99          # type: ignore[misc]

    after = record_peek(
        record, SegmentPeek(datetime(2026, 7, 26), {}, "sharpe_ratio", 0.0)
    )
    assert record.peeks == ()
    assert len(after.peeks) == 1


# ══════════════════════════════════════════════════════════════════════
# 3. 与引擎守卫同解
# ══════════════════════════════════════════════════════════════════════

ENGINE_KWARGS: dict = {
    "vt_symbol": "700.SEHK",
    "interval": Interval.DAILY,
    "rate": 0.0013,
    "slippage": 0.0,
    "size": 1,
    "pricetick": 0.01,
    "capital": 1_000_000,
    "annual_days": 252,
}


def _engine_blocks(split, start: datetime, end: datetime) -> bool:
    engine = SegmentGuardedEngine()
    engine.bind_split(split)
    engine.set_parameters(start=start, end=end, **ENGINE_KWARGS)
    try:
        engine.run_bf_optimization(OptimizationSetting())
    except SegmentLeakError:
        return True
    return False


@pytest.mark.parametrize(
    "which",
    [
        "train_only",
        "train_and_valid",
        "valid_only",
        "whole_sample",
        "touches_test_by_one_bar",
        "test_only",
        "after_test",
    ],
)
def test_hits_test_segment_agrees_with_the_engine_guard(which: str) -> None:
    """GUI 用的判据必须与引擎守卫**逐例同解** —— 两套语义会给出两种答案。"""
    split = _split()
    windows = {
        "train_only": (split.train_start, split.train_end),
        "train_and_valid": (split.train_start, split.valid_end),
        "valid_only": (split.valid_start, split.valid_end),
        "whole_sample": (split.train_start, split.test_end),
        "touches_test_by_one_bar": (split.train_start, split.test_start),
        "test_only": (split.test_start, split.test_end),
        "after_test": (
            split.test_end + timedelta(days=1),
            split.test_end + timedelta(days=30),
        ),
    }
    start, end = windows[which]
    assert hits_test_segment(split, start, end) == _engine_blocks(split, start, end)


def test_hits_test_segment_is_date_granular() -> None:
    """GUI 的日期控件只给到"天"，所以判据必须按天算：
    TEST 段起始日当天的任何时刻都算碰到。"""
    split = _split()
    same_day_early = split.test_start.replace(hour=0, minute=0)
    assert hits_test_segment(split, split.train_start, same_day_early)
    assert not hits_test_segment(
        split, split.train_start, split.test_start - timedelta(days=1)
    )


def test_record_hits_test_delegates_to_the_same_judgement(tmp_path: Path) -> None:
    record = _open(tmp_path / RECORD_FILENAME)
    split = record.split
    assert record.hits_test(split.test_start, split.test_end)
    assert not record.hits_test(split.train_start, split.train_end)


# ══════════════════════════════════════════════════════════════════════
# 4. 人读文本（GUI 与 CLI 共用同一段话）
# ══════════════════════════════════════════════════════════════════════

def test_summary_is_compact_and_leaves_the_peek_detail_to_describe(
    tmp_path: Path,
) -> None:
    """`summary()` 给侧边栏那种窄地方用：三段边界 + 剩余次数，到此为止。

    每一次查看的参数字典（`{'entry_window': 10.0, ...}`）在终端里有用，塞进
    GUI 的标签只会把它挤到截断 —— 实测面板上那行就是被切在半个字典处的。
    """
    path = tmp_path / RECORD_FILENAME
    record = record_peek(
        _open(path),
        SegmentPeek(
            datetime(2026, 7, 26), {"entry_window": 10.0, "atr_stop": 1.5},
            "sharpe_ratio", -1.78,
        ),
    )

    summary = record.summary()
    assert len(summary.splitlines()) == 5, "紧凑版应当是 1 行抬头 + 3 段 + 1 行额度"
    assert "entry_window" not in summary
    assert "剩余 0 次" in summary
    for segment in Segment:
        assert segment.name in summary

    # 详版仍然给得出来（GUI 挂在 tooltip 上，命令行直接打印）
    assert "entry_window" in record.describe()
    assert record.describe().startswith(summary.splitlines()[0])


def test_describe_states_the_windows_and_the_remaining_budget(tmp_path: Path) -> None:
    record = _open(tmp_path / RECORD_FILENAME)
    text = record.describe()
    for segment in Segment:
        assert segment.name in text
    assert record.vt_symbol in text
    assert "剩余" in text

    spent = record_peek(
        record, SegmentPeek(datetime(2026, 7, 26), {}, "sharpe_ratio", 0.4)
    )
    assert "0" in spent.describe()
