"""把三段切分与测试段预算落到磁盘 —— 让纪律活过进程边界。

`segments.SegmentedRunner` 的测试段预算是**进程内计数器**（`_test_calls`）。
那是它该有的形态：一次研究流程 = 一个 runner 对象 = 一份预算。但它有一个
它管不到的边界 ——

    进程退出，计数器归零。

命令行工具"跑一次就退出"，图形界面是长驻进程但每次点按钮可能新建对象。
两种形态下"只看一次测试段"都会退化成"想看几次看几次"。本模块补的就是
这一层：把**切分边界 + 已经用掉的查看次数**写进一个 JSON，下一个进程打开
同一份切分时接着数。

━━━ 身份判定：什么叫"同一份切分" ━━━

`(vt_symbol, strategy_class, interval, 六个日期边界)` 全等才算同一份。
任何一项变了就是**另一个测试集**，预算从头计 —— 这不是漏洞，是定义：
把 TEST 段往后挪一天，那一段数据确实还没被看过。

反过来说，这也是本模块唯一能被"合法"绕过的方式（挪一天边界换一次查看）。
它绕不掉的是**留痕**：文件里记着上一份切分是什么、看过几次。诚实的研究
需要的是账本，不是牢房。

━━━ 与 `SegmentedRunner` 的关系 ━━━

两者不互相依赖，也不互相替代：

    SegmentedRunner   一次流程内的预算（对象活着的时候）
    segment_record    跨流程的账本（进程之间）

命令行入口 `segment_cli` 把两者串起来：开跑前查账本还有没有额度，
跑完把这一次查看记回账本。

━━━ 容错口径：读取一律 fail-open ━━━

`load_record` 在文件缺失 / 损坏 / 字段不认识时**返回 None，不抛**。
理由是这个文件会被 GUI 在启动路径上读 —— 一个写坏的研究文件不该让整个
交易终端起不来。写入侧则相反：`save_record` 的任何失败都要抛出来，
因为"以为记上了其实没记"比"记不上"更危险。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .segments import Segment, SegmentBudgetExhaustedError, ThreeWaySplit

__all__ = [
    "RECORD_FILENAME",
    "SegmentBudgetRefused",
    "SegmentPeek",
    "SplitRecord",
    "hits_test_segment",
    "load_record",
    "open_record",
    "record_path",
    "record_peek",
    "reset_test_budget",
    "save_record",
]


class SegmentBudgetRefused(SegmentBudgetExhaustedError):
    """账本里这份切分的测试段额度已经用完。

    刻意继承 `SegmentBudgetExhaustedError`：进程内预算与跨进程账本拦的是
    同一件事（"测试段看太多次"），调用方按老类型 catch 也能接住新的这条。
    """



RECORD_FILENAME = "cta_holdout_split.json"

# 文件格式版本。读到不认识的版本一律当"读不懂"（返回 None），
# 而不是按当前字段硬解析出一份似是而非的记录。
SCHEMA_VERSION = 1

_SPLIT_DATETIME_FIELDS = (
    "train_start", "train_end",
    "valid_start", "valid_end",
    "test_start", "test_end",
)
_SPLIT_INT_FIELDS = ("train_bars", "valid_bars", "test_bars")


def record_path(path: str | Path | None = None) -> Path:
    """账本路径。默认落在 vnpy 的用户目录（与 GUI 读的是同一个位置）。"""
    if path is not None:
        return Path(path)
    from vnpy.trader.utility import get_file_path

    return Path(get_file_path(RECORD_FILENAME))


@dataclass(frozen=True)
class SegmentPeek:
    """一次"看了测试段"的记录。

    记下参数与目标值，是为了让账本能回答"上次看的是哪一组参数、看到了什么"——
    没有这两项的计数器只能说"用完了"，说不清用在了哪里。
    """

    at: datetime
    setting: dict[str, Any]
    target_name: str
    target_value: float
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "at": self.at.isoformat(),
            "setting": dict(self.setting),
            "target_name": self.target_name,
            "target_value": float(self.target_value),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SegmentPeek:
        return cls(
            at=datetime.fromisoformat(str(raw["at"])),
            setting=dict(raw.get("setting") or {}),
            target_name=str(raw.get("target_name", "")),
            target_value=float(raw.get("target_value", float("nan"))),
            note=str(raw.get("note", "")),
        )

    def describe(self) -> str:
        text = (
            f"{self.at:%Y-%m-%d %H:%M} "
            f"{self.target_name}={self.target_value:.4f} {self.setting}"
        )
        return f"{text}（{self.note}）" if self.note else text


@dataclass(frozen=True)
class SplitRecord:
    """一份切分的账本：切在哪里、属于谁、测试段被看过几次。"""

    split: ThreeWaySplit
    vt_symbol: str
    strategy_class: str
    interval: str
    target_name: str
    test_budget: int
    created_at: datetime
    updated_at: datetime
    peeks: tuple[SegmentPeek, ...] = ()
    resets: tuple[str, ...] = ()

    def identity(self) -> tuple[Any, ...]:
        """"同一份切分"的判据。任何一项不同即另一个测试集。"""
        return (
            self.vt_symbol,
            self.strategy_class,
            self.interval,
            *(getattr(self.split, name) for name in _SPLIT_DATETIME_FIELDS),
        )

    def remaining_test_budget(self) -> int:
        """还能看几次测试段（不为负）。"""
        return max(0, int(self.test_budget) - len(self.peeks))

    def hits_test(self, start: datetime, end: datetime) -> bool:
        """窗口 [start, end] 有没有碰到 TEST 段。"""
        return hits_test_segment(self.split, start, end)

    def summary(self) -> str:
        """紧凑版：抬头 + 三段边界 + 剩余额度，固定 5 行。

        逐次查看的参数字典（`{'entry_window': 10.0, ...}`）留给 `describe()`：
        它在终端里有用，但在任何宽度受限的地方都会把版面撑爆。
        """
        lines = [
            f"{self.vt_symbol} / {self.strategy_class} / {self.interval} "
            f"（目标 {self.target_name}）",
        ]
        for segment in Segment:
            lo, hi = self.split.as_period(segment)
            mark = "样本外" if segment is Segment.TEST else "样本内"
            lines.append(
                f"  {segment.name:<5} [{mark}] {lo:%Y-%m-%d} ~ {hi:%Y-%m-%d}"
                f"  {self.split.bars(segment)} 根"
            )
        lines.append(
            f"  TEST 段已查看 {len(self.peeks)}/{self.test_budget} 次，"
            f"剩余 {self.remaining_test_budget()} 次"
        )
        return "\n".join(lines)

    def describe(self) -> str:
        """详版：`summary()` 再加上每一次查看与每一次重开的明细。"""
        lines = [self.summary()]
        lines.extend(f"    · {peek.describe()}" for peek in self.peeks)
        lines.extend(f"    ⟲ {reason}" for reason in self.resets)
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "split": {
                **{
                    name: getattr(self.split, name).isoformat()
                    for name in _SPLIT_DATETIME_FIELDS
                },
                **{name: int(getattr(self.split, name)) for name in _SPLIT_INT_FIELDS},
            },
            "vt_symbol": self.vt_symbol,
            "strategy_class": self.strategy_class,
            "interval": self.interval,
            "target_name": self.target_name,
            "test_budget": int(self.test_budget),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "peeks": [view.as_dict() for view in self.peeks],
            "resets": list(self.resets),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SplitRecord:
        """反序列化。字段缺失 / 类型不对一律抛 —— 由 `load_record` 兜成 None。"""
        version = int(raw.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise ValueError(f"账本版本 {version} 不是本模块认识的 {SCHEMA_VERSION}")

        raw_split = raw["split"]

        def moment(name: str) -> datetime:
            return datetime.fromisoformat(str(raw_split[name]))

        # 逐个具名传入而不是两个 `**dict` splat：splat 的值类型是
        # dict[str, datetime] / dict[str, int]，类型检查器没法确认它们各自落到
        # 对的形参上（mypy 实测报 arg-type）。字段就九个，写全了换来可检查。
        split = ThreeWaySplit(
            train_start=moment("train_start"), train_end=moment("train_end"),
            valid_start=moment("valid_start"), valid_end=moment("valid_end"),
            test_start=moment("test_start"), test_end=moment("test_end"),
            train_bars=int(raw_split["train_bars"]),
            valid_bars=int(raw_split["valid_bars"]),
            test_bars=int(raw_split["test_bars"]),
        )
        return cls(
            split=split,
            vt_symbol=str(raw["vt_symbol"]),
            strategy_class=str(raw["strategy_class"]),
            interval=str(raw["interval"]),
            target_name=str(raw["target_name"]),
            test_budget=int(raw["test_budget"]),
            created_at=datetime.fromisoformat(str(raw["created_at"])),
            updated_at=datetime.fromisoformat(str(raw["updated_at"])),
            peeks=tuple(
                SegmentPeek.from_dict(item) for item in raw.get("peeks") or ()
            ),
            resets=tuple(str(item) for item in raw.get("resets") or ()),
        )


def hits_test_segment(split: ThreeWaySplit, start: datetime, end: datetime) -> bool:
    """窗口 [start, end] 与 TEST 段有没有交集（按【天】判）。

    判据刻意与 `segments.SegmentGuardedEngine.guard_optimization` 完全一致：
    TEST 段按 [test_start 当天 00:00:00, test_end 当天 23:59:59.999999] 展开，
    与窗口做区间相交。图形界面的日期控件只给到"天"，两侧用同一把尺子才不会
    出现"引擎拦了但界面说没事"。`tests/test_segment_record.py` 用逐例对拍
    钉住这一点。
    """
    lo = _naive(start)
    hi = _naive(end)
    t0 = _naive(split.test_start).replace(hour=0, minute=0, second=0, microsecond=0)
    t1 = _naive(split.test_end).replace(
        hour=23, minute=59, second=59, microsecond=999999
    )
    return not (hi < t0 or lo > t1)


def _naive(moment: datetime) -> datetime:
    return moment.replace(tzinfo=None) if moment.tzinfo is not None else moment


def load_record(path: str | Path | None = None) -> SplitRecord | None:
    """读账本。**缺失 / 损坏 / 版本不认识一律返回 None，绝不抛。**

    GUI 在启动路径上调它 —— 一个写坏的研究文件不该让交易终端起不来。
    """
    target = record_path(path)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        return SplitRecord.from_dict(raw)
    except (KeyError, TypeError, ValueError):
        return None


def save_record(record: SplitRecord, path: str | Path | None = None) -> Path:
    """写账本（先写临时文件再原子改名）。失败照抛 —— 静默漏记比写不进去更危险。"""
    target = record_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(record.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, target)
    return target


def open_record(
    split: ThreeWaySplit,
    *,
    vt_symbol: str,
    strategy_class: str,
    interval: str,
    target_name: str,
    test_budget: int = 1,
    path: str | Path | None = None,
    now: datetime | None = None,
) -> SplitRecord:
    """打开这一份切分的账本：**同一份就接着数，换了一份就重新开一本。**

    "同一份"由 `SplitRecord.identity()` 判定（标的 / 策略 / 周期 / 六个边界）。
    换了边界就是另一段没被看过的数据，预算从头计。
    """
    if int(test_budget) < 1:
        raise ValueError(f"test_budget 必须 ≥1，收到 {test_budget}")

    moment = now or datetime.now()
    fresh = SplitRecord(
        split=split,
        vt_symbol=vt_symbol,
        strategy_class=strategy_class,
        interval=interval,
        target_name=target_name,
        test_budget=int(test_budget),
        created_at=moment,
        updated_at=moment,
    )

    existing = load_record(path)
    if existing is None or existing.identity() != fresh.identity():
        return fresh

    return replace(
        existing,
        target_name=target_name,
        test_budget=int(test_budget),
        updated_at=moment,
    )


def record_peek(
    record: SplitRecord, view: SegmentPeek, now: datetime | None = None
) -> SplitRecord:
    """记一次测试段查看。返回新记录（`SplitRecord` 是 frozen 的）。"""
    return replace(
        record,
        peeks=(*record.peeks, view),
        updated_at=now or datetime.now(),
    )


def reset_test_budget(
    record: SplitRecord, reason: str, now: datetime | None = None
) -> SplitRecord:
    """重开预算 —— 必须给理由，理由落盘。

    与 `SegmentedRunner.reset_test_budget` 同一口径：这不是"绕过"，是留痕。
    重开测试集是一个研究决策，它必须在账本上留下一行。
    """
    text = str(reason).strip()
    if not text:
        raise ValueError(
            "reset_test_budget 必须给非空 reason —— 重开测试集不允许无声发生"
        )
    moment = now or datetime.now()
    trail = (
        f"{moment:%Y-%m-%d %H:%M} 在用掉 {len(record.peeks)} 次后重开：{text}"
    )
    return replace(
        record,
        peeks=(),
        resets=(*record.resets, trail),
        updated_at=moment,
    )
