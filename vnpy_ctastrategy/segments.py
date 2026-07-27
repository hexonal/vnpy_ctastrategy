"""样本内外三段切分（TRAIN / VALID / TEST）—— 把投研纪律写进函数签名。

本模块回答的问题与同目录另外三个模块正交：

    robust_metrics  ——  这条曲线【形状】好不好？（样本内描述统计）
    overfitting     ——  这条曲线是不是【挑出来的运气】？（选择过程的检验）
    deflated_sharpe ——  多重比较校正后还剩多少？（最大值序统计）
    segments        ——  这个数字【是在哪一段上算出来的】？（样本内 / 样本外）

前三者都不知道自己算的是哪一段：样本内的 `sharpe_significant=True`
与样本外的长得一模一样。本模块补的就是这个缺口。

━━━ 三段的口径（对齐 vnpy.alpha 与官方第 4 篇原文）━━━

    Segment.TRAIN   训练段 —— 学参数 / 扫网格
    Segment.VALID   验证段 —— 调参、早停、模型选择
    Segment.TEST    测试段 —— 只用于最后观察效果

**VALID 归样本内。** 官方原文：

    "验证集不是完全意义上的样本外。虽然模型参数不一定直接在验证集上训练，
     但只要我们根据验证集表现调整因子、调参数、选择训练轮数或更换模型，
     它就已经参与了研究决策。"

因此 `is_out_of_sample()` 只对 TEST 返回 True。把 VALID 的统计量
当样本外证据发布，是本模块存在的头号防范对象。

━━━ 为什么是新文件而不是改引擎 ━━━

`BacktestingEngine.set_parameters` 只有单段 start/end，`run_backtesting`
是单趟回放，引擎内部没有任何按区间分段的插入点。切分只可能发生在
"喂什么起止"这一层 —— 这正是 `overfitting.EngineRunner` 已经在做的事
（每个窗口新建一个引擎，调用未经修改的 `set_parameters`）。

本模块沿用同一形态：**`backtesting.py` 零修改**，三段能力全部长在
`SegmentedRunner` 这一层。合并上游时冲突面为零。

━━━ 强制显式性 ━━━

抄官方 `AlphaModel.predict(dataset, segment)` 的形状：凡是"取某一段结果"
的 API，`segment` 一律是**无默认值的必填参数**。写代码的人必须每次都
把"我现在看的是哪一段"打出来，纪律因此变成签名而不是注释。

━━━ 两道闸 ━━━

1. `SegmentedRunner.scan(settings, segment)` 在 segment 为 TEST 时
   直接抛 `SegmentLeakError` —— 在测试段上扫参数是被代码挡住的动作。
2. `SegmentedRunner.run(setting, Segment.TEST)` 有**次数预算**（默认 1）。
   超预算抛 `SegmentBudgetExhaustedError`，并把每一次查看记进
   `test_audit` 留痕。确需重开必须显式调 `reset_test_budget(reason=...)`，
   reason 为空会被拒绝 —— 重开测试集这件事无法悄悄发生。

━━━ 诚实边界 ━━━

* 一次性 holdout **不能**替代 Walk-Forward。它只有一段样本外、一个
  参数集，样本外样本量小、且无法回答"参数是否随时间漂移"。
  它便宜（1 × 网格 + 2 次回测）而 WF 昂贵（折数 × 网格），
  便宜的用途是"别因为太贵就干脆不看样本外"，不是"WF 的替代品"。
* 三段切分本身不产生 alpha。它只保证"这个数字是样本外的"这句话为真，
  不保证这个数字好看，也不保证它在未来能复现。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

import numpy as np
from pandas import DataFrame

from .backtesting import BacktestingEngine
from .overfitting import (
    BacktestRunner,
    EngineRunner,
    Selector,
    SignificanceResult,
    argmax_selector,
    assess_significance,
    daily_log_returns,
)

__all__ = [
    "HoldoutReport",
    "Segment",
    "SegmentBudgetExhaustedError",
    "SegmentGuardedEngine",
    "SegmentLeakError",
    "SegmentResult",
    "SegmentedRunner",
    "ThreeWaySplit",
    "assert_segment_parity",
    "in_sample_segments",
    "is_out_of_sample",
    "make_three_way_split",
    "run_holdout",
    "split_by_ratio",
    "valid_contradicts_train",
]


def _naive(moment: datetime) -> datetime:
    """去掉时区信息。

    数据库返回 tz-aware 时间戳，`set_parameters` 常收到裸 datetime，
    两者不能直接比较（与 `overfitting._naive` 同一处理，日线窗口比较
    不依赖时区）。
    """
    return moment.replace(tzinfo=None) if moment.tzinfo is not None else moment


# ══════════════════════════════════════════════════════════════════════
# 1. 段的定义
# ══════════════════════════════════════════════════════════════════════

class Segment(Enum):
    """样本段。

    **取值与 `vnpy.alpha.dataset.utility.Segment` 逐项对齐，但刻意不 import 它。**

    理由：`vnpy.alpha` 是 vnpy 的 optional extra，import 那个枚举会连带
    拉入 polars / alphalens / pandas / tqdm，把可选依赖变成 vnpy_ctastrategy
    的事实硬依赖（本包只声明 vnpy / pandas / plotly 三项）。

    对齐由 `assert_segment_parity()` 在测试里钉住：上游若改了枚举值，
    测试会红，而不是两套语义静默错配。
    """

    TRAIN = 1
    VALID = 2
    TEST = 3


class SegmentLeakError(RuntimeError):
    """在测试段上做了只允许在样本内做的事（典型：扫参数网格）。"""


class SegmentBudgetExhaustedError(RuntimeError):
    """测试段的查看次数预算已用尽。"""


def in_sample_segments() -> tuple[Segment, ...]:
    """样本内的段。VALID 在内 —— 它参与研究决策（官方第 4 篇口径）。"""
    return (Segment.TRAIN, Segment.VALID)


def is_out_of_sample(segment: Segment) -> bool:
    """这一段是不是样本外。只有 TEST 是。"""
    if not isinstance(segment, Segment):
        raise TypeError(f"segment 必须是 Segment，收到 {type(segment).__name__}")
    return segment is Segment.TEST


def assert_segment_parity() -> bool:
    """断言本地 Segment 与 `vnpy.alpha` 的枚举逐项一致。

    返回 True = 已对比且一致；返回 False = `vnpy.alpha` 不可导入
    （未装 optional extra），本次未对比。不一致则抛 AssertionError。

    惰性 import：只在调用时才碰 vnpy.alpha，模块导入期零开销。
    """
    try:
        from vnpy.alpha.dataset.utility import Segment as AlphaSegment
    except ImportError:
        return False

    ours: dict[str, int] = {m.name: int(m.value) for m in Segment}
    theirs: dict[str, int] = {m.name: int(m.value) for m in AlphaSegment}
    if ours != theirs:
        raise AssertionError(
            f"Segment 与 vnpy.alpha 不一致：本地 {ours} vs 上游 {theirs}。"
            f"上游改了枚举 —— 必须同步本模块，否则跨栈语义静默错配。"
        )
    return True


# ══════════════════════════════════════════════════════════════════════
# 2. 切分
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ThreeWaySplit:
    """三段的窗口边界。

    边界取自【真实 K 线时间戳】而不是自然日，所以 *_bars 是交易日数
    （与 `overfitting.WalkForwardSplit` 同一约定）。

    不变式（`__post_init__` 强制，违反即抛 ValueError）：

        train_start ≤ train_end < valid_start ≤ valid_end < test_start ≤ test_end

    两个 `<` 是严格的：三段**不共享任何一根 K 线**。
    """

    train_start: datetime
    train_end: datetime
    valid_start: datetime
    valid_end: datetime
    test_start: datetime
    test_end: datetime
    train_bars: int
    valid_bars: int
    test_bars: int

    def __post_init__(self) -> None:
        if self.train_start > self.train_end:
            raise ValueError("train_start 晚于 train_end")
        if self.valid_start > self.valid_end:
            raise ValueError("valid_start 晚于 valid_end")
        if self.test_start > self.test_end:
            raise ValueError("test_start 晚于 test_end")
        if self.train_end >= self.valid_start:
            raise ValueError(
                f"训练段与验证段重叠：train_end={self.train_end} "
                f"≥ valid_start={self.valid_start}"
            )
        if self.valid_end >= self.test_start:
            raise ValueError(
                f"验证段与测试段重叠：valid_end={self.valid_end} "
                f"≥ test_start={self.test_start}"
            )
        for name in ("train_bars", "valid_bars", "test_bars"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} 必须 ≥1")

    def as_period(self, segment: Segment) -> tuple[datetime, datetime]:
        """某一段的 (起, 止)。**segment 无默认值 —— 必须显式写出来。**"""
        if not isinstance(segment, Segment):
            raise TypeError(f"segment 必须是 Segment，收到 {type(segment).__name__}")
        if segment is Segment.TRAIN:
            return self.train_start, self.train_end
        if segment is Segment.VALID:
            return self.valid_start, self.valid_end
        return self.test_start, self.test_end

    def bars(self, segment: Segment) -> int:
        """某一段的 K 线根数。segment 无默认值。"""
        if not isinstance(segment, Segment):
            raise TypeError(f"segment 必须是 Segment，收到 {type(segment).__name__}")
        if segment is Segment.TRAIN:
            return int(self.train_bars)
        if segment is Segment.VALID:
            return int(self.valid_bars)
        return int(self.test_bars)

    def in_sample_period(self) -> tuple[datetime, datetime]:
        """样本内整段 (TRAIN 起, VALID 止) —— 官方口径下可用于拟合的全部数据。"""
        return self.train_start, self.valid_end

    def as_dict(self) -> dict[str, Any]:
        return {
            "train_start": self.train_start,
            "train_end": self.train_end,
            "valid_start": self.valid_start,
            "valid_end": self.valid_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "train_bars": int(self.train_bars),
            "valid_bars": int(self.valid_bars),
            "test_bars": int(self.test_bars),
        }


def _checked_datetimes(bar_datetimes: Sequence[datetime]) -> list[datetime]:
    """校验 K 线时间戳序列：非空、严格递增（不允许重复）。"""
    dts = list(bar_datetimes)
    if not dts:
        raise ValueError("bar_datetimes 为空 —— 没有 K 线就无法按交易日切分")
    for i in range(1, len(dts)):
        if dts[i] <= dts[i - 1]:
            raise ValueError(
                f"bar_datetimes 必须严格递增：下标 {i - 1}/{i} 处 "
                f"{dts[i - 1]} → {dts[i]}"
            )
    return dts


def make_three_way_split(
    bar_datetimes: Sequence[datetime],
    train_bars: int,
    valid_bars: int,
    test_bars: int,
    anchor: Literal["start", "end"] = "end",
) -> ThreeWaySplit:
    """按真实交易日切出 TRAIN / VALID / TEST 三段（时序相邻、互不重叠）。

    anchor="end"（默认）：三段贴着样本【末尾】排，多出来的老 K 线被丢弃。
        TEST 因此总是最近的一段 —— "把最近这段锁起来最后看一眼"。
    anchor="start"：三段贴着样本【开头】排，多出来的新 K 线被丢弃。
        用于复现历史上某次研究的切分。

    被丢弃的 K 线不会悄悄混进任何一段：三段的并集就是被使用的全部样本。
    """
    dts = _checked_datetimes(bar_datetimes)
    n = len(dts)

    if train_bars < 2:
        raise ValueError(f"train_bars 必须 ≥2，收到 {train_bars}")
    if valid_bars < 1:
        raise ValueError(f"valid_bars 必须 ≥1，收到 {valid_bars}")
    if test_bars < 1:
        raise ValueError(f"test_bars 必须 ≥1，收到 {test_bars}")

    total = int(train_bars) + int(valid_bars) + int(test_bars)
    if total > n:
        raise ValueError(
            f"三段合计 {total} 根 > 样本 {n} 根：样本长度不足以切出三段"
        )
    if anchor not in ("start", "end"):
        raise ValueError(f"anchor 只能是 'start' 或 'end'，收到 {anchor!r}")

    offset = 0 if anchor == "start" else n - total
    a = offset
    b = a + int(train_bars)
    c = b + int(valid_bars)
    d = c + int(test_bars)

    return ThreeWaySplit(
        train_start=dts[a], train_end=dts[b - 1],
        valid_start=dts[b], valid_end=dts[c - 1],
        test_start=dts[c], test_end=dts[d - 1],
        train_bars=int(train_bars), valid_bars=int(valid_bars),
        test_bars=int(test_bars),
    )


def split_by_ratio(
    bar_datetimes: Sequence[datetime],
    train: float = 0.6,
    valid: float = 0.2,
    test: float = 0.2,
) -> ThreeWaySplit:
    """按比例切三段，**用满全部样本**（余数归 TEST）。

    三个比例必须为正且和为 1（容差 1e-9）。切出的根数满足
    train_bars + valid_bars + test_bars == len(bar_datetimes)。
    """
    dts = _checked_datetimes(bar_datetimes)
    n = len(dts)

    for name, value in (("train", train), ("valid", valid), ("test", test)):
        if not (value > 0.0):
            raise ValueError(f"{name} 比例必须 >0，收到 {value}")
    if abs(train + valid + test - 1.0) > 1e-9:
        raise ValueError(
            f"三段比例之和必须为 1，收到 {train + valid + test!r}"
        )

    train_bars = int(n * train)
    valid_bars = int(n * valid)
    test_bars = n - train_bars - valid_bars
    if train_bars < 2 or valid_bars < 1 or test_bars < 1:
        raise ValueError(
            f"样本 {n} 根按 {train}/{valid}/{test} 切出 "
            f"{train_bars}/{valid_bars}/{test_bars} 根 —— 至少有一段不足"
        )

    return make_three_way_split(
        dts, train_bars, valid_bars, test_bars, anchor="start"
    )


# ══════════════════════════════════════════════════════════════════════
# 3. 分段执行器
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SegmentResult:
    """某一段上、某一组参数的回测结果。"""

    segment: Segment
    setting: dict
    start: datetime
    end: datetime
    daily_df: DataFrame
    statistics: dict

    @property
    def out_of_sample(self) -> bool:
        return is_out_of_sample(self.segment)

    def target(self, target_name: str) -> float:
        """取 statistics 面板里的某个目标值，缺失 / 非数返回 nan。"""
        value = self.statistics.get(target_name)
        try:
            result = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return float("nan")
        return result

    def trade_count(self) -> float:
        """该段成交笔数合计。daily_df 没有该列时返回 nan（不是 0）。"""
        df = self.daily_df
        if df is None or df.empty or "trade_count" not in df.columns:
            return float("nan")
        return float(np.nansum(df["trade_count"].to_numpy(dtype=float)))


class SegmentedRunner:
    """把 `(setting, start, end) -> daily_df` 的执行器包成"按段取数"的执行器。

    底层 runner 原样复用（生产用 `overfitting.EngineRunner`），
    因此三段结果与普通回测走**同一套 statistics 口径**，
    含本 fork 加的 RAR / R³ / Robust Sharpe。

    两处纪律长在这里：

      * `scan(settings, segment)` 在 TEST 段直接抛 `SegmentLeakError`；
      * `run(setting, Segment.TEST)` 消耗次数预算，超额抛
        `SegmentBudgetExhaustedError`，每次查看都进 `test_audit`。

    注入 statistics：每段的 statistics 字典会被追加两个键 ——
    `segment`（段名）与 `is_out_of_sample`（布尔）。下游把面板写进
    预测卡时，读到 `is_out_of_sample=False` 必须标注"样本内描述统计"。
    """

    def __init__(
        self,
        runner: BacktestRunner,
        split: ThreeWaySplit,
        statistics_func: Callable[[DataFrame], dict] | None = None,
        capital: float | None = None,
        test_budget: int = 1,
    ) -> None:
        if int(test_budget) < 1:
            raise ValueError(f"test_budget 必须 ≥1，收到 {test_budget}")

        if statistics_func is None:
            if not isinstance(runner, EngineRunner):
                raise ValueError(
                    "runner 不是 EngineRunner 时必须显式传入 statistics_func"
                )
            statistics_func = runner.statistics

        if capital is None:
            capital = (
                float(runner.capital) if isinstance(runner, EngineRunner) else 1.0
            )

        self.runner: BacktestRunner = runner
        self.split: ThreeWaySplit = split
        self.statistics_func: Callable[[DataFrame], dict] = statistics_func
        self.capital: float = float(capital)
        self.test_budget: int = int(test_budget)
        self.test_audit: list[str] = []
        self._test_calls: int = 0

    @property
    def test_calls(self) -> int:
        """测试段已被查看的次数（含被预算拒绝的那次尝试）。"""
        return self._test_calls

    def reset_test_budget(self, reason: str) -> None:
        """重开测试段预算 —— 必须给理由，理由进审计日志。

        这不是"绕过"，是**留痕**：重开测试集是一个研究决策，
        它必须在代码里显式出现，并且在 `test_audit` 里留下一行。
        """
        text = str(reason).strip()
        if not text:
            raise ValueError(
                "reset_test_budget 必须给非空 reason —— 重开测试集不允许无声发生"
            )
        self.test_audit.append(f"RESET after {self._test_calls} 次查看：{text}")
        self._test_calls = 0

    def run(self, setting: dict, segment: Segment) -> SegmentResult:
        """在指定段上跑一组参数。**segment 无默认值 —— 必须显式写出来。**"""
        if not isinstance(segment, Segment):
            raise TypeError(f"segment 必须是 Segment，收到 {type(segment).__name__}")

        if segment is Segment.TEST:
            self._consume_test_budget(setting)

        start, end = self.split.as_period(segment)
        df = self.runner(dict(setting), start, end)
        if df is None:
            df = DataFrame()
        df.attrs["segment"] = segment.name

        stats = dict(self.statistics_func(df))
        stats["segment"] = segment.name
        stats["is_out_of_sample"] = is_out_of_sample(segment)

        return SegmentResult(
            segment=segment, setting=dict(setting), start=start, end=end,
            daily_df=df, statistics=stats,
        )

    def scan(self, settings: Sequence[dict], segment: Segment) -> list[SegmentResult]:
        """在指定段上扫一整个参数网格。**TEST 段被硬闸挡住。**"""
        if not isinstance(segment, Segment):
            raise TypeError(f"segment 必须是 Segment，收到 {type(segment).__name__}")
        if segment is Segment.TEST:
            raise SegmentLeakError(
                "禁止在 TEST 段上扫参数网格：扫参数 = 用该段做模型选择，"
                "这一步做完测试段就变成样本内，之后任何【样本外】表述都是假的。"
                "扫参数只允许在 TRAIN / VALID 上做；"
                "TEST 只能用 run(setting, Segment.TEST) 对【已选定的单组参数】"
                "看一次。"
            )
        if not settings:
            raise ValueError("settings 为空")
        return [self.run(setting, segment) for setting in settings]

    def _consume_test_budget(self, setting: dict) -> None:
        self._test_calls += 1
        self.test_audit.append(f"#{self._test_calls} setting={setting!r}")
        if self._test_calls > self.test_budget:
            trail = "\n  ".join(self.test_audit)
            raise SegmentBudgetExhaustedError(
                f"TEST 段预算 {self.test_budget} 次已用尽，本次是第 "
                f"{self._test_calls} 次。查看记录：\n  {trail}\n"
                f"反复在测试段上取数 = 把测试集研究成样本内"
                f"（官方第 4 篇：'测试集也会逐渐变成被研究过的数据'）。"
                f"确需重开必须显式调用 reset_test_budget(reason=...)。"
            )


# ══════════════════════════════════════════════════════════════════════
# 4. 一次性 holdout
# ══════════════════════════════════════════════════════════════════════

def valid_contradicts_train(
    train_target: float, valid_target: float, target_name: str
) -> str | None:
    """选中参数在 VALID 段翻负、TRAIN 段却为正时的警告文本；不满足返回 None。

    这是 VALID 段唯一的产出：**它不改选择**（改了它就变成第二个选参段），
    它只回答"选中的参数换一段样本内数据还站不站得住"。站不住时，测试段
    结果无论好看难看都已经失去解释力 —— 因为样本内两段就已经互相打脸。

    单独抽成函数是因为它有两个调用点（`run_holdout` 与 `segment_cli` 的
    select 档）。判据与措辞只能有一份：两份迟早会漂成两套口径。
    """
    if not (np.isfinite(valid_target) and np.isfinite(train_target)):
        return None
    if not valid_target <= 0.0 < train_target:
        return None
    return (
        f"选中参数在 VALID 段的 {target_name}={valid_target:.4f} ≤ 0，"
        f"而 TRAIN 段为 {train_target:.4f}：样本内两段已经不一致，"
        f"TEST 段结果无论好坏都不足以支撑结论"
    )



@dataclass
class HoldoutReport:
    """一次三段 holdout 的全部结果。

    `statistics(segment)` / `daily_df(segment)` / `result(segment)`
    的 segment 都**没有默认值** —— 取数时必须写明是哪一段。
    """

    target_name: str
    chosen_setting: dict
    chosen_index: int
    n_candidates: int
    train_scores: list[float]
    train_results: list[SegmentResult]
    results: dict[Segment, SegmentResult]
    test_significance: SignificanceResult | None = None
    warnings: list[str] = field(default_factory=list)

    def result(self, segment: Segment) -> SegmentResult:
        """某一段上被选中参数的完整结果。segment 无默认值。"""
        if not isinstance(segment, Segment):
            raise TypeError(f"segment 必须是 Segment，收到 {type(segment).__name__}")
        if segment not in self.results:
            raise KeyError(f"报告里没有 {segment.name} 段的结果")
        return self.results[segment]

    def statistics(self, segment: Segment) -> dict:
        """某一段的 statistics 面板。segment 无默认值。

        面板里带 `segment` 与 `is_out_of_sample` 两个键，
        因此"这个数字是样本内还是样本外"随字典一起流到下游。
        """
        return self.result(segment).statistics

    def daily_df(self, segment: Segment) -> DataFrame:
        """某一段的逐日结果表。segment 无默认值。"""
        return self.result(segment).daily_df

    def target(self, segment: Segment) -> float:
        """某一段上被选中参数的目标值。segment 无默认值。"""
        return self.result(segment).target(self.target_name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_name": self.target_name,
            "chosen_setting": dict(self.chosen_setting),
            "chosen_index": int(self.chosen_index),
            "n_candidates": int(self.n_candidates),
            "targets": {
                seg.name: self.target(seg) for seg in self.results
            },
            "test_significance": (
                self.test_significance.as_dict()
                if self.test_significance is not None else None
            ),
            "warnings": list(self.warnings),
        }

    def summary(self) -> str:
        """人读的一段文本。不含裁决 —— 裁决要看显著性与警告一起判。"""
        lines: list[str] = [
            f"三段 holdout（目标 {self.target_name}）",
            f"候选参数 {self.n_candidates} 组，选中第 {self.chosen_index} 组："
            f"{self.chosen_setting}",
        ]
        for seg in (Segment.TRAIN, Segment.VALID, Segment.TEST):
            if seg not in self.results:
                continue
            res = self.results[seg]
            mark = "样本外" if res.out_of_sample else "样本内"
            lines.append(
                f"  {seg.name:<5} [{mark}] {self.target_name}="
                f"{res.target(self.target_name):.4f} "
                f"成交 {res.trade_count():.0f} 笔 "
                f"({res.start:%Y-%m-%d} ~ {res.end:%Y-%m-%d})"
            )
        if self.test_significance is not None:
            sig = self.test_significance
            lines.append(
                f"  TEST 显著性：Sharpe={sig.sharpe:.3f} "
                f"t={sig.t_stat:.2f} p_block={sig.p_block_bootstrap:.4f} "
                f"显著={sig.significant}"
            )
        for note in self.warnings:
            lines.append(f"  ⚠ {note}")
        return "\n".join(lines)


def run_holdout(
    runner: SegmentedRunner,
    settings: Sequence[dict],
    target_name: str = "sharpe_ratio",
    selector: Selector = argmax_selector,
    significance: bool = True,
    annual_days: int = 252,
    alpha: float = 0.05,
    n_bootstrap: int = 4000,
    block_size: int | None = None,
    seed: int = 20260726,
    min_test_bars: int = 20,
) -> HoldoutReport:
    """跑一次三段 holdout：TRAIN 选参 → VALID 复核 → TEST 只看一次。

    流程与官方第 4 篇的样本内外边界逐条对应：

      1. **TRAIN**：跑完整参数网格，按 `target_name` 用 `selector` 选参。
      2. **VALID**：只把【已选中的那一组】拿过来复算。这一段属样本内，
         它的作用是"选中的参数在另一段样本内数据上还站不站得住"，
         **不回头改选择**（回头改 = 把 VALID 也做成选参段，
         那就该改用 `selector_on=VALID` 的显式流程而不是偷偷改）。
      3. **TEST**：同一组参数只跑一次，由 `SegmentedRunner` 的预算钉住。

    整个流程中，TEST 段的任何数字都不参与第 1、2 步 —— 这由
    `test_holdout_selection_ignores_test_segment` 构造性钉住
    （造一个"TRAIN 最差但 TEST 最好"的参数，它不许被选中）。

    significance=True 时对 TEST 段的逐日收益跑 block-bootstrap 显著性检验
    （口径与 `overfitting.assess_significance` 一致）。样本 <10 日时该函数
    自己返回 nan，不会给假的 p 值。
    """
    if not isinstance(runner, SegmentedRunner):
        raise TypeError(
            f"runner 必须是 SegmentedRunner，收到 {type(runner).__name__}"
        )
    if not settings:
        raise ValueError("settings 为空")

    grid: list[dict] = [dict(s) for s in settings]
    warnings: list[str] = []

    # ── 1. TRAIN：扫网格选参 ────────────────────────────────────────
    train_results = runner.scan(grid, Segment.TRAIN)
    scores = np.array(
        [res.target(target_name) for res in train_results], dtype=float
    )
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    if not np.isfinite(scores).any():
        raise ValueError(
            f"TRAIN 段所有参数的 {target_name} 都无效 —— 无法选参"
        )

    chosen = int(selector(grid, scores))
    if not 0 <= chosen < len(grid):
        raise ValueError(f"selector 返回越界下标 {chosen}（网格 {len(grid)} 组）")

    # ── 2. VALID：复核选中的那一组（样本内）────────────────────────
    valid_result = runner.run(grid[chosen], Segment.VALID)

    # ── 3. TEST：只看一次 ──────────────────────────────────────────
    test_result = runner.run(grid[chosen], Segment.TEST)

    results: dict[Segment, SegmentResult] = {
        Segment.TRAIN: train_results[chosen],
        Segment.VALID: valid_result,
        Segment.TEST: test_result,
    }

    # ── 4. 诚实警告 ────────────────────────────────────────────────
    if runner.split.test_bars < min_test_bars:
        warnings.append(
            f"TEST 段仅 {runner.split.test_bars} 根 K 线"
            f"（阈值 {min_test_bars}）：样本外统计量的标准误极大，"
            f"任何结论都不可裁决"
        )

    test_trades = test_result.trade_count()
    if np.isfinite(test_trades) and test_trades == 0.0:
        warnings.append(
            "TEST 段零成交：先排查指标预热"
            "（EngineRunner 的 warmup_bars / on_init 的 load_bar），"
            "零成交不等于'样本外持平'"
        )

    contradiction = valid_contradicts_train(
        train_results[chosen].target(target_name),
        valid_result.target(target_name),
        target_name,
    )
    if contradiction is not None:
        warnings.append(contradiction)

    # ── 5. TEST 段显著性 ───────────────────────────────────────────
    test_sig: SignificanceResult | None = None
    if significance:
        df = test_result.daily_df
        if df is not None and not df.empty and "net_pnl" in df.columns:
            returns = daily_log_returns(
                df["net_pnl"].to_numpy(dtype=float), runner.capital
            )
            test_sig = assess_significance(
                returns, annual_days=annual_days, alpha=alpha,
                n_bootstrap=n_bootstrap, block_size=block_size, seed=seed,
            )
        else:
            warnings.append("TEST 段无逐日结果，显著性检验未计算")

    return HoldoutReport(
        target_name=target_name,
        chosen_setting=dict(grid[chosen]),
        chosen_index=chosen,
        n_candidates=len(grid),
        train_scores=[float(v) for v in scores],
        train_results=train_results,
        results=results,
        test_significance=test_sig,
        warnings=warnings,
    )


# ══════════════════════════════════════════════════════════════════════
# 5. 守卫版引擎 —— 挡住"直接拿引擎在 TEST 段上扫参数"
# ══════════════════════════════════════════════════════════════════════

class SegmentGuardedEngine(BacktestingEngine):
    """绑定了三段切分的回测引擎：寻优窗口碰到 TEST 段就拒绝执行。

    `SegmentedRunner.scan` 只管得住走本模块的那条路。研究者完全可以绕开它，
    直接 `engine.set_parameters(start=测试段起, end=测试段止)` 然后
    `run_bf_optimization` —— 那就是在测试段上扫参数，而原引擎毫无察觉。
    本类补的就是这条旁路。

    **不改 `backtesting.py`**：只是它的子类，覆写两个寻优入口 + 一个
    按段设窗口的便利方法。合并上游时冲突面为零。

    典型用法::

        engine = SegmentGuardedEngine()
        engine.bind_split(split)
        engine.use_segment(
            Segment.TRAIN, vt_symbol="700.SEHK", interval=Interval.DAILY,
            rate=0.0013, slippage=0.0, size=1, pricetick=0.01, capital=1_000_000,
        )
        engine.add_strategy(MyStrategy, {})
        engine.load_data()
        results = engine.run_bf_optimization(optimization_setting)   # 放行

        engine.use_segment(Segment.TEST, ...)
        engine.run_bf_optimization(optimization_setting)             # SegmentLeakError

    未绑定切分时**不阻塞**（引擎有大量与三段无关的既有用法），
    但每次寻优都会往 `self.output` 打一条告警并记进 `segment_warnings` ——
    "不知道自己在哪一段"这件事必须留痕，不能静默通过。
    """

    def __init__(self) -> None:
        super().__init__()
        self.split: ThreeWaySplit | None = None
        self.segment: Segment | None = None
        self.segment_warnings: list[str] = []

    def bind_split(self, split: ThreeWaySplit) -> None:
        """绑定三段切分。绑定后寻优窗口才可被校验。"""
        if not isinstance(split, ThreeWaySplit):
            raise TypeError(
                f"split 必须是 ThreeWaySplit，收到 {type(split).__name__}"
            )
        self.split = split

    def require_split(self) -> ThreeWaySplit:
        """取已绑定的切分，没绑就抛 —— 不给"悄悄按全样本跑"的余地。"""
        if self.split is None:
            raise ValueError(
                "尚未 bind_split(...)：按段设窗口前必须先绑定 ThreeWaySplit"
            )
        return self.split

    def use_segment(self, segment: Segment, **kwargs: Any) -> None:
        """按段设置回测窗口。**segment 无默认值。**

        start / end 由切分决定，其余参数原样转发 `set_parameters`。
        调用方自己传 start / end 会被拒绝 —— "换了段却忘了改日期"
        （两份手抄字面量各改各的）正是本方法要消灭的错误。
        """
        if not isinstance(segment, Segment):
            raise TypeError(f"segment 必须是 Segment，收到 {type(segment).__name__}")
        for banned in ("start", "end"):
            if banned in kwargs:
                raise ValueError(
                    f"use_segment 不接受 {banned}：窗口由 {segment.name} 段决定，"
                    f"手抄日期正是这一层要消灭的错误"
                )
        start, end = self.require_split().as_period(segment)
        self.set_parameters(start=start, end=end, **kwargs)
        self.segment = segment

    def current_segment(self) -> Segment | None:
        """当前窗口正好等于哪一段（按日期比对）；都不匹配返回 None。"""
        if self.split is None:
            return None
        start = getattr(self, "start", None)
        end = getattr(self, "end", None)
        if start is None or end is None:
            return None
        for segment in Segment:
            lo, hi = self.split.as_period(segment)
            if (
                _naive(start).date() == _naive(lo).date()
                and _naive(end).date() == _naive(hi).date()
            ):
                return segment
        return None

    def run_bf_optimization(self, *args: Any, **kwargs: Any) -> Any:
        """穷举寻优 —— 窗口碰到 TEST 段直接拒绝，其余原样转发父类。

        **刻意用 `*args/**kwargs` 而不是抄一份父类签名**：父类的寻优入口
        还在长参数（gates / gate_config / collect_returns 等）。抄签名等于
        把新参数挡在门外，而且是静默挡 —— 调用方传了却不生效。
        透传则永远与父类同步，这一层只负责"拦不拦"，不负责"怎么跑"。
        """
        self.guard_optimization("run_bf_optimization")
        return super().run_bf_optimization(*args, **kwargs)

    run_optimization = run_bf_optimization

    def run_ga_optimization(self, *args: Any, **kwargs: Any) -> Any:
        """遗传算法寻优 —— 窗口碰到 TEST 段直接拒绝，其余原样转发父类。"""
        self.guard_optimization("run_ga_optimization")
        return super().run_ga_optimization(*args, **kwargs)

    def guard_optimization(self, method: str) -> None:
        """寻优前的三段守卫。

        未绑定切分 → 告警留痕后放行（引擎还有大量与三段无关的既有用法）。
        窗口与 TEST 段有任何交集 → 抛 `SegmentLeakError`。
        注意"全样本窗口"也算有交集：在全样本上扫参数同样把测试段用掉了。
        """
        if self.split is None:
            note = (
                f"{method}：未绑定 ThreeWaySplit，无法判断寻优窗口是否碰到 "
                f"TEST 段 —— 本次寻优的样本内外性质不可自证。"
                f"先 bind_split(...) + use_segment(...) 才有守卫。"
            )
            self.segment_warnings.append(note)
            self.output(note)
            return

        start = getattr(self, "start", None)
        end = getattr(self, "end", None)
        if start is None or end is None:
            note = (
                f"{method}：尚未 set_parameters，无法校验窗口 —— 守卫未生效"
            )
            self.segment_warnings.append(note)
            self.output(note)
            return

        lo = _naive(start)
        hi = _naive(end)
        t0 = _naive(self.split.test_start).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        t1 = _naive(self.split.test_end).replace(
            hour=23, minute=59, second=59, microsecond=999999
        )
        if hi < t0 or lo > t1:
            return

        raise SegmentLeakError(
            f"{method} 的窗口 [{lo:%Y-%m-%d} ~ {hi:%Y-%m-%d}] 与 TEST 段 "
            f"[{t0:%Y-%m-%d} ~ {t1:%Y-%m-%d}] 有交集：扫参数 = 用该段做模型选择，"
            f"做完测试段就变成样本内。"
            f"寻优只允许在 TRAIN / VALID 上跑 —— 用 "
            f"use_segment(Segment.TRAIN, ...) 或 use_segment(Segment.VALID, ...) "
            f"重设窗口；TEST 段只能对【已选定的单组参数】跑一次普通回测。"
        )
