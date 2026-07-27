"""三段样本内外回测的命令行入口 —— 这个流程唯一的执行口。

    python -m vnpy_ctastrategy.segment_cli --help

`segments.py` 把纪律写进了函数签名，但一个没有调用口的模块等于不存在：
在本文件之前，`SegmentedRunner` 的全部调用点都在它自己的测试里。回测器
GUI（`vnpy_ctabacktester`）也没有三段的概念 —— 它只有一组 start/end。

**为什么入口在命令行而不在 GUI**（判断与理由的完整版见
`vnpy_app/fluent_ui/backtester_segments.py` 的模块文档，那边还负责在面板上
把这条命令印出来）：

  1. 测试段预算是**计数器**，而 GUI 是长驻进程 + 反复点按钮的交互形态。
     按钮点第二下要么报错（用户读作"这按钮坏了"）要么静默重置（闸等于没有）。
     命令行"一次调用 = 一次流程"与预算的形状天然对齐，跨进程那一段由
     `segment_record` 的账本接上。
  2. 三段流程的产出是一份**带审计轨迹的报告**（整个网格的分数、警告、
     显著性、看过几次测试段），不是一张 key→value 的统计面板。硬塞进
     `StatisticsMonitor` 只能显示 TEST 那一列 —— 而"把样本外数字单独拎出来
     当结论"恰恰是这套东西要防的事。
  3. 面板上那两个按钮本身就是扫参数的入口，而 TEST 段禁止扫参数。GUI 真正
     缺的不是三个日期框，是**一道拦截** —— 那一道装在 vnpy_app 侧。

━━━ 两档：select 与 final ━━━

    --stage select   （默认）TRAIN 扫网格选参 → VALID 复核。**碰都不碰 TEST。**
    --stage final    上面两步之后，对选中的那一组在 TEST 段上跑一次。

默认档刻意是 select：随手敲一条命令不该烧掉测试集。看测试段是一个需要
显式说出口的决定，而且它会被记进账本（`~/.vntrader/cta_holdout_split.json`）。
额度用尽后再跑 final 会被拒，除非给出 `--reset-test-budget "理由"` ——
理由同样落盘。

━━━ 一条诚实边界 ━━━

一次性 holdout 不是 Walk-Forward 的替代品：它只有一段样本外、一个参数集。
`overfitting_audit.py` 那条路（WF + CSCV/PBO）贵得多，但能回答"参数是否随
时间漂移"。本入口便宜（1×网格 + 2 次回测），便宜的用途是"别因为太贵就干脆
不看样本外"。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .overfitting import Selector, argmax_selector
from .segment_record import (
    SegmentBudgetRefused,
    SegmentPeek,
    SplitRecord,
    open_record,
    record_path,
    record_peek,
    reset_test_budget,
    save_record,
)
from .segments import (
    HoldoutReport,
    Segment,
    SegmentedRunner,
    SegmentResult,
    ThreeWaySplit,
    make_three_way_split,
    run_holdout,
    split_by_ratio,
    valid_contradicts_train,
)

__all__ = [
    "FINAL",
    "SELECT",
    "StageOutcome",
    "authorize_final",
    "build_parser",
    "main",
    "plan_split",
    "run_stage",
]


SELECT = "select"
FINAL = "final"

#: 默认参数网格 —— 与 `overfitting_audit.main()` 用的是同一套海龟参数，
#: 两条路（一次性 holdout / Walk-Forward + PBO）的结论才可以并排读。
DEFAULT_GRID = ("entry_window=10:55:5", "exit_window=5:20:5", "atr_stop=1.5:3.0:0.5")


# ══════════════════════════════════════════════════════════════════════
# 1. 一次执行的结果
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class StageOutcome:
    """一次命令行执行的产出。

    `test` 为 None 表示这一次**根本没看测试段**（select 档）。这不是
    "没算出来"，是"按纪律没去看"，`text()` 会照实写出来。

    `warnings` 在两档下来源不同但读法一样：final 档由 `run_holdout` 产出
    （含零成交 / 测试段过短 / VALID 与 TRAIN 矛盾），select 档没有 TEST 相关的
    那几条，只留【VALID 与 TRAIN 矛盾】—— 判据与 `run_holdout` 逐字相同。
    """

    stage: str
    target_name: str
    chosen_setting: dict
    chosen_index: int
    n_candidates: int
    train: SegmentResult
    valid: SegmentResult
    test: SegmentResult | None = None
    report: HoldoutReport | None = None
    warnings: list[str] = field(default_factory=list)

    def _line(self, result: SegmentResult) -> str:
        mark = "样本外" if result.out_of_sample else "样本内"
        return (
            f"  {result.segment.name:<5} [{mark}] {self.target_name}="
            f"{result.target(self.target_name):.4f} "
            f"成交 {result.trade_count():.0f} 笔 "
            f"({result.start:%Y-%m-%d} ~ {result.end:%Y-%m-%d})"
        )

    def text(self) -> str:
        """人读的一段文本。**每一行都标了这个数字是样本内还是样本外。**"""
        lines = [
            f"阶段 {self.stage}（目标 {self.target_name}）",
            f"候选参数 {self.n_candidates} 组，选中第 {self.chosen_index} 组："
            f"{self.chosen_setting}",
            self._line(self.train),
            self._line(self.valid),
        ]
        if self.test is not None:
            lines.append(self._line(self.test))
        else:
            lines.append(
                "  TEST  [未查看] 本次按 select 档执行，测试段一根 K 线都没跑；"
                "上面两行都是样本内数字，不能当样本外证据。"
                "确需查看请显式 --stage final。"
            )

        if self.report is not None and self.report.test_significance is not None:
            sig = self.report.test_significance
            lines.append(
                f"  TEST 显著性：Sharpe={sig.sharpe:.3f} t={sig.t_stat:.2f} "
                f"p_block={sig.p_block_bootstrap:.4f} 显著={sig.significant}"
            )
        lines.extend(f"  ⚠ {note}" for note in self.warnings)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# 2. 切分规划
# ══════════════════════════════════════════════════════════════════════

def plan_split(
    bar_datetimes: Sequence[datetime], args: argparse.Namespace
) -> ThreeWaySplit:
    """按命令行参数把 K 线时间戳切成三段。

    给了 `--ratio` 就按比例用满全部样本；否则按根数切，`--anchor end`（默认）
    让 TEST 贴着**最新**的数据，前面多出来的老 K 线留给指标预热。
    """
    if args.ratio:
        train, valid, test = args.ratio
        return split_by_ratio(bar_datetimes, train, valid, test)
    return make_three_way_split(
        bar_datetimes,
        args.train_bars,
        args.valid_bars,
        args.test_bars,
        anchor=args.anchor,
    )


# ══════════════════════════════════════════════════════════════════════
# 3. 账本额度
# ══════════════════════════════════════════════════════════════════════

def authorize_final(
    record: SplitRecord, reset_reason: str | None
) -> SplitRecord:
    """跑 final 档前查账本。额度还在就原样放行，用完了就拒。

    `reset_reason` 非 None 即视为"我知道我在重开测试集"，理由落盘；空白理由
    不算理由（与 `SegmentedRunner.reset_test_budget` 同一口径）。
    """
    if reset_reason is not None:
        return reset_test_budget(record, reset_reason)

    if record.remaining_test_budget() > 0:
        return record

    raise SegmentBudgetRefused(
        f"这份切分的 TEST 段额度 {record.test_budget} 次已用尽"
        f"（账本 {record_path()}）。\n{record.describe()}\n"
        f"反复在测试段上取数 = 把测试集研究成样本内。确需重开请显式给理由："
        f'--reset-test-budget "为什么旧的样本外结论作废"'
    )


# ══════════════════════════════════════════════════════════════════════
# 4. 执行
# ══════════════════════════════════════════════════════════════════════

def run_stage(
    runner: SegmentedRunner,
    settings: Sequence[dict],
    *,
    stage: str,
    target_name: str,
    selector: Selector = argmax_selector,
    significance: bool = True,
    annual_days: int = 252,
) -> StageOutcome:
    """跑一档。`select` 只走 TRAIN + VALID，`final` 才碰 TEST。"""
    if stage not in (SELECT, FINAL):
        raise ValueError(f"stage 只能是 {SELECT!r} 或 {FINAL!r}，收到 {stage!r}")
    if not settings:
        raise ValueError("settings 为空")

    grid = [dict(s) for s in settings]

    if stage == FINAL:
        report = run_holdout(
            runner, grid, target_name=target_name, selector=selector,
            significance=significance, annual_days=annual_days,
        )
        return StageOutcome(
            stage=stage,
            target_name=target_name,
            chosen_setting=report.chosen_setting,
            chosen_index=report.chosen_index,
            n_candidates=report.n_candidates,
            train=report.result(Segment.TRAIN),
            valid=report.result(Segment.VALID),
            test=report.result(Segment.TEST),
            report=report,
            warnings=list(report.warnings),
        )

    # select 档：与 run_holdout 的第 1、2 步逐条相同，只是**没有第 3 步**。
    train_results = runner.scan(grid, Segment.TRAIN)
    scores = np.array(
        [res.target(target_name) for res in train_results], dtype=float
    )
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    if not np.isfinite(scores).any():
        raise ValueError(f"TRAIN 段所有参数的 {target_name} 都无效 —— 无法选参")

    chosen = int(selector(grid, scores))
    if not 0 <= chosen < len(grid):
        raise ValueError(f"selector 返回越界下标 {chosen}（网格 {len(grid)} 组）")

    valid_result = runner.run(grid[chosen], Segment.VALID)
    contradiction = valid_contradicts_train(
        train_results[chosen].target(target_name),
        valid_result.target(target_name),
        target_name,
    )

    return StageOutcome(
        stage=stage,
        target_name=target_name,
        chosen_setting=dict(grid[chosen]),
        chosen_index=chosen,
        n_candidates=len(grid),
        train=train_results[chosen],
        valid=valid_result,
        warnings=[] if contradiction is None else [contradiction],
    )


# ══════════════════════════════════════════════════════════════════════
# 5. 命令行
# ══════════════════════════════════════════════════════════════════════

def _grid_spec(text: str) -> tuple[str, float, float, float]:
    """解析 `name=start:stop:step`。"""
    name, _, body = text.partition("=")
    parts = body.split(":")
    if not name or len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"网格写法应为 name=start:stop:step，收到 {text!r}"
        )
    try:
        start, stop, step = (float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"网格 {text!r} 的三个数解析失败") from exc
    return name.strip(), start, stop, step


def _fixed_spec(text: str) -> tuple[str, str]:
    name, sep, value = text.partition("=")
    if not name or not sep:
        raise argparse.ArgumentTypeError(f"固定参数写法应为 name=value，收到 {text!r}")
    return name.strip(), value


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。默认值就是本项目 700.SEHK / LongOnlyTurtleStrategy 那套。"""
    parser = argparse.ArgumentParser(
        prog="python -m vnpy_ctastrategy.segment_cli",
        description=(
            "三段样本内外回测（TRAIN 选参 / VALID 复核 / TEST 只看一次）。"
            "默认档 select 不碰测试段。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage", choices=[SELECT, FINAL], default=SELECT,
        help="select=只跑 TRAIN+VALID；final=对选中参数在 TEST 段跑一次（消耗账本额度）",
    )
    parser.add_argument(
        "--strategy-path",
        default="/Users/flink/tradingview/vnpy_app/strategies/long_only_turtle_strategy.py",
    )
    parser.add_argument("--strategy-class", default="LongOnlyTurtleStrategy")
    parser.add_argument("--vt-symbol", default="700.SEHK")
    parser.add_argument("--interval", default="d", help="K 线周期（vnpy Interval 的值）")
    parser.add_argument("--data-start", default="2023-07-24", help="数据加载起点 YYYY-MM-DD")
    parser.add_argument("--end", default="2026-07-22", help="数据加载终点 YYYY-MM-DD")

    parser.add_argument("--train-bars", type=int, default=360)
    parser.add_argument("--valid-bars", type=int, default=120)
    parser.add_argument("--test-bars", type=int, default=120)
    parser.add_argument(
        "--anchor", choices=["start", "end"], default="end",
        help="end=三段贴着最新数据排（TEST 最近）；start=贴着最早数据排",
    )
    parser.add_argument(
        "--ratio", nargs=3, type=float, metavar=("TRAIN", "VALID", "TEST"),
        help="改按比例切（和为 1），用满全部样本；给了它就忽略 --*-bars",
    )

    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--rate", type=float, default=0.0011, help="港股综合费率(单边)")
    parser.add_argument("--slippage", type=float, default=0.2)
    parser.add_argument("--pricetick", type=float, default=0.2)
    parser.add_argument("--size", type=float, default=1)
    parser.add_argument("--board-lot", type=int, default=100)
    parser.add_argument("--warmup-bars", type=int, default=120)
    parser.add_argument("--annual-days", type=int, default=252)

    parser.add_argument(
        "--grid", action="append", type=_grid_spec, metavar="NAME=START:STOP:STEP",
        help=f"可重复。缺省为 {' '.join(DEFAULT_GRID)}",
    )
    parser.add_argument(
        "--fixed", action="append", type=_fixed_spec, metavar="NAME=VALUE",
        help="附加到每组参数上的固定字段，可重复",
    )
    parser.add_argument("--target", default="sharpe_ratio")
    parser.add_argument("--no-significance", action="store_true")

    parser.add_argument(
        "--test-budget", type=int, default=1, help="这份切分总共允许查看 TEST 几次"
    )
    parser.add_argument(
        "--reset-test-budget", metavar="REASON", default=None,
        help="重开测试段额度，必须给非空理由（理由落进账本）",
    )
    parser.add_argument(
        "--record-path", default=None,
        help=f"账本路径，缺省 {record_path()}",
    )
    parser.add_argument(
        "--no-record", action="store_true", help="不读写账本（跨进程预算随之失效）"
    )
    return parser


def _build_settings(args: argparse.Namespace) -> list[dict]:
    from vnpy.trader.optimize import OptimizationSetting

    opt = OptimizationSetting()
    for name, start, stop, step in args.grid or [
        _grid_spec(spec) for spec in DEFAULT_GRID
    ]:
        opt.add_parameter(name, start, stop, step)

    settings: list[dict] = opt.generate_settings()

    fixed: dict[str, Any] = {
        "board_lot": args.board_lot,
        "trading_capital": float(args.capital),
    }
    for name, raw in args.fixed or []:
        try:
            fixed[name] = float(raw)
        except ValueError:
            fixed[name] = raw
    for setting in settings:
        setting.update(fixed)
    return settings


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。返回 0=跑完 / 1=被账本拒绝 / 2=数据或参数不成立。"""
    from vnpy.trader.constant import Interval

    from .overfitting import EngineRunner
    from .overfitting_audit import load_strategy_class

    args = build_parser().parse_args(argv)

    strategy_class = load_strategy_class(args.strategy_path, args.strategy_class)
    runner = EngineRunner(
        strategy_class=strategy_class,
        vt_symbol=args.vt_symbol,
        interval=Interval(args.interval),
        rate=args.rate,
        slippage=args.slippage,
        size=args.size,
        pricetick=args.pricetick,
        capital=int(args.capital),
        start=datetime.strptime(args.data_start, "%Y-%m-%d"),
        end=datetime.strptime(args.end, "%Y-%m-%d"),
        annual_days=args.annual_days,
        warmup_bars=args.warmup_bars,
    )

    bar_datetimes = runner.bar_datetimes()
    if not bar_datetimes:
        print(
            f"{args.vt_symbol} 在 {args.data_start} ~ {args.end} 没有 K 线 —— "
            f"先用回测器的[下载数据]或 vnpy_app/data_tools 灌进数据库",
        )
        return 2

    try:
        split = plan_split(bar_datetimes, args)
    except ValueError as exc:
        print(f"切分不成立：{exc}")
        return 2

    record_target: Path | None = None if args.no_record else record_path(args.record_path)
    record: SplitRecord | None = None
    if record_target is not None:
        record = open_record(
            split,
            vt_symbol=args.vt_symbol,
            strategy_class=args.strategy_class,
            interval=args.interval,
            target_name=args.target,
            test_budget=args.test_budget,
            path=record_target,
        )
        if args.stage == FINAL:
            try:
                record = authorize_final(record, args.reset_test_budget)
            except (SegmentBudgetRefused, ValueError) as exc:
                print(exc)
                return 1

    settings = _build_settings(args)
    segmented = SegmentedRunner(runner=runner, split=split, test_budget=1)

    print(f"数据 {len(bar_datetimes)} 根，网格 {len(settings)} 组")
    if record is not None:
        print(record.describe())
    else:
        print("（--no-record：本次不读写账本）")

    outcome = run_stage(
        segmented, settings,
        stage=args.stage,
        target_name=args.target,
        significance=not args.no_significance,
        annual_days=args.annual_days,
    )
    print(outcome.text())

    if record is not None and record_target is not None:
        if outcome.test is not None:
            record = record_peek(
                record,
                SegmentPeek(
                    at=datetime.now(),
                    setting=outcome.chosen_setting,
                    target_name=args.target,
                    target_value=outcome.test.target(args.target),
                    note=f"segment_cli --stage {FINAL}",
                ),
            )
        save_record(record, path=record_target)
        print(f"账本已更新：{record_target}（剩余 {record.remaining_test_budget()} 次）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
