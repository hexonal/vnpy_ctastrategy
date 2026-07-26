"""参数寻优的多重比较闸：DSR（去膨胀夏普）+ PBO（回测过拟合概率）。

━━━ 为什么这一层必须存在 ━━━

`sharpe_inference.py` 的 t 检验、`permutation_test.py` 的分块重排，回答的都是
**"这一条曲线是不是运气"**。它们的 p 值建立在"只看了这一条"之上。

一旦开始扫参数，这个前提就没了：25 组参数取最好的一组，等于做了 25 次检验只报最
显著的那次。样本内最优的那组必然带着 N 组试验里最大的运气分量，它的 t 值、p 值、
Sharpe 全部被选择偏差污染 —— 这是回测里最常见、也最贵的一类自欺。

本模块把两道专治多重比较的闸接进 `BacktestingEngine.run_bf_optimization()` /
`run_ga_optimization()`：

    DSR  ——  把"试了 N 组"折进夏普的显著性判定里。零技能下 N 组试验的最大夏普
             期望 SR* 随 N 增长；观测夏普必须超过 SR* 才谈得上显著。
             实现见 sharpe_inference.deflate_optimization_results（headline）
             与 deflated_sharpe.deflate_optimization（用真实收益矩的详版）。

    PBO  ——  把"样本内最优在样本外还排第几"直接测出来。CSCV 枚举 C(S,S/2) 种
             时间切分，统计"样本内第一名掉进样本外后半段"的概率。
             实现见 overfitting.cscv_pbo / pbo_from_matrix。

━━━ 两个诚实边界（读数前必须知道）━━━

* **两道闸都是否决工具，不是背书工具。** DSR 高 + PBO 低 只说明"没被这两关拦下"，
  不等于样本外会赚钱。反过来，DSR 低 或 PBO 高 是明确的拒绝信号。

* **本项目的典型样本（单标的 600 日线、十几到几十笔完整交易）处于低功效区。**
  低功效下 PBO 天然向 0.5 靠拢，DSR 天然偏低。此时"不显著"的正确读法是
  **"数据不足以支持从这个网格里挑参数"**，而不是"策略一定是假的"。
  两种情况的处置相同（别用寻优出来的参数），结论不同。
  `PBOStudy.diagnosis` 里的最小可检出夏普就是为了让人分清这两者。

━━━ 数据来源：为什么 PBO 必须吃寻优自己产出的矩阵 ━━━

CSCV 需要 (T, N) 的逐期收益矩阵，而 vnpy 的寻优只返回标量 statistics —— 每组参数的
`daily_df` 在子进程里算完就被丢掉了。`overfitting_audit` 绕过这一点的办法是用
`EngineRunner` 把整个网格**重跑一遍**，那是一条与寻优完全平行的代码路径，
连取到的 K 线根数都可能不同（分块查询 vs 单次查询）。

本模块的做法是让寻优流程自己把已经算出来的逐日盈亏带回来
（`backtesting.evaluate(collect_returns=True)` 多返回一个元素），
代价约 14 KiB/组，PBO 与 DSR 因此测的是**寻优当时那一批回测**，不是另一批。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date as Date
from typing import Any

import numpy as np
from pandas import DataFrame

from .deflated_sharpe import (
    DEFAULT_CONFIDENCE,
    DeflatedSharpeResult,
    deflate_optimization,
)
from .overfitting import PBOStudy, pbo_from_matrix, returns_matrix
from .sharpe_inference import SHARPE_TARGET_NAMES, deflate_optimization_results

#: 一组参数的逐日盈亏：(日期序列, 该日 net_pnl)。
#: 由 `backtesting.evaluate(collect_returns=True)` 产出。
ReturnsPayload = tuple[list[Date], np.ndarray]


@dataclass(frozen=True)
class OptimizationGateConfig:
    """两道闸的全部可调项。默认值即生产口径。

    confidence      DSR 判显著的阈值，也是 PBO 零分布检验的 α。
    sharpe_key      从 statistics 里读观测夏普用的键。
    n_blocks        CSCV 的块数 S；None = 按样本长度自动推荐（见 recommend_n_blocks）。
    min_block_obs   自动推荐块数时的最小块长，应取约 1 倍中位持仓天数。
    n_offsets       块网格相位数。块边界位置是任意的，
                    多相位跑一遍才知道 PBO 估计本身稳不稳。
    n_null_sims     PBO 零分布的模拟次数。约 0.09 秒/次；置 0 跳过（不推荐，
                    跳过后只能拿固定阈值判据，没有 p 值 —— `passed` 会因
                    "未校准"直接判不通过）。
    seed            零分布模拟的随机种子，固定以保证同一批结果可复现。
    pbo_max         `passed` 允许的 PBO 点估计上限。原先是写死在 `passed`
                    里的 0.25 字面量，提出来是为了让阈值与判据在同一处可见。
    """

    confidence: float = DEFAULT_CONFIDENCE
    sharpe_key: str = "sharpe_ratio"
    n_blocks: int | None = None
    min_block_obs: int = 20
    n_offsets: int = 8
    n_null_sims: int = 200
    seed: int = 20260725
    pbo_max: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise ValueError(f"confidence 必须落在 (0, 1)，收到 {self.confidence}")
        if not 0.0 < self.pbo_max < 1.0:
            raise ValueError(f"pbo_max 必须落在 (0, 1)，收到 {self.pbo_max}")
        if self.n_blocks is not None and (self.n_blocks < 2 or self.n_blocks % 2):
            raise ValueError(f"n_blocks 必须是 ≥2 的偶数或 None，收到 {self.n_blocks}")
        if self.n_offsets < 1:
            raise ValueError(f"n_offsets 必须 ≥1，收到 {self.n_offsets}")
        if self.n_null_sims < 0:
            raise ValueError(f"n_null_sims 不能为负，收到 {self.n_null_sims}")


@dataclass(frozen=True)
class OptimizationGateReport:
    """一次寻优的多重比较体检结果。

    每个字段都可能是 None —— **None 表示"这道闸没跑成"，不表示"通过"**。
    `dsr_verdict` / `pbo_verdict` 会把没跑成的原因写进 `notes`，
    绝不把缺失当成绿灯（这正是本模块要防的那类自欺）。
    """

    #: 头名那组的 DSR 概要（sharpe_inference.deflate_optimization_results 的返回值）。
    #: 仅当寻优目标属夏普族时可得 —— 该函数对非夏普目标显式 raise，这里如实记为 None。
    dsr: dict[str, float | int | bool] | None

    #: 用真实 (γ₃, γ₄) 与 total_days 算的详版 DSR，被检验的是【网格里夏普最大的那组】。
    dsr_detail: DeflatedSharpeResult | None

    #: CSCV / PBO 研究（点估计 + 相位稳定性 + 零分布 + 样本诊断）。
    pbo: PBOStudy | None

    target_name: str
    annual_days: int
    n_results: int
    matrix_shape: tuple[int, int] | None
    source: str = "bf"
    config: OptimizationGateConfig = field(default_factory=OptimizationGateConfig)
    notes: list[str] = field(default_factory=list)

    # ── 判定 ────────────────────────────────────────────────────────

    @property
    def dsr_values(self) -> list[float]:
        """两条 DSR 路径各自的值（可能 0/1/2 个）。判定取最保守的那个。"""
        out: list[float] = []
        if self.dsr is not None:
            value = float(self.dsr.get("deflated_sharpe_ratio", float("nan")))
            if math.isfinite(value):
                out.append(value)
        if self.dsr_detail is not None:
            value = self.dsr_detail.deflated_sharpe_ratio
            if math.isfinite(value):
                out.append(value)
        return out

    @property
    def dsr_value(self) -> float:
        """保守口径的 DSR：两条路径取最小。都没跑成时返回 nan。

        取 min 而不是取平均或取 headline：两条路径的差别在于用正态矩还是用真实矩，
        真实收益多为负偏厚尾，正态假设会【高估】显著性。既然拿不准，就用低的那个。
        """
        values = self.dsr_values
        return min(values) if values else float("nan")

    @property
    def dsr_significant(self) -> bool | None:
        """DSR 是否过关。None = 没算成，**不是通过**。"""
        value = self.dsr_value
        if not math.isfinite(value):
            return None
        return value >= self.config.confidence

    @property
    def dsr_verdict(self) -> str:
        significant = self.dsr_significant
        if significant is None:
            return "DSR 未计算（见 notes）"
        if significant:
            return (
                f"DSR={self.dsr_value:.3f} ≥ {self.config.confidence:.2f}："
                f"扣掉选参偏差后仍显著"
            )
        return (
            f"DSR={self.dsr_value:.3f} < {self.config.confidence:.2f}："
            f"挑出来的，不可采信"
        )

    @property
    def pbo_verdict(self) -> str:
        if self.pbo is None:
            return "PBO 未计算（见 notes）"
        if self.pbo.null is None:
            # 没有零分布就没有 p 值，只能拿固定阈值说话。说清楚这一点，
            # 免得"稳健"被当成已经排除了运气。
            return f"{self.pbo.verdict}（PBO 未校准：未跑零分布，无 p 值）"
        return self.pbo.verdict

    @property
    def passed(self) -> bool:
        """两道闸是否都明确通过。任一没算成 = 未通过（缺证据不是证据）。

        PBO 一侧必须与 `pbo_verdict` 用同一套判据，否则同一份报告会自相
        矛盾：`text()` 印"与'选参毫无信息'不可区分"，而按 `.passed` 自动
        放行的脚本把这个网格判为可上生产。零分布已算时，要求点估计低于
        `PBO_MAX` **且** 在零分布下显著；没算零分布 = 未校准 = 不通过。
        """
        if self.dsr_significant is not True or self.pbo is None:
            return False
        if self.pbo.result.pbo > self.config.pbo_max:
            return False
        if self.pbo.null is None:
            return False                    # 未校准，缺证据不是证据
        p_value = self.pbo.p_value
        if not math.isfinite(p_value):
            return False
        return p_value < 1.0 - self.config.confidence

    def as_dict(self) -> dict[str, Any]:
        """只含标量与嵌套 dict，便于落 JSON / CSV。"""
        out: dict[str, Any] = {
            "target_name": self.target_name,
            "annual_days": self.annual_days,
            "n_results": self.n_results,
            "source": self.source,
            "matrix_shape": self.matrix_shape,
            "dsr_value": self.dsr_value,
            "dsr_significant": self.dsr_significant,
            "dsr_verdict": self.dsr_verdict,
            "pbo_verdict": self.pbo_verdict,
            "passed": self.passed,
            "notes": list(self.notes),
        }
        out["dsr"] = dict(self.dsr) if self.dsr is not None else None
        detail = self.dsr_detail
        out["dsr_detail"] = detail.as_dict() if detail is not None else None
        out["pbo"] = self.pbo.as_dict() if self.pbo is not None else None
        return out

    def text(self) -> str:
        """人读报告。"""
        lines = [
            "═" * 66,
            f"寻优多重比较闸（目标 {self.target_name}，{self.n_results} 组参数，"
            f"{'穷举' if self.source == 'bf' else '遗传'}）",
            "─" * 66,
            f"  {self.dsr_verdict}",
        ]
        headline = self.dsr
        if headline is not None:
            best = float(headline.get("best_sharpe_annual", float("nan")))
            std = float(headline.get("sharpe_std_annual", float("nan")))
            sr_star = float(headline.get("expected_max_sharpe_annual", float("nan")))
            lines.append(
                f"    headline  最优夏普 {best:+.2f} / 试验间 std {std:.2f}"
                f" / 零技能最大夏普期望 SR* {sr_star:.2f}"
            )
        detail = self.dsr_detail
        if detail is not None:
            lines.extend("    " + line for line in detail.summary().splitlines())
        lines.append("─" * 66)
        if self.pbo is not None:
            lines.append(self.pbo.text())
        else:
            lines.append(f"  {self.pbo_verdict}")
        if self.notes:
            lines.append("─" * 66)
            lines.append("  备注")
            lines.extend(f"    · {note}" for note in self.notes)
        lines.append("═" * 66)
        return "\n".join(lines)


class OptimizationResults(list):
    """寻优返回值：**就是原来那个 list**，额外挂一个 `.gates`。

    向后兼容是硬要求 —— 已有调用方（UI、脚本、`deflate_optimization`、
    `get_target_value`）拿到的仍然是按目标值降序排好的
    `list[tuple[setting, target, statistics]]`，`len()` / 下标 / 解包 / 迭代
    行为逐字不变。新增的两道闸只能【附加】信息，不能改变既有形状，
    否则"加一道闸"会变成"破坏所有下游"。
    """

    def __init__(
        self,
        results: Sequence[tuple] = (),
        gates: OptimizationGateReport | None = None,
    ) -> None:
        super().__init__(results)
        self.gates: OptimizationGateReport | None = gates


def returns_matrix_from_payloads(
    payloads: Sequence[ReturnsPayload | None],
    capital: float,
) -> tuple[np.ndarray, list[int]]:
    """`evaluate(collect_returns=True)` 的逐日盈亏 → CSCV 需要的 (T, N) 对数收益矩阵。

    直接复用 `overfitting.returns_matrix`（按日期并集对齐、缺失日填 0 收益、
    口径与面板的 `sharpe_ratio` 逐字一致），本函数只负责把 payload 还原成
    它要的 daily_df 形状。**不另写一套对齐逻辑** —— 两套口径一旦分叉，
    PBO 里的夏普就和面板上的夏普不是同一个数了。

    返回 (矩阵, 被剔除的列下标)。空 payload（该组回测没有任何成交日）会被剔除，
    下标随返回值交回调用方，由调用方决定怎么在报告里交代。

    **爆仓组同样剔除**，理由与"空"不同，必须单独说清：引擎的
    `calculate_statistics` 遇到 balance ≤ 0 会走 positive_balance 分支，把
    statistics 清成全 0（total_days=0）—— 但它照样把 net_pnl 序列交回来。
    而 `daily_log_returns` 复刻引擎口径时只做 `x[x <= 0] = nan`：穿零之后
    balance 一直为负，x = 负/负 = **正**，于是爆仓段的每一天都算出正的对数
    收益。CSCV 眼里那就是一条单调"盈利"的高夏普列，样本内外双双第一，
    足以把整个网格的 PBO 从 0.5 拉到 0 —— 一个本该被拒的网格因此拿到绿灯。
    引擎已经判定这组参数把账户打穿了，这里就不该让它的净值曲线继续投票。
    """
    frames: list[DataFrame] = []
    for payload in payloads:
        if payload is None:
            frames.append(DataFrame())
            continue
        index, net_pnl = payload
        values = np.asarray(net_pnl, dtype=float)
        if len(index) == 0 or values.size == 0:
            frames.append(DataFrame())
            continue
        if len(index) != values.size:
            raise ValueError(
                f"payload 的日期数 {len(index)} 与 net_pnl 长度 {values.size} 不一致"
            )
        if went_bankrupt(values, capital):
            frames.append(DataFrame())
            continue
        frames.append(DataFrame({"net_pnl": values}, index=list(index)))
    return returns_matrix(frames, capital)


def went_bankrupt(net_pnl: Sequence[float] | np.ndarray, capital: float) -> bool:
    """账户在这条净值曲线上是否被打穿过。

    复刻引擎 `calculate_statistics` 的 positive_balance 判据（balance =
    net_pnl.cumsum() + capital，任一日 ≤ 0 即爆仓），这样"哪些组算爆仓"
    在引擎和这道闸里是同一个定义。
    """
    values = np.asarray(net_pnl, dtype=float)
    if values.size == 0:
        return False
    balance = np.cumsum(values) + capital
    return bool(np.any(balance <= 0.0))


def bankrupt_columns(
    payloads: Sequence[ReturnsPayload | None],
    capital: float,
) -> list[int]:
    """`payloads` 里被引擎判定爆仓的列下标。

    单独一个函数而不是塞进 `returns_matrix_from_payloads` 的返回值：后者的
    `(矩阵, 被剔除下标)` 二元组是既有调用方的形状，改成三元组等于为了一条
    备注去破坏所有下游。剔除仍在矩阵构建里做，这里只负责回答"为什么被剔"。
    """
    return [
        column
        for column, payload in enumerate(payloads)
        if payload is not None
        and len(payload[0]) > 0
        and went_bankrupt(payload[1], capital)
    ]


def _statistics_of(item: Any) -> dict[str, Any] | None:
    """从一条寻优结果里取 statistics。

    接受两种形状：`(setting, target, statistics)` 三元组（vnpy 寻优的真实返回值）
    与裸 statistics dict（手工构造的结果集，`deflated_sharpe` 的两个入口也这么收）。

    返回 None 只表示"这条根本不是结果记录"。回测爆仓那一组返回的是**空 dict**，
    两者必须分开：前者是调用方传错了东西，后者是一次真实但没有绩效的试验，
    它仍要计入 N（它确实被搜索过），只是进不了夏普分布。
    """
    if isinstance(item, dict):
        return item
    is_row = isinstance(item, Sequence) and not isinstance(item, str | bytes)
    if is_row and len(item) >= 3:
        stats = item[2]
        if isinstance(stats, dict):
            return stats
    return None


def _selected_statistics(results: Sequence[tuple]) -> dict[str, Any]:
    """取头名那组的 statistics（results 已按目标值降序）。取不到则返回空 dict。"""
    if not results:
        return {}
    return _statistics_of(results[0]) or {}


def _read_float(statistics: dict[str, Any], key: str) -> float | None:
    """读一个有限浮点数；缺键 / 类型不对 / nan / inf 一律返回 None。"""
    if key not in statistics:
        return None
    try:
        value = float(statistics[key])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _argmax_sharpe(results: Sequence[tuple], sharpe_key: str) -> int | None:
    """网格里夏普最大那组的下标。全都读不到夏普时返回 None。

    详版 DSR 必须打在这一组上：SR*（零技能下 N 组试验的最大夏普期望）是**最大值
    序统计**的期望，整套推导的前提就是"被检验的这一个是 N 个里最大的那个"。
    寻优目标若不是夏普（例如 return_drawdown_ratio），results[0] 只是那个目标下的
    第一名，夏普未必最大，对它套 SR* 是错配。
    """
    best_index: int | None = None
    best_value = -math.inf
    for i, item in enumerate(results):
        stats = _statistics_of(item)
        if stats is None:
            continue
        value = _read_float(stats, sharpe_key)
        if value is not None and value > best_value:
            best_value = value
            best_index = i
    return best_index


def _average_round_trips(results: Sequence[tuple]) -> float | None:
    """全网格的平均完整回合数，用 `total_trade_count / 2` 估。

    喂给 `sample_size_diagnosis`，它据此判断"排名是不是由少数几笔的运气决定"。
    一个回合至少一买一卖，所以除以 2；金字塔加仓（4 买 1 卖 = 5 笔成交 = 1 个回合）
    会被高估一倍以上，调用方拿到的样本诊断因此偏宽松 —— 这条口径限制由
    `run_optimization_gates` 写进备注，不藏着。

    一组都读不到 `total_trade_count` 时返回 None（诊断里留空），
    而不是返回 0 假装"数出来是 0 笔"。
    """
    counts: list[float] = []
    for item in results:
        stats = _statistics_of(item)
        if stats is None:
            continue
        value = _read_float(stats, "total_trade_count")
        if value is not None:
            counts.append(value)
    if not counts:
        return None
    return sum(counts) / len(counts) / 2.0


def run_optimization_gates(
    results: Sequence[tuple],
    target_name: str,
    annual_days: int,
    capital: float,
    payloads: Sequence[ReturnsPayload | None] | None = None,
    config: OptimizationGateConfig | None = None,
    source: str = "bf",
) -> OptimizationGateReport:
    """对一次寻优的完整结果跑 DSR + PBO。

    results     `run_bf_optimization` / `run_ga_optimization` 的返回值（全部试验，
                已按目标值降序）。**必须是全部**，不能是截断过的 top-N ——
                N 与试验夏普分布是 DSR 的输入，截断会让 SR* 系统性偏低、DSR 偏乐观。
    payloads    与 results 逐一对应的逐日盈亏；None 表示这次寻优没收集，
                此时 PBO 无法计算（如实记 None + 备注，不做任何替代估计）。
    source      'bf' 穷举 / 'ga' 遗传。GA 会向高适应度收敛，试验夏普被截尾 →
                std 偏小 → SR* 偏低 → **DSR 偏乐观**，这里强制写一条备注。

    任何一道闸算不出来都不抛异常 —— 寻优已经跑完了，把它炸掉只会让人丢掉结果、
    下次干脆不开闸。但也绝不静默：失败原因一律进 `notes`，判定一律记为"未计算"。
    """
    cfg = config or OptimizationGateConfig()
    notes: list[str] = []
    n_results = len(results)

    if source == "ga":
        notes.append(
            "遗传算法：试验夏普分布被收敛截尾（std 偏小 → SR* 偏低 → DSR 偏乐观）。"
            "要拿 DSR 当验收闸，应在同一参数空间用穷举采一次样估试验间 std，"
            "再显式传给 deflated_sharpe.deflate_optimization"
        )

    selected_stats = _selected_statistics(results)

    # ── 闸一：DSR headline（按寻优目标本身的量纲）──────────────────
    dsr: dict[str, float | int | bool] | None = None
    if not results:
        notes.append("寻优结果为空，DSR 与 PBO 都无从算起")
    elif target_name not in SHARPE_TARGET_NAMES:
        notes.append(
            f"寻优目标 {target_name!r} 不属夏普族"
            f"（{', '.join(sorted(SHARPE_TARGET_NAMES))}），"
            f"headline DSR 不适用：DSR 把试验值当年化夏普做最大值序统计校正，"
            f"喂别的量纲进去算出的数没有统计含义。下方详版 DSR 改打在"
            f"【夏普最大的那组】上，它回答的是「这个网格里最好的夏普是否显著」，"
            f"不是「你按 {target_name} 选出的那组是否显著」"
        )
    else:
        n_periods = 0
        try:
            n_periods = int(selected_stats.get("total_days", 0))
        except (TypeError, ValueError):
            n_periods = 0
        if n_periods < 2:
            notes.append(
                "头名那组的 statistics 里没有可用的 total_days（该组可能已爆仓），"
                "headline DSR 跳过"
            )
        else:
            try:
                dsr = deflate_optimization_results(
                    results,
                    n_periods=n_periods,
                    target_name=target_name,
                    annual_days=annual_days,
                )
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                notes.append(f"headline DSR 计算失败：{type(exc).__name__}: {exc}")

    # ── 闸一之详版：用真实收益矩，打在夏普最大的那组上 ──────────────
    dsr_detail: DeflatedSharpeResult | None = None
    if results:
        index = _argmax_sharpe(results, cfg.sharpe_key)
        if index is None:
            notes.append(
                f"没有任何一组的 statistics 带 {cfg.sharpe_key!r}"
                f"（全网格爆仓？），详版 DSR 跳过"
            )
        else:
            if index != 0:
                notes.append(
                    f"详版 DSR 打在第 {index} 组（夏普最大的那组），"
                    f"不是按 {target_name} 排第一的那组"
                )
            try:
                dsr_detail = deflate_optimization(
                    results,
                    annual_days=annual_days,
                    selected_index=index,
                    confidence=cfg.confidence,
                    sharpe_key=cfg.sharpe_key,
                )
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                notes.append(f"详版 DSR 计算失败：{type(exc).__name__}: {exc}")

    # ── 闸二：PBO ────────────────────────────────────────────────
    pbo: PBOStudy | None = None
    matrix_shape: tuple[int, int] | None = None
    if payloads is None:
        if results:
            notes.append(
                "本次寻优未收集逐日收益，PBO 无法计算。"
                "PBO 需要 (T, N) 收益矩阵，寻优默认只带回标量 statistics —— "
                "用 run_bf_optimization(collect_returns=True) 重跑才有"
            )
    elif len(payloads) != n_results:
        notes.append(
            f"逐日收益条数 {len(payloads)} 与结果组数 {n_results} 不一致，PBO 跳过"
        )
    elif results:
        bankrupt: list[int] = []
        try:
            matrix, dropped = returns_matrix_from_payloads(payloads, capital)
            bankrupt = bankrupt_columns(payloads, capital)
        except ValueError as exc:
            matrix, dropped = np.zeros((0, 0), dtype=float), []
            notes.append(f"收益矩阵构建失败：{exc}")
        matrix_shape = (int(matrix.shape[0]), int(matrix.shape[1]))
        pbo_notes: list[str] = []
        if dropped:
            pbo_notes.append(
                f"{len(dropped)} 组参数逐日收益为空，已剔除（下标 {dropped[:10]}）"
            )
        if bankrupt:
            # 留痕而不是静默丢：被剔除的是"引擎判定爆仓"的组，这件事本身
            # 就是这次寻优的结论之一（杠杆/手数扫过头了），不能只在 PBO
            # 里消失。
            bankrupt_note = (
                f"{len(bankrupt)} 组参数在回测中爆仓（balance ≤ 0），"
                f"已从 PBO 矩阵剔除（下标 {bankrupt[:10]}）—— "
                f"其净值曲线穿零后对数收益会翻正，留下会把 PBO 假性压低"
            )
            pbo_notes.append(bankrupt_note)
            notes.append(bankrupt_note)
        if matrix.size == 0:
            notes.append("所有参数组的逐日收益都为空，PBO 无从算起")
        else:
            avg_round_trips = _average_round_trips(results)
            extra_notes: list[str] = []
            if avg_round_trips is not None:
                extra_notes.append(
                    "回合数按 total_trade_count/2 估算：金字塔加仓策略会被高估，"
                    "样本诊断偏宽松"
                )
            try:
                pbo = pbo_from_matrix(
                    matrix,
                    annual_days=annual_days,
                    n_blocks=cfg.n_blocks,
                    min_block_obs=cfg.min_block_obs,
                    n_offsets=cfg.n_offsets,
                    n_null_sims=cfg.n_null_sims,
                    seed=cfg.seed,
                    avg_round_trips=avg_round_trips,
                    notes=pbo_notes,
                    extra_notes=extra_notes,
                )
            except (ValueError, TypeError) as exc:
                notes.append(f"PBO 计算失败：{type(exc).__name__}: {exc}")

    return OptimizationGateReport(
        dsr=dsr,
        dsr_detail=dsr_detail,
        pbo=pbo,
        target_name=target_name,
        annual_days=annual_days,
        n_results=n_results,
        matrix_shape=matrix_shape,
        source=source,
        config=cfg,
        notes=notes,
    )
