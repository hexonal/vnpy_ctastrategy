"""样本外准入裁决 —— 把 Walk-Forward 与 CSCV/PBO 合成一个可执行的 GO / NO-GO / 无法裁决。

`overfitting.py` 提供的是【零件】：切分、CSCV、零分布、显著性检验。
本模块提供的是【裁决】：把这些零件按正确顺序装起来，并且——这是本模块存在的理由——
在裁决之前先检查**这个样本有没有资格被裁决**。

━━━ 为什么需要这一层（本项目实测催生·两次真实运行） ━━━

标的 700.SEHK，2024-01-02 ~ 2026-07-23 共 627 根日线，
参数网格 entry_window×exit_window×atr_stop = 160 组，
成本 rate=0.0011 单边 + slippage=0.2，warmup_bars=120。

【运行 A · trading_capital=100k】暴露了本模块要修的那个陷阱：

    Walk-Forward（train=252, test=63, 5 折）
      折 0  IS 年化 +36.3%  →  OOS 年化   0.0%   效率 0.00
      折 1  IS 年化 +30.3%  →  OOS 年化 -13.2%   效率 -0.43
      折 2  IS 年化 +11.5%  →  OOS 年化 +11.4%   效率 0.99
      折 3  IS 年化 +31.3%  →  OOS 年化   0.0%   效率 0.00
      折 4  IS 年化 +44.1%  →  OOS 年化   0.0%   效率 0.00
      → 效率中位数 0.00

「效率中位数 0.00」看上去是压倒性的衰减证据。**它不是。**
那三个 0.0% 的折不是「策略失效」，是那 63 个交易日里**一笔都没开仓**——
该策略全样本 627 天只有 9~10 个完整回合（约 70 天一个回合），
一个 63 天的测试窗期望成交 0.9 个回合，出现 0 个完全正常。
把「没交易」记成「效率 0」再取中位数，得到的是**采样假象，不是衰减**。

于是本模块做两件事：
  1. 逐折数真实回合数（按仓位序列，不是 fills/2），把无交易折剔出效率统计；
     并在样本根本撑不起逐折统计时**直接拒绝报告该指标**（`plan_walk_forward`）。
  2. 把 PBO 的三种结局（有害 / 无法区分 / 通过）与 Walk-Forward 的结论按正确逻辑合并——
     **低功效下「证明不了稳健」与「证明了有害」是两回事**：前者是「无法裁决」，
     后者才是「拒绝」。混为一谈会让人以为所有策略都被否定，从而干脆不看这份报告。

【运行 B · trading_capital=1M，即本模块 CLI 默认值，可直接复现】：

    python -m vnpy_ctastrategy.overfitting_audit

    逐折：IS +42.2/+47.1/+30.2/+36.2/+69.7%  →  OOS +97.7/+23.6/-18.0/-33.1/-12.6%
          5 折均有成交（1~6 个回合），效率中位数 -0.18
    拼接样本外：年化 +11.5%，最大回撤 -19.2%，Sharpe +0.52 ± 0.89(SE)，
                block-bootstrap p = 0.613  → 与 0 不可区分
    对照（教科书默认 20/10/2.0）：样本外 Sharpe +1.18
    PBO = 0.841（8 个块网格相位区间 [0.819, 0.887]），
    零分布(200 次模拟, T=616/N=160/S=14) 均值 0.506、95% 分位 0.743，单侧 p = 0.995
    截面 rank IC = -0.211（样本内排名在样本外系统性反转）
    → 裁决 NO-GO

即：**在这个参数空间上做优化，比不优化更差。** 这正是「样本外 -19.5%」那类事故的
统计指纹。注意该结论**不依赖高功效**：低功效让人证明不了「稳健」，
但「样本内最优在样本外系统性落到后半段」是在分布的另一侧尾巴上被拒绝的，
证据方向相反，在小样本上依然成立。

结论对块数 S 不敏感（同一条收益矩阵，S 从 6 到 16）：
    S= 6 块长104天 PBO=0.900 rankIC=-0.505     S=12 块长52天 PBO=0.937 rankIC=-0.391
    S= 8 块长 78天 PBO=0.800 rankIC=-0.283     S=14 块长44天 PBO=0.887 rankIC=-0.354
    S=10 块长 62天 PBO=0.937 rankIC=-0.516     S=16 块长39天 PBO=0.906 rankIC=-0.386
全部高于各自零分布的 95% 分位。只有 S=4 退化（C(4,2)=6 个组合，无分辨率）不可用。

━━━ 边界（必须一起读）━━━

* 本模块只能否决，不能背书。全部通过 = 「没被这几关拦下」，不等于样本外会赚钱。
* 627 日 / 每侧约 4~5 个回合的样本，`sample_size_diagnosis` 给出最小可检出
  Sharpe ≈ 2.5。真实 Sharpe 在 0.5-1.0 的策略在这个样本上**无法被证明有效**。
  这是数据量的物理墙，不是方法的缺陷，换指标绕不过去。
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from pandas import DataFrame

from .overfitting import (
    EngineRunner,
    PBOStudy,
    SampleDiagnosis,
    Selector,
    WalkForwardFold,
    WalkForwardReport,
    argmax_selector,
    format_pbo_report,
    format_walk_forward_report,
    make_walk_forward_splits,
    pbo_from_settings,
    run_walk_forward,
    sample_size_diagnosis,
)

__all__ = [
    "AuditResult",
    "FoldActivity",
    "WalkForwardPlan",
    "count_round_trips",
    "effective_efficiency",
    "fold_activity",
    "format_audit",
    "load_strategy_class",
    "main",
    "mean_holding_days",
    "plan_walk_forward",
    "run_audit",
]


# ══════════════════════════════════════════════════════════════════════
# 1. 回合计数 —— 效率指标的分母到底有没有内容
# ══════════════════════════════════════════════════════════════════════

def count_round_trips(df: DataFrame) -> tuple[int, int]:
    """从 daily_df 数出 (开仓次数, 完整平仓次数)。

    为什么不用 `trade_count`：`trade_count` 数的是**成交笔数（fills）**。
    海龟会金字塔加仓，一个回合可能是 4 笔买入 + 1 笔卖出 = 5 笔成交。
    用 `fills / 2` 估回合数在加仓策略上会系统性【高估】一倍以上，
    而回合数正是小样本诊断里唯一真正重要的分母，估错方向就等于自欺。

    这里改用仓位序列：`end_pos` 由 0 变正 = 开一个回合；由正变 0 = 平掉一个回合。
    对 long-only 策略这是精确的（不是近似）；含空头的策略同样成立，
    因为判据用的是"是否为 0"而不是符号。

    末尾仍持仓时，开仓次数会比完整平仓次数多 1 —— 这是对的：
    那个回合的盈亏还没实现，不该计入"完成了几笔"。
    """
    if df is None or df.empty or "end_pos" not in df.columns:
        return 0, 0

    pos = np.asarray(df["end_pos"].to_numpy(), dtype=float)
    prev = np.empty_like(pos)
    prev[0] = float(df["start_pos"].iloc[0]) if "start_pos" in df.columns else 0.0
    prev[1:] = pos[:-1]

    flat_prev = np.isclose(prev, 0.0)
    flat_now = np.isclose(pos, 0.0)
    opened = int(np.count_nonzero(flat_prev & ~flat_now))
    closed = int(np.count_nonzero(~flat_prev & flat_now))
    return opened, closed


@dataclass(frozen=True)
class FoldActivity:
    """逐折的"这一折到底有没有发生交易"。效率指标只有在 traded 的折上才有意义。"""

    fold_index: int
    fills: int
    round_trips_opened: int
    round_trips_closed: int
    traded: bool
    efficiency: float
    is_annual_return: float
    oos_annual_return: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "fills": self.fills,
            "round_trips_opened": self.round_trips_opened,
            "round_trips_closed": self.round_trips_closed,
            "traded": self.traded,
            "efficiency": self.efficiency,
            "is_annual_return": self.is_annual_return,
            "oos_annual_return": self.oos_annual_return,
        }


def mean_holding_days(df: DataFrame) -> float:
    """平均每个完整回合持仓多少个交易日。CSCV 的块长必须 ≥2× 这个数。

    分子是"有仓位的交易日数"，分母是完整平仓次数。末尾未平仓的那个回合不计入分母，
    但它的持仓日仍在分子里 —— 这会让结果略微偏大，方向是**保守的**
    （高估持仓周期 → 要求更长的块 → 更谨慎的 S），所以不做修正。
    """
    if df is None or df.empty or "end_pos" not in df.columns:
        return float("nan")
    _, closed = count_round_trips(df)
    if closed <= 0:
        return float("nan")
    held = int(np.count_nonzero(~np.isclose(df["end_pos"].to_numpy(dtype=float), 0.0)))
    return held / closed


def fold_activity(folds: Sequence[WalkForwardFold]) -> list[FoldActivity]:
    """逐折算成交活跃度。`traded` 的定义是【样本外窗口内至少开了一个回合】。

    注意不能用"OOS 年化 != 0"来判断：一折可能开了仓、盈亏恰好抵消；
    也可能带着上一折的持仓进来（warmup 造成）只吃到持仓盈亏而没有新开仓。
    前者应计入效率统计，后者不应 —— 只有仓位序列能分清。
    """
    out: list[FoldActivity] = []
    for f in folds:
        opened, closed = count_round_trips(f.oos_daily_df)
        fills = (
            int(f.oos_daily_df["trade_count"].sum())
            if f.oos_daily_df is not None
            and not f.oos_daily_df.empty
            and "trade_count" in f.oos_daily_df.columns
            else 0
        )
        out.append(
            FoldActivity(
                fold_index=f.split.index,
                fills=fills,
                round_trips_opened=opened,
                round_trips_closed=closed,
                traded=opened > 0,
                efficiency=f.efficiency,
                is_annual_return=f.is_annual_return,
                oos_annual_return=f.oos_annual_return,
            )
        )
    return out


def effective_efficiency(activity: Sequence[FoldActivity]) -> tuple[float, int]:
    """只在【真的交易过】的折上算 WF 效率中位数，返回 (中位数, 参与折数)。

    这是本模块对 `WalkForwardReport.efficiency_median` 的修正。
    无交易折的效率恒为 0，把它们计入会把中位数系统性拉向 0，
    制造"严重衰减"的假象（本项目实测：5 折里 3 折无交易 → 中位数 0.00，
    剔除后只剩 2 折有效 —— 而 2 折的中位数本身也没有统计意义，
    所以真正的结论是"这个切分方案根本不支持逐折效率分析"，见 plan_walk_forward）。
    """
    vals = [a.efficiency for a in activity if a.traded and math.isfinite(a.efficiency)]
    if not vals:
        return float("nan"), 0
    return float(np.median(vals)), len(vals)


# ══════════════════════════════════════════════════════════════════════
# 2. 切分可行性 —— "600 日能切几折/几块" 的正面回答
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class WalkForwardPlan:
    """给定样本长度与该策略的真实交易频率，Walk-Forward 到底切不切得动。"""

    n_bars: int
    round_trips_total: int
    bars_per_round_trip: float
    train_bars: int
    test_bars: int
    n_folds: int
    expected_round_trips_per_fold: float
    feasible: bool                     # 折数与每折回合数是否同时达标
    per_fold_efficiency_usable: bool   # 逐折效率指标是否可报
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_bars": self.n_bars,
            "round_trips_total": self.round_trips_total,
            "bars_per_round_trip": self.bars_per_round_trip,
            "train_bars": self.train_bars,
            "test_bars": self.test_bars,
            "n_folds": self.n_folds,
            "expected_round_trips_per_fold": self.expected_round_trips_per_fold,
            "feasible": self.feasible,
            "per_fold_efficiency_usable": self.per_fold_efficiency_usable,
            "notes": list(self.notes),
        }


def plan_walk_forward(
    n_bars: int,
    round_trips_total: int,
    train_bars: int = 252,
    test_bars: int = 63,
    min_folds: int = 4,
    min_round_trips_per_fold: int = 5,
) -> WalkForwardPlan:
    """回答"这个样本切成这样，逐折效率还能不能看"。

    两个约束天然打架，600 日样本上通常**无法同时满足**：

        折数 ≥ min_folds                       → 测试窗要短
        每折期望回合数 ≥ min_round_trips_per_fold → 测试窗要长

    本项目实测：627 根日线、约 10 个完整回合 → 平均 63 根 K 线一个回合。
    要每折 5 个回合就得 test_bars ≈ 315，那么 (627-252-315)/315 + 1 = 1 折。
    要 4 折就得 test_bars ≈ 63，每折期望回合数 ≈ 1。
    **两者不可兼得 —— 这不是参数没调好，是 600 日样本对日线趋势策略的物理上限。**

    结论落到 `per_fold_efficiency_usable`：为 False 时，
    逐折效率（含中位数）**不可作为判据**，只能看拼接后那条样本外曲线的整体显著性
    （它把所有折的交易汇集起来，回合数是各折之和，是这个样本上唯一站得住的统计量）。
    """
    notes: list[str] = []
    n_bars = int(n_bars)
    round_trips_total = max(0, int(round_trips_total))

    step = max(1, test_bars)
    n_folds = 0
    i = 0
    while train_bars + i + test_bars <= n_bars:
        n_folds += 1
        i += step

    if round_trips_total > 0:
        bars_per_rt = n_bars / round_trips_total
        expected = test_bars / bars_per_rt
    else:
        bars_per_rt = float("inf")
        expected = 0.0

    enough_folds = n_folds >= min_folds
    enough_trades = expected >= min_round_trips_per_fold

    if not enough_folds:
        notes.append(
            f"只切得出 {n_folds} 折（要求 ≥{min_folds}）："
            f"train={train_bars} + test={test_bars} 相对 {n_bars} 根样本太长"
        )
    if round_trips_total <= 0:
        notes.append("全样本回合数为 0：策略在该区间从未开仓，任何样本外分析都无意义")
    elif not enough_trades:
        notes.append(
            f"平均 {bars_per_rt:.0f} 根 K 线才有一个完整回合 → 每折期望仅 {expected:.1f} 个回合"
            f"（要求 ≥{min_round_trips_per_fold}）：会出现大量【零交易折】，"
            f"其效率恒为 0，把它们计入中位数就是把采样假象当成衰减"
        )

    if not enough_trades:
        need_test = (
            int(math.ceil(min_round_trips_per_fold * bars_per_rt))
            if math.isfinite(bars_per_rt) else 0
        )
        if need_test:
            folds_at_need = max(
                0, (n_bars - train_bars - need_test) // max(1, need_test) + 1
            ) if n_bars - train_bars - need_test >= 0 else 0
            notes.append(
                f"要每折 {min_round_trips_per_fold} 个回合需 test_bars≈{need_test}，"
                f"那样只剩 {folds_at_need} 折 —— 折数与每折回合数在本样本上不可兼得，"
                f"**逐折效率不可报，只报拼接样本外曲线的整体显著性**"
            )

    return WalkForwardPlan(
        n_bars=n_bars,
        round_trips_total=round_trips_total,
        bars_per_round_trip=bars_per_rt,
        train_bars=train_bars,
        test_bars=test_bars,
        n_folds=n_folds,
        expected_round_trips_per_fold=expected,
        feasible=bool(enough_folds and enough_trades),
        per_fold_efficiency_usable=bool(enough_folds and enough_trades),
        notes=notes,
    )


# ══════════════════════════════════════════════════════════════════════
# 3. 合并裁决
# ══════════════════════════════════════════════════════════════════════

GO = "GO"
NO_GO = "NO-GO"
UNDECIDABLE = "无法裁决"


@dataclass
class AuditResult:
    """一次完整准入审查。三态结论，不是二态 —— 这是本模块最重要的设计决定。

        NO-GO       有【正面证据】表明这套优化结果站不住（PBO 落在有害尾部 /
                    样本外显著为负 / 打不过默认参数）。
        无法裁决     没有正面反证，但样本功效不足以支持"稳健"的主张。
                    **处置与 NO-GO 相同（不许拿优化参数下真钱），但结论不同**：
                    NO-GO 说"这个策略被证伪了"，无法裁决说"这份数据不够格发证"。
        GO          全部关卡通过。仍然只意味着"没被拦下"。
    """

    plan: WalkForwardPlan
    diagnosis: SampleDiagnosis
    wf: WalkForwardReport
    activity: list[FoldActivity]
    eff_median_traded: float
    n_traded_folds: int
    pbo: PBOStudy | None
    decision: str
    reasons: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "decision": self.decision,
            "reasons": list(self.reasons),
            "blockers": list(self.blockers),
            "eff_median_traded": self.eff_median_traded,
            "n_traded_folds": self.n_traded_folds,
            "plan": self.plan.as_dict(),
            "diagnosis": self.diagnosis.as_dict(),
            "walk_forward": self.wf.as_dict(),
            "fold_activity": [a.as_dict() for a in self.activity],
        }
        if self.pbo is not None:
            out["pbo"] = self.pbo.as_dict()
        return out

    def text(self) -> str:
        return format_audit(self)


def _pbo_state(study: PBOStudy, alpha: float) -> tuple[str, str]:
    """把 PBO 研究压成 (状态, 人读理由)。状态 ∈ {harmful, inconclusive, pass}。

    三态而非二态的理由见模块头：**"证明不了稳健"与"证明了有害"是不同的结论**，
    在低功效样本上尤其不能混。前者对应 inconclusive，后者对应 harmful。
    """
    pbo = study.result.pbo
    null = study.null
    if null is None:
        if pbo > 0.5:
            return "harmful", f"PBO={pbo:.3f} > 0.5（未跑零分布，用固定阈值兜底）"
        if pbo <= 0.25:
            return "pass", f"PBO={pbo:.3f} ≤ 0.25（未跑零分布，判据未校准）"
        return "inconclusive", f"PBO={pbo:.3f} 处于 0.25-0.50 灰区（未跑零分布）"

    p = null.p_value(pbo)
    if pbo > null.q95:
        return "harmful", (
            f"PBO={pbo:.3f} 高于零分布 95% 分位 {null.q95:.3f}（p={p:.3f}）："
            f"样本内最优参数在样本外系统性落到后半段，优化是负贡献"
        )
    if not math.isfinite(p) or p >= alpha:
        return "inconclusive", (
            f"PBO={pbo:.3f}，零分布下单侧 p={p:.3f} ≥ {alpha}："
            f"与「选参毫无信息」不可区分（零分布均值 {null.mean:.3f}）"
        )
    if pbo <= 0.25:
        return "pass", f"PBO={pbo:.3f}，p={p:.3f} < {alpha}，显著优于零分布"
    return "inconclusive", (
        f"PBO={pbo:.3f} 显著低于零分布(p={p:.3f})但绝对水平仍高于 0.25"
    )


def run_audit(
    runner: EngineRunner,
    settings: Sequence[dict],
    start: datetime,
    end: datetime,
    baseline_setting: dict | None = None,
    train_bars: int = 252,
    test_bars: int = 63,
    target_name: str = "sharpe_ratio",
    selector: Selector = argmax_selector,
    annual_days: int = 252,
    alpha: float = 0.05,
    min_block_obs: int = 40,
    n_offsets: int = 8,
    n_null_sims: int = 200,
    min_round_trips_per_side: int = 20,
    with_pbo: bool = True,
    statistics_func: Callable[[DataFrame], dict] | None = None,
    capital: float | None = None,
    seed: int = 20260725,
) -> AuditResult:
    """跑完整审查：可行性规划 → Walk-Forward → CSCV/PBO → 三态裁决。

    顺序是有意的：**先规划再测量**。可行性规划用全样本的真实回合数决定
    "逐折效率这个指标今天能不能报"，避免事后看到难看的数字再找理由。

    runner              `EngineRunner`（生产必须设 warmup_bars ≥ 策略最长窗口 ×2）
    settings            候选参数网格
    baseline_setting    对照参数（教科书默认值）。强烈建议给 ——
                        "优化打不打得过不优化"是最有信息量的单条判据。
    min_round_trips_per_side  样本资格线：CSCV 每侧的完整回合数下限。
                        低于它 → 结论最多是"无法裁决"，不会给 GO。
    with_pbo=False      只跑 Walk-Forward（CSCV 需要把整个网格在全样本上再跑一遍，
                        网格大时耗时翻倍）。
    statistics_func     daily_df -> statistics dict。runner 不是 `EngineRunner` 时必传
                        （测试用假 runner 就走这条路）。
    """
    if not settings:
        raise ValueError("settings 为空")

    if statistics_func is None:
        if not isinstance(runner, EngineRunner):
            raise ValueError("runner 不是 EngineRunner 时必须显式传入 statistics_func")
        statistics_func = runner.statistics
    if capital is None:
        capital = float(runner.capital)

    dts = [d for d in runner.bar_datetimes() if _within(d, start, end)]
    if len(dts) < train_bars + test_bars:
        raise ValueError(
            f"区间内仅 {len(dts)} 根 K 线，切不出 train={train_bars}+test={test_bars} 的一折"
        )

    # ── 全样本跑一次基准参数，拿真实回合数做规划 ──
    probe_setting = dict(baseline_setting) if baseline_setting else dict(settings[0])
    probe_df = runner(probe_setting, dts[0], dts[-1])
    _, probe_closed = count_round_trips(probe_df)
    holding = mean_holding_days(probe_df)
    plan = plan_walk_forward(
        n_bars=len(dts),
        round_trips_total=probe_closed,
        train_bars=train_bars,
        test_bars=test_bars,
    )

    # ── Walk-Forward ──
    splits = make_walk_forward_splits(dts, train_bars=train_bars, test_bars=test_bars)
    wf = run_walk_forward(
        runner, settings, splits,
        target_name=target_name, selector=selector, baseline_setting=baseline_setting,
        statistics_func=statistics_func,
        annual_days=annual_days, alpha=alpha, capital=capital, seed=seed,
    )
    activity = fold_activity(wf.folds)
    eff_med, n_traded = effective_efficiency(activity)

    # ── CSCV / PBO ──
    study: PBOStudy | None = None
    if with_pbo:
        study = pbo_from_settings(
            runner, settings, dts[0], dts[-1], capital=capital,
            min_block_obs=min_block_obs, annual_days=annual_days,
            n_offsets=n_offsets, n_null_sims=n_null_sims, seed=seed,
            # 让 PBO 段内的样本诊断与下面【2】段用同一把尺子（仓位序列数回合），
            # 否则同一份报告里会印出两个互相矛盾的回合数（fills/2 约为其两倍）。
            round_trip_counter=lambda df: count_round_trips(df)[1],
        )

    # 样本诊断一律用【仓位序列数出来的完整回合数】，不用 fills/2。
    # `pbo_from_settings` 内部那份诊断走的是 fills/2，对金字塔加仓策略会高估一倍以上
    # （海龟一个回合最多 4 买 1 卖 = 5 笔成交 → fills/2 报 2.5 个回合，实际 1 个），
    # 而回合数正是"这个样本够不够格"的唯一分母，宁可用更严的那个。
    diagnosis = sample_size_diagnosis(
        n_obs=len(dts),
        n_blocks=study.result.n_blocks if study is not None else 14,
        annual_days=annual_days,
        trades_total=probe_closed,
        median_holding_days=holding if math.isfinite(holding) else None,
    )

    # ── 裁决 ──
    reasons: list[str] = []
    blockers: list[str] = []          # 正面反证 → NO-GO
    unresolved: list[str] = []        # 功效不足 → 无法裁决

    sig = wf.significance
    if math.isfinite(sig.p_block_bootstrap) and sig.p_block_bootstrap < alpha and sig.sharpe < 0:
        blockers.append(
            f"拼接样本外 Sharpe={sig.sharpe:.2f} 显著为负（block-bootstrap p={sig.p_block_bootstrap:.3f}）"
        )
    elif not (
        sig.sharpe > 0
        and math.isfinite(sig.p_block_bootstrap)
        and sig.p_block_bootstrap < alpha
    ):
        unresolved.append(
            f"拼接样本外 Sharpe={sig.sharpe:+.2f}±{sig.sharpe_se:.2f}(SE)，"
            f"block-bootstrap p={sig.p_block_bootstrap:.3f} ≥ {alpha}：与 0 不可区分"
        )
    else:
        reasons.append(
            f"拼接样本外 Sharpe={sig.sharpe:+.2f}，p={sig.p_block_bootstrap:.3f} < {alpha}"
        )

    if wf.baseline_statistics is not None:
        opt_sharpe = float(wf.oos_statistics.get("sharpe_ratio", 0.0) or 0.0)
        base_sharpe = float(wf.baseline_statistics.get("sharpe_ratio", 0.0) or 0.0)
        if opt_sharpe < base_sharpe:
            blockers.append(
                f"优化参数样本外 Sharpe {opt_sharpe:+.2f} < 默认参数 {base_sharpe:+.2f}："
                f"优化这个动作本身是负贡献"
            )
        else:
            reasons.append(
                f"优化参数样本外 Sharpe {opt_sharpe:+.2f} ≥ 默认参数 {base_sharpe:+.2f}"
            )
    else:
        unresolved.append("未提供 baseline_setting：无法判断优化相对不优化是否有增量")

    if plan.per_fold_efficiency_usable:
        if math.isfinite(eff_med) and eff_med >= 0.5:
            reasons.append(f"WF 效率中位数（仅计有交易的 {n_traded} 折）={eff_med:.2f} ≥ 0.5")
        else:
            blockers.append(
                f"WF 效率中位数（仅计有交易的 {n_traded} 折）={eff_med:.2f} < 0.5：样本外严重衰减"
            )
    else:
        unresolved.append(
            f"逐折效率不可报：{plan.n_folds} 折 / 每折期望回合数 "
            f"{plan.expected_round_trips_per_fold:.1f}（{n_traded}/{len(activity)} 折真的交易过）"
        )

    # 参数稳定性与逐折效率共用同一个前提：折本身要够厚。
    # 折里只有 1 个回合时，"每折选出的参数都不一样"是样本不足的症状，不是过拟合的证据 ——
    # 此时报 NO-GO 等于用无信息的数据给策略定罪，应归入"无法裁决"。
    if wf.parameter_stability >= 0.5:
        reasons.append(f"参数稳定性 {wf.parameter_stability:.2f} ≥ 0.5")
    elif plan.per_fold_efficiency_usable:
        blockers.append(f"参数稳定性 {wf.parameter_stability:.2f} < 0.5：每折都在换参数")
    else:
        unresolved.append(
            f"参数稳定性 {wf.parameter_stability:.2f} < 0.5，但折太薄"
            f"（每折期望 {plan.expected_round_trips_per_fold:.1f} 个回合）："
            f"无法区分「参数在拟合噪声」与「样本不足以选参」"
        )

    if study is not None:
        state, why = _pbo_state(study, alpha)
        if state == "harmful":
            blockers.append(why)
        elif state == "inconclusive":
            unresolved.append(why)
        else:
            reasons.append(why)
        if study.stability.get("pbo_spread", 0.0) > 0.15:
            unresolved.append(
                f"PBO 随块网格相位漂移 {study.stability['pbo_spread']:.3f} > 0.15："
                f"点估计本身不稳，区间 "
                f"[{study.stability['pbo_min']:.3f}, {study.stability['pbo_max']:.3f}]"
            )
    else:
        unresolved.append("未跑 CSCV/PBO（with_pbo=False）")

    rt_per_side = (diagnosis.trades_per_side or 0.0)
    if rt_per_side < min_round_trips_per_side:
        unresolved.append(
            f"CSCV 每侧仅约 {rt_per_side:.0f} 个完整回合（资格线 {min_round_trips_per_side}）："
            f"最小可检出 Sharpe ≈ {diagnosis.min_detectable_sharpe:.2f}，"
            f"真实 Sharpe 低于它的策略在本样本上无法被证明有效"
        )

    if blockers:
        decision = NO_GO
    elif unresolved:
        decision = UNDECIDABLE
    else:
        decision = GO

    return AuditResult(
        plan=plan, diagnosis=diagnosis, wf=wf, activity=activity,
        eff_median_traded=eff_med, n_traded_folds=n_traded, pbo=study,
        decision=decision, reasons=reasons, blockers=blockers + unresolved,
    )


def _within(moment: datetime, start: datetime, end: datetime) -> bool:
    """区间判定，先剥掉时区 —— 数据库返回 tz-aware，调用方通常传裸 datetime。"""
    naive = moment.replace(tzinfo=None) if moment.tzinfo is not None else moment
    lo = start.replace(tzinfo=None) if start.tzinfo is not None else start
    hi = end.replace(tzinfo=None) if end.tzinfo is not None else end
    return lo <= naive <= hi.replace(hour=23, minute=59, second=59)


# ══════════════════════════════════════════════════════════════════════
# 4. 报告
# ══════════════════════════════════════════════════════════════════════

def format_audit(audit: AuditResult) -> str:
    """人读的完整审查报告。"""
    p = audit.plan
    lines = [
        "═" * 72,
        "样本外准入审查",
        "═" * 72,
        "",
        "【0】切分可行性（先规划，后测量）",
        f"  样本 {p.n_bars} 根 K 线，基准参数全样本完成 {p.round_trips_total} 个完整回合"
        f" → 平均 {p.bars_per_round_trip:.0f} 根/回合",
        f"  train={p.train_bars} / test={p.test_bars} → {p.n_folds} 折，"
        f"每折期望回合数 {p.expected_round_trips_per_fold:.1f}",
        f"  逐折效率可否作为判据: {'可' if p.per_fold_efficiency_usable else '否'}",
    ]
    for note in p.notes:
        lines.append(f"  · {note}")

    lines += ["", "【1】逐折成交活跃度（效率指标的分母）"]
    lines.append(f"  {'折':>3} {'成交笔':>7} {'开仓':>5} {'平仓':>5} {'IS年化':>9} {'OOS年化':>9} {'效率':>8}")
    for a in audit.activity:
        flag = "" if a.traded else "   ← 零交易折，效率无意义"
        lines.append(
            f"  {a.fold_index:>3} {a.fills:>7} {a.round_trips_opened:>5} "
            f"{a.round_trips_closed:>5} {a.is_annual_return:>8.1f}% "
            f"{a.oos_annual_return:>8.1f}% {a.efficiency:>8.2f}{flag}"
        )
    lines.append(
        f"  效率中位数：全部折 {audit.wf.efficiency_median:.2f} → "
        f"仅计有交易的 {audit.n_traded_folds} 折 {audit.eff_median_traded:.2f}"
    )

    lines += ["", format_walk_forward_report(audit.wf)]
    if audit.pbo is not None:
        lines += ["", format_pbo_report(audit.pbo)]

    lines += ["", "【2】样本资格诊断（回合数按仓位序列精确计数，非 fills/2）"]
    for note in audit.diagnosis.notes:
        lines.append(f"  · {note}")

    lines += ["", "═" * 72, f"裁决：{audit.decision}", ""]
    if audit.reasons:
        lines.append("  通过的关卡：")
        lines += [f"    ✓ {r}" for r in audit.reasons]
    if audit.blockers:
        lines.append("  未通过 / 无法裁决：")
        lines += [f"    ✗ {b}" for b in audit.blockers]
    lines += [
        "",
        "  提醒：本审查只能否决，不能背书。全部通过仅意味着「没有被这几关拦下」，",
        "  不构成「样本外会赚钱」的证据。",
        "═" * 72,
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# 5. CLI —— 直接对 LongOnlyTurtleStrategy 跑
# ══════════════════════════════════════════════════════════════════════

def load_strategy_class(path: str, class_name: str) -> type:
    """按文件路径加载策略类，并把它所在目录**及其父目录**加进 sys.path。

    本项目的策略文件用 `import strategy_state` 这样的裸 import，而
    `strategy_state.py` 位于 `strategies/` 的**父目录** `vnpy_app/`
    （运行时由 `run.py` 从 vnpy_app 启动，所以那才是它的 sys.path[0]）。
    只加同目录会 ModuleNotFoundError —— 两级都加才等价于生产环境的导入上下文。
    """
    import os

    directory = os.path.dirname(os.path.abspath(path))
    for candidate in (directory, os.path.dirname(directory)):
        if candidate and candidate not in sys.path:
            sys.path.insert(0, candidate)

    spec = importlib.util.spec_from_file_location("_audit_strategy_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从 {path} 加载策略模块")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_audit_strategy_module"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, class_name):
        raise ImportError(f"{path} 里没有 {class_name}")
    obj = getattr(module, class_name)
    if not isinstance(obj, type):
        raise ImportError(f"{path} 里的 {class_name} 不是类（拿到 {type(obj).__name__}）")
    return obj


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。默认值就是本项目 700.SEHK / LongOnlyTurtleStrategy 那套。"""
    from vnpy.trader.constant import Interval
    from vnpy.trader.optimize import OptimizationSetting

    parser = argparse.ArgumentParser(description="Walk-Forward + CSCV/PBO 样本外准入审查")
    parser.add_argument(
        "--strategy-path",
        default="/Users/flink/tradingview/vnpy_app/strategies/long_only_turtle_strategy.py",
    )
    parser.add_argument("--strategy-class", default="LongOnlyTurtleStrategy")
    parser.add_argument("--vt-symbol", default="700.SEHK")
    parser.add_argument("--start", default="2024-01-01", help="分析区间起点 YYYY-MM-DD")
    parser.add_argument("--end", default="2026-07-23")
    parser.add_argument(
        "--data-start", default="2023-07-20",
        help="数据加载起点，必须早于 --start 以提供 warmup K 线",
    )
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--rate", type=float, default=0.0011, help="港股综合费率(单边)")
    parser.add_argument("--slippage", type=float, default=0.2)
    parser.add_argument("--pricetick", type=float, default=0.2)
    parser.add_argument("--board-lot", type=int, default=100)
    parser.add_argument("--warmup-bars", type=int, default=120)
    parser.add_argument("--train-bars", type=int, default=252)
    parser.add_argument("--test-bars", type=int, default=63)
    parser.add_argument("--target", default="sharpe_ratio")
    parser.add_argument("--null-sims", type=int, default=200)
    parser.add_argument("--no-pbo", action="store_true")
    args = parser.parse_args(argv)

    strategy_class = load_strategy_class(args.strategy_path, args.strategy_class)
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    data_start = datetime.strptime(args.data_start, "%Y-%m-%d")

    runner = EngineRunner(
        strategy_class=strategy_class,
        vt_symbol=args.vt_symbol,
        interval=Interval.DAILY,
        rate=args.rate,
        slippage=args.slippage,
        size=1,
        pricetick=args.pricetick,
        capital=int(args.capital),
        start=data_start,
        end=end,
        annual_days=252,
        warmup_bars=args.warmup_bars,
    )

    opt = OptimizationSetting()
    opt.add_parameter("entry_window", 10, 55, 5)
    opt.add_parameter("exit_window", 5, 20, 5)
    opt.add_parameter("atr_stop", 1.5, 3.0, 0.5)
    settings = opt.generate_settings()
    fixed = {"board_lot": args.board_lot, "trading_capital": float(args.capital)}
    for s in settings:
        s.update(fixed)

    baseline = {"entry_window": 20, "exit_window": 10, "atr_stop": 2.0, **fixed}

    audit = run_audit(
        runner, settings, start, end,
        baseline_setting=baseline,
        train_bars=args.train_bars, test_bars=args.test_bars,
        target_name=args.target, n_null_sims=args.null_sims,
        with_pbo=not args.no_pbo,
    )
    print(audit.text())
    return 0 if audit.decision == GO else 1


if __name__ == "__main__":
    raise SystemExit(main())
