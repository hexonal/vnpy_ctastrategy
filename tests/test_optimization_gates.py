"""寻优多重比较闸（DSR + PBO）的操作特性与接入验证。

━━━ 本文件要证明的一件事 ━━━

一道统计闸的价值不在"它算得出一个数"，而在**它能不能把噪声和信号分开**。
所以本文件的核心不是断言某个数等于某个值，而是**负对照 / 正对照的对拍**：

    负对照  ——  40 组纯噪声参数（零真实 edge，列间等相关 ρ=0.7，模拟真实参数网格
                里相邻参数持仓高度重合）。闸必须拒绝：DSR 低、PBO ≈ 0.5。
    正对照  ——  同一批噪声，只把第 0 列加上一个真实的正漂移。闸必须放行。

**若两者的 DSR / PBO 分不开，这道闸就是假的** —— 那种情况下本文件会直接失败，
而不是把一个"看起来通过了"的结果糊过去。实测（见 test_gates_separate_signal_
from_noise 的断言）：

    纯噪声    DSR 0.570   PBO 0.536（零分布 p=0.66，与"完全没有选参能力"无法区分）
    真信号    DSR 0.981   PBO 0.000（零分布 p=0.016）

操作特性（test_negative_control_false_positive_rate，40 次独立噪声网格）：

    DSR 判显著        0/40   （最大 DSR 0.880 < 0.95）
    PBO ≤ 0.25        1/40
    两闸同时通过      0/40

对照的功效（本文件不断言，仅记录，实测 40 次重复）：真实年化夏普 2.0/2.5/3.0 时
两闸同时通过 18/30/35 次。**DSR 在 N=40 组试验下要求真实夏普 ≈3 才有 87% 功效**
—— 这不是实现缺陷，是"扫了 40 组参数"这件事本身的代价，也正是这道闸要传达的信息。
"""

from __future__ import annotations

import math
from datetime import date as Date
from datetime import datetime, timedelta

import numpy as np
import pytest
from pandas import DataFrame
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData

from vnpy_ctastrategy import backtesting as backtesting_module
from vnpy_ctastrategy.backtesting import (
    BacktestingEngine,
    evaluate,
    get_target_value,
    wrap_evaluate,
)
from vnpy_ctastrategy.deflated_sharpe import deflate_optimization
from vnpy_ctastrategy.optimization_gates import (
    OptimizationGateConfig,
    OptimizationGateReport,
    OptimizationResults,
    returns_matrix_from_payloads,
    run_optimization_gates,
)
from vnpy_ctastrategy.overfitting import (
    annualised_sharpe,
    daily_log_returns,
    pbo_from_matrix,
    pbo_from_settings,
    returns_matrix,
)
from vnpy_ctastrategy.template import CtaTemplate

CAPITAL = 1_000_000.0
ANNUAL_DAYS = 252
N_OBS = 600
N_CONFIGS = 40
GRID_CORRELATION = 0.7          # 相邻参数持仓重合 → 列间高相关，真实网格的常态


# ══════════════════════════════════════════════════════════════════════
# 工具：把一张 (T, N) 逐日盈亏表做成寻优返回值的形状
# ══════════════════════════════════════════════════════════════════════

def _days(n: int) -> list[Date]:
    base = Date(2023, 1, 2)
    return [base + timedelta(days=i) for i in range(n)]


def _statistics(pnl_column: np.ndarray, n_obs: int, trade_count: int = 40) -> dict:
    """复刻 `calculate_statistics` 里本闸会用到的那几个键（口径逐字一致）。"""
    returns = daily_log_returns(pnl_column, CAPITAL)
    std = float(np.std(returns))
    skew = float(np.mean((returns - returns.mean()) ** 3) / std**3) if std else 0.0
    kurt = float(np.mean((returns - returns.mean()) ** 4) / std**4) if std else 3.0
    return {
        "total_days": n_obs,
        "total_trade_count": trade_count,
        "sharpe_ratio": annualised_sharpe(returns, ANNUAL_DAYS),
        "return_drawdown_ratio": float(np.sum(pnl_column) / CAPITAL),
        "sharpe_skew": skew,
        "sharpe_kurtosis": kurt,
    }


def make_optimization_output(
    pnl: np.ndarray,
    target_name: str = "sharpe_ratio",
    trade_count: int = 40,
) -> tuple[list[tuple], list[tuple]]:
    """(T, N) 逐日盈亏 → (results, payloads)，与 vnpy 寻优返回值同形状且按目标降序。"""
    n_obs, n_cols = pnl.shape
    days = _days(n_obs)
    rows: list[tuple] = []
    for j in range(n_cols):
        column = pnl[:, j].copy()
        stats = _statistics(column, n_obs, trade_count)
        rows.append(({"idx": j}, stats[target_name], stats, (days, column)))
    rows.sort(key=lambda row: row[1], reverse=True)
    return [row[:3] for row in rows], [row[3] for row in rows]


def noise_pnl(seed: int, n_obs: int = N_OBS, n_cols: int = N_CONFIGS) -> np.ndarray:
    """零真实 edge 的参数网格：等相关高斯因子模型，日波动 1% 本金。"""
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal((n_obs, 1))
    idio = rng.standard_normal((n_obs, n_cols))
    mixed = (
        math.sqrt(GRID_CORRELATION) * factor
        + math.sqrt(1.0 - GRID_CORRELATION) * idio
    )
    return mixed * (CAPITAL * 0.01)


def with_signal(pnl: np.ndarray, annual_sharpe: float) -> np.ndarray:
    """给第 0 列注入一个真实的正漂移，其余列保持纯噪声。"""
    out = pnl.copy()
    out[:, 0] += annual_sharpe / math.sqrt(ANNUAL_DAYS) * (CAPITAL * 0.01)
    return out


def gate(
    pnl: np.ndarray,
    target_name: str = "sharpe_ratio",
    config: OptimizationGateConfig | None = None,
) -> OptimizationGateReport:
    results, payloads = make_optimization_output(pnl, target_name=target_name)
    return run_optimization_gates(
        results,
        target_name=target_name,
        annual_days=ANNUAL_DAYS,
        capital=CAPITAL,
        payloads=payloads,
        config=config or FAST_CONFIG,
    )


#: 测试用配置：块数固定（不依赖样本长度的自动推荐）、相位少、零分布少。
#: 生产默认 n_null_sims=200，这里降到 60 只为控制单测时长，不改变判据。
FAST_CONFIG = OptimizationGateConfig(n_blocks=10, n_offsets=4, n_null_sims=60)
#: 不需要零分布 p 值的断言用这个，快一个数量级。
NO_NULL_CONFIG = OptimizationGateConfig(n_blocks=10, n_offsets=2, n_null_sims=0)


# ══════════════════════════════════════════════════════════════════════
# 1. 负对照：纯噪音网格必须被拒
# ══════════════════════════════════════════════════════════════════════

def test_negative_control_noise_grid_is_rejected() -> None:
    """40 组纯噪声参数，挑夏普最高的一组：DSR 必须低，PBO 必须高。

    这一组数据里**确定没有任何 edge**（所有列同分布、零均值）。
    第一名的年化夏普仍有 0.83 —— 那就是"扫了 40 组"白送的运气分量，
    单条曲线的 t 检验会把它当成一个正常的观测值，DSR 与 PBO 是唯一能拆穿它的两道闸。
    """
    report = gate(noise_pnl(seed=20260726))

    assert report.dsr is not None
    assert report.dsr["trustworthy"] is False
    assert report.dsr_significant is False
    assert report.dsr_value < 0.95

    assert report.pbo is not None
    # 纯噪声下 PBO 的期望是 0.5；这里只要求它没有掉到"稳健"区
    assert report.pbo.result.pbo > 0.25
    assert "稳健" not in report.pbo.result.verdict
    # 零分布检验：噪声的 PBO 必须与"完全没有选参能力"无法区分
    assert report.pbo.p_value > 0.05

    assert report.passed is False


def test_negative_control_false_positive_rate() -> None:
    """操作特性：40 次独立的纯噪声网格，两闸同时放行的次数必须 ≤ 2（5%）。

    单次抽样可能碰巧好看，所以判定一道检验有没有用，只能看重复实验下的假阳性率。
    实测 0/40 通过、DSR 单独 0/40 判显著、PBO ≤0.25 单独 1/40。
    """
    dsr_hits = 0
    pbo_hits = 0
    passed = 0
    dsr_values: list[float] = []

    for seed in range(40):
        report = gate(noise_pnl(seed=1000 + seed), config=NO_NULL_CONFIG)
        assert report.pbo is not None
        dsr_values.append(report.dsr_value)
        dsr_hits += bool(report.dsr_significant)
        pbo_hits += report.pbo.result.pbo <= 0.25
        passed += report.passed

    assert dsr_hits <= 2, f"DSR 在纯噪声上判显著 {dsr_hits}/40，假阳性率超 5%"
    assert pbo_hits <= 4, f"PBO ≤0.25 在纯噪声上出现 {pbo_hits}/40"
    assert passed <= 2, f"两闸同时放行纯噪声 {passed}/40，闸失效"
    assert max(dsr_values) < 0.95


# ══════════════════════════════════════════════════════════════════════
# 2. 正对照 + 判别力：闸不能只会说"不"
# ══════════════════════════════════════════════════════════════════════

def test_positive_control_real_signal_passes() -> None:
    """同一批噪声，第 0 列注入真实年化夏普 2.5 的漂移：两闸都必须放行。

    只会拒绝的闸和永远返回 False 的函数没有区别。正对照证明它拒绝的是噪声本身，
    而不是"任何东西都拒绝"。
    """
    report = gate(with_signal(noise_pnl(seed=20260726), annual_sharpe=2.5))

    assert report.dsr is not None
    assert report.dsr["trustworthy"] is True
    assert report.dsr_significant is True
    assert report.dsr_value >= 0.95

    assert report.pbo is not None
    assert report.pbo.result.pbo <= 0.10
    assert report.pbo.p_value <= 0.05
    assert report.passed is True


def test_gates_separate_signal_from_noise() -> None:
    """判别力主张：同一随机种子下，信号组的两个指标都必须显著优于噪声组。

    **这是本文件最关键的一条断言。** 若它失败，说明 DSR / PBO 在本项目的样本规模上
    根本分不出噪声和信号，那么整套闸就是装饰品 —— 这种情况必须让测试红掉，
    而不是留一个"两边都算出了数"的假通过。
    """
    base = noise_pnl(seed=20260726)
    noise = gate(base, config=NO_NULL_CONFIG)
    signal = gate(with_signal(base, annual_sharpe=2.5), config=NO_NULL_CONFIG)

    assert noise.pbo is not None
    assert signal.pbo is not None

    # DSR：信号组必须跨过阈值，噪声组必须跨不过，且差距不能是数值噪音
    assert signal.dsr_value - noise.dsr_value > 0.30
    assert noise.dsr_value < 0.95 <= signal.dsr_value

    # PBO：信号组必须显著低于噪声组
    assert noise.pbo.result.pbo - signal.pbo.result.pbo > 0.30
    assert signal.pbo.result.pbo <= 0.10 < noise.pbo.result.pbo

    assert signal.passed is True
    assert noise.passed is False


# ══════════════════════════════════════════════════════════════════════
# 3. 缺证据不是证据：算不出来的闸必须说"没算"，不能默认放行
# ══════════════════════════════════════════════════════════════════════

def test_missing_payloads_reports_instead_of_guessing() -> None:
    """没收集逐日收益 → PBO 记 None + 备注，绝不用任何替代量硬凑一个 PBO。"""
    results, _ = make_optimization_output(noise_pnl(seed=3))
    report = run_optimization_gates(
        results, target_name="sharpe_ratio", annual_days=ANNUAL_DAYS,
        capital=CAPITAL, payloads=None, config=NO_NULL_CONFIG,
    )

    assert report.pbo is None
    assert report.matrix_shape is None
    assert report.passed is False
    assert any("PBO 无法计算" in note for note in report.notes)
    assert "未计算" in report.pbo_verdict


def test_non_sharpe_target_does_not_fake_a_headline_dsr() -> None:
    """寻优目标不是夏普族时，headline DSR 必须缺席并说明原因，而不是照样出数。

    `deflate_optimization_results` 对非夏普目标显式 raise（它会把 row[1] 当年化夏普，
    喂 return_drawdown_ratio 会算出一个没有统计含义却看起来可信的数）。
    闸接住这个 raise，如实记 None —— 但**详版 DSR 仍然要算**，只是改打在
    "网格里夏普最大的那组"上，因为 SR* 是最大值序统计，前提就是"被检验的是最大的那个"。
    """
    report = gate(noise_pnl(seed=5), target_name="return_drawdown_ratio")

    assert report.dsr is None
    assert any("不属夏普族" in note for note in report.notes)
    # 详版仍在，且明确标注打在了另一组上
    assert report.dsr_detail is not None
    assert any("夏普最大的那组" in note for note in report.notes)
    assert report.dsr_significant is False


def test_blown_up_top_group_does_not_crash_the_run() -> None:
    """头名那组爆仓（statistics 为空 dict）时，闸降级留痕，绝不炸掉已跑完的寻优。"""
    results, payloads = make_optimization_output(noise_pnl(seed=7))
    results = [(results[0][0], results[0][1], {}), *results[1:]]

    report = run_optimization_gates(
        results, target_name="sharpe_ratio", annual_days=ANNUAL_DAYS,
        capital=CAPITAL, payloads=payloads, config=NO_NULL_CONFIG,
    )

    assert report.dsr is None
    assert any("total_days" in note for note in report.notes)
    # 详版改打在还有夏普的那些组里的最大者，PBO 不受影响
    assert report.dsr_detail is not None
    assert report.pbo is not None
    assert report.passed is False


def test_all_groups_blown_up_reports_both_gates_missing() -> None:
    """全网格爆仓：两道闸都记 None + 备注，`passed` 仍是 False。"""
    results, payloads = make_optimization_output(noise_pnl(seed=9))
    results = [(setting, target, {}) for setting, target, _ in results]

    report = run_optimization_gates(
        results, target_name="sharpe_ratio", annual_days=ANNUAL_DAYS,
        capital=CAPITAL, payloads=payloads, config=NO_NULL_CONFIG,
    )

    assert report.dsr is None
    assert report.dsr_detail is None
    assert report.dsr_significant is None
    assert "未计算" in report.dsr_verdict
    assert report.passed is False
    assert any("sharpe_ratio" in note for note in report.notes)


def test_empty_results_is_reported_not_raised() -> None:
    report = run_optimization_gates(
        [], target_name="sharpe_ratio", annual_days=ANNUAL_DAYS,
        capital=CAPITAL, payloads=[], config=NO_NULL_CONFIG,
    )
    assert report.dsr is None
    assert report.dsr_detail is None
    assert report.pbo is None
    assert report.passed is False
    assert any("寻优结果为空" in note for note in report.notes)


def test_payload_length_mismatch_is_reported() -> None:
    results, payloads = make_optimization_output(noise_pnl(seed=11))
    report = run_optimization_gates(
        results, target_name="sharpe_ratio", annual_days=ANNUAL_DAYS,
        capital=CAPITAL, payloads=payloads[:-1], config=NO_NULL_CONFIG,
    )
    assert report.pbo is None
    assert any("不一致" in note for note in report.notes)


def test_ga_source_records_the_optimistic_bias() -> None:
    """GA 的 DSR 偏乐观是结构性的，闸无法自动修正，但必须写在报告里。"""
    results, _ = make_optimization_output(noise_pnl(seed=13))
    report = run_optimization_gates(
        results, target_name="sharpe_ratio", annual_days=ANNUAL_DAYS,
        capital=CAPITAL, payloads=None, config=NO_NULL_CONFIG, source="ga",
    )
    assert report.source == "ga"
    assert any("偏乐观" in note for note in report.notes)


# ══════════════════════════════════════════════════════════════════════
# 4. 向后兼容：接闸不许改变寻优返回值的形状
# ══════════════════════════════════════════════════════════════════════

def test_optimization_results_is_a_plain_list_of_triples() -> None:
    """`.gates` 是【附加】属性；列表本体必须与接闸前逐字相同。"""
    results, _ = make_optimization_output(noise_pnl(seed=17, n_cols=5))
    wrapped = OptimizationResults(results, gates=None)

    assert isinstance(wrapped, list)
    assert len(wrapped) == len(results)
    assert list(wrapped) == list(results)

    # 既有调用方的四种用法
    for setting, target, statistics in wrapped:          # 三元组解包
        assert isinstance(setting, dict)
        assert isinstance(target, float)
        assert "sharpe_ratio" in statistics
    assert get_target_value(wrapped[0]) == wrapped[0][1]  # 排序键
    assert wrapped[0][2]["total_days"] == N_OBS           # statistics 下标
    assert deflate_optimization(wrapped, annual_days=ANNUAL_DAYS).n_trials == 5

    # 目标值仍是降序
    targets = [row[1] for row in wrapped]
    assert targets == sorted(targets, reverse=True)


def test_gate_config_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError, match="confidence"):
        OptimizationGateConfig(confidence=1.5)
    with pytest.raises(ValueError, match="n_blocks"):
        OptimizationGateConfig(n_blocks=7)
    with pytest.raises(ValueError, match="n_offsets"):
        OptimizationGateConfig(n_offsets=0)
    with pytest.raises(ValueError, match="n_null_sims"):
        OptimizationGateConfig(n_null_sims=-1)


def test_report_serialises_to_scalars_and_text() -> None:
    report = gate(noise_pnl(seed=19), config=NO_NULL_CONFIG)
    payload = report.as_dict()

    assert payload["target_name"] == "sharpe_ratio"
    assert payload["passed"] is False
    assert payload["dsr"] is not None
    assert report.pbo is not None
    assert payload["pbo"]["pbo"] == report.pbo.result.pbo

    text = report.text()
    assert "寻优多重比较闸" in text
    assert "DSR" in text
    assert "PBO" in text


# ══════════════════════════════════════════════════════════════════════
# 5. 口径一致：新入口不许另起一套算法
# ══════════════════════════════════════════════════════════════════════

def test_payload_matrix_matches_overfitting_returns_matrix() -> None:
    """payload → 矩阵 走的必须是 `overfitting.returns_matrix` 那一套对齐与对数收益。"""
    pnl = noise_pnl(seed=23, n_obs=120, n_cols=6)
    _, payloads = make_optimization_output(pnl)

    frames = [
        DataFrame({"net_pnl": np.asarray(values)}, index=list(index))
        for index, values in payloads
    ]
    expected, expected_dropped = returns_matrix(frames, CAPITAL)
    actual, actual_dropped = returns_matrix_from_payloads(payloads, CAPITAL)

    assert actual_dropped == expected_dropped
    np.testing.assert_allclose(actual, expected)


def test_empty_payload_column_is_dropped_and_reported() -> None:
    pnl = noise_pnl(seed=29, n_obs=120, n_cols=4)
    _, payloads = make_optimization_output(pnl)
    payloads = [([], np.zeros(0)), *payloads[1:]]

    matrix, dropped = returns_matrix_from_payloads(payloads, CAPITAL)
    assert dropped == [0]
    assert matrix.shape == (120, 3)

    assert returns_matrix_from_payloads([None, *payloads[1:]], CAPITAL)[1] == [0]


def test_payload_with_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError, match="不一致"):
        returns_matrix_from_payloads([(_days(5), np.zeros(4))], CAPITAL)


def test_pbo_from_matrix_matches_pbo_from_settings() -> None:
    """重构守卫：`pbo_from_settings` 现在委托给 `pbo_from_matrix`，两条路必须同结果。"""
    pnl = noise_pnl(seed=31, n_obs=240, n_cols=8)
    dates = [datetime(2023, 1, 2) + timedelta(days=i) for i in range(240)]
    settings = [{"idx": j} for j in range(8)]

    class Runner:
        def __call__(self, setting: dict, start: datetime, end: datetime) -> DataFrame:
            col = setting["idx"]
            return DataFrame(
                {"net_pnl": pnl[:, col], "trade_count": np.full(240, 1.0)},
                index=[d.date() for d in dates],
            )

    kwargs = {
        "annual_days": ANNUAL_DAYS, "n_blocks": 10, "n_offsets": 2, "n_null_sims": 0,
    }
    from_settings = pbo_from_settings(
        Runner(), settings, dates[0], dates[-1], capital=CAPITAL, **kwargs  # type: ignore[arg-type]
    )
    matrix, _ = returns_matrix_from_payloads(
        [([d.date() for d in dates], pnl[:, j]) for j in range(8)], CAPITAL
    )
    from_matrix = pbo_from_matrix(
        matrix, avg_round_trips=240 / 2.0,
        extra_notes=[
            "回合数按 trade_count/2 估算（未传 round_trip_counter）："
            "金字塔加仓策略会被高估，样本诊断偏宽松"
        ],
        **kwargs,  # type: ignore[arg-type]
    )

    assert from_matrix.result.pbo == from_settings.result.pbo
    assert from_matrix.stability == from_settings.stability
    assert from_matrix.diagnosis.trades_total == from_settings.diagnosis.trades_total
    assert from_matrix.notes == from_settings.notes


def test_pbo_from_matrix_leaves_trade_count_blank_when_unknown() -> None:
    """数不出回合数时留空，而不是塞一个猜出来的数进样本诊断。"""
    matrix, _ = returns_matrix_from_payloads(
        [(_days(240), noise_pnl(seed=37, n_obs=240, n_cols=4)[:, j]) for j in range(4)],
        CAPITAL,
    )
    study = pbo_from_matrix(
        matrix, annual_days=ANNUAL_DAYS, n_blocks=10, n_offsets=1, n_null_sims=0
    )
    assert study.diagnosis.trades_total is None


def test_pbo_from_matrix_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError, match="二维"):
        pbo_from_matrix(np.zeros(10))
    with pytest.raises(ValueError, match="为空"):
        pbo_from_matrix(np.zeros((0, 0)))


# ══════════════════════════════════════════════════════════════════════
# 6. 端到端：真引擎、真回测、真参数网格（无数据库、无多进程）
# ══════════════════════════════════════════════════════════════════════

VT_SYMBOL = "NOISE.SEHK"
BACKTEST_START = datetime(2023, 1, 2)


class ChannelBreakout(CtaTemplate):
    """long-only 通道突破。参数网格用的就是这两个窗口。

    刻意不用 `load_bar` 预热（回测引擎的 load_bar 默认查分钟线，本测试只有日线），
    自己维护收盘价列表，行为完全由喂进来的 K 线决定。
    """

    author = "test"
    entry_window = 20
    exit_window = 10
    parameters = ["entry_window", "exit_window"]
    variables = ["pos"]

    def __init__(
        self, cta_engine: object, strategy_name: str, vt_symbol: str, setting: dict
    ) -> None:
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.closes: list[float] = []

    def on_init(self) -> None:
        self.closes = []

    def on_bar(self, bar: BarData) -> None:
        self.cancel_all()
        self.closes.append(bar.close_price)
        warmup = max(int(self.entry_window), int(self.exit_window)) + 1
        if len(self.closes) <= warmup:
            return
        history = self.closes[:-1]
        if self.pos == 0:
            if bar.close_price > max(history[-int(self.entry_window):]):
                self.buy(bar.close_price, 100)
        elif bar.close_price < min(history[-int(self.exit_window):]):
            self.sell(bar.close_price, abs(self.pos))


def random_walk_bars(n: int, seed: int) -> list[BarData]:
    """零漂移随机游走：任何参数组合在其上都没有真实 edge。"""
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.02))
    bars: list[BarData] = []
    moment = BACKTEST_START
    for price in prices:
        while moment.weekday() >= 5:
            moment += timedelta(days=1)
        bars.append(BarData(
            symbol="NOISE", exchange=Exchange.SEHK, datetime=moment,
            interval=Interval.DAILY, open_interest=0.0,
            volume=1000.0, turnover=1000.0 * price,
            open_price=price, high_price=price, low_price=price, close_price=price,
            gateway_name="TEST",
        ))
        moment += timedelta(days=1)
    return bars


@pytest.fixture
def injected_bars(monkeypatch: pytest.MonkeyPatch) -> list[BarData]:
    """把随机游走 K 线注入 `load_bar_data`，并静音引擎输出。

    `evaluate()` 在自己进程里 new 一个 `BacktestingEngine`，走的是模块级的
    `load_bar_data`，所以只要替换模块属性就能完全绕开数据库。
    """
    bars = random_walk_bars(700, seed=20260726)

    def fake_load_bar_data(
        symbol: str, exchange: Exchange, interval: Interval,
        start: datetime, end: datetime,
    ) -> list[BarData]:
        return [bar for bar in bars if start <= bar.datetime <= end]

    monkeypatch.setattr(backtesting_module, "load_bar_data", fake_load_bar_data)
    monkeypatch.setattr(BacktestingEngine, "output", lambda self, msg: None)
    return bars


def _engine(bars: list[BarData], annual_days: int = ANNUAL_DAYS) -> BacktestingEngine:
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=VT_SYMBOL, interval=Interval.DAILY, start=BACKTEST_START,
        end=bars[-1].datetime, rate=0.0, slippage=0.0, size=1, pricetick=0.001,
        capital=int(CAPITAL), annual_days=annual_days,
    )
    engine.add_strategy(ChannelBreakout, {})
    return engine


GRID = [
    {"entry_window": entry, "exit_window": exit_}
    for entry in (10, 15, 20, 25, 30)
    for exit_ in (5, 10, 15, 20, 25)
]


def test_end_to_end_noise_grid_is_rejected(injected_bars: list[BarData]) -> None:
    """真引擎 + 真回测 + 25 组参数网格，跑在零漂移随机游走上：两闸必须拒绝。

    这条走的是与生产完全相同的代码路径（`wrap_evaluate` → `evaluate` →
    `_finish_optimization`），只把 `ProcessPoolExecutor` 换成串行 map ——
    进程池只影响调度，不影响每组参数算出来的东西。
    """
    engine = _engine(injected_bars)
    func = wrap_evaluate(engine, "sharpe_ratio", collect_returns=True)
    raw = [func(setting) for setting in GRID]
    raw.sort(reverse=True, key=get_target_value)

    results = engine._finish_optimization(
        raw, target_name="sharpe_ratio", output=False, gates=True,
        gate_config=FAST_CONFIG, collect_returns=True, source="bf",
    )

    # 返回值形状：仍是 (setting, target, statistics) 三元组的降序列表
    assert isinstance(results, list)
    assert len(results) == len(GRID)
    assert all(len(row) == 3 for row in results)
    targets = [row[1] for row in results]
    assert targets == sorted(targets, reverse=True)

    report = results.gates
    assert report is not None
    assert report.matrix_shape is not None
    assert report.matrix_shape[1] >= 2
    assert report.dsr_significant is False
    assert report.pbo is not None
    assert report.pbo.result.pbo > 0.25
    assert report.passed is False


def test_end_to_end_gates_can_be_switched_off(injected_bars: list[BarData]) -> None:
    engine = _engine(injected_bars)
    func = wrap_evaluate(engine, "sharpe_ratio", collect_returns=False)
    raw = [func(setting) for setting in GRID[:4]]
    raw.sort(reverse=True, key=get_target_value)

    results = engine._finish_optimization(
        raw, target_name="sharpe_ratio", output=False, gates=False,
        gate_config=None, collect_returns=False, source="bf",
    )
    assert results.gates is None
    assert all(len(row) == 3 for row in results)


def test_evaluate_returns_payload_only_when_asked(injected_bars: list[BarData]) -> None:
    """`collect_returns` 是唯一改变 evaluate 返回值形状的开关，默认关。"""
    args = (
        "sharpe_ratio", ChannelBreakout, VT_SYMBOL, Interval.DAILY, BACKTEST_START,
        0.0, 0.0, 1, 0.001, int(CAPITAL), injected_bars[-1].datetime,
        backtesting_module.BacktestingMode.BAR,
    )
    setting = {"entry_window": 20, "exit_window": 10}

    plain = evaluate(*args, setting)
    assert len(plain) == 3

    rich = evaluate(*args, setting, collect_returns=True)
    assert len(rich) == 4
    assert rich[:3] == plain

    index, net_pnl = rich[3]
    assert len(index) == len(net_pnl) == plain[2]["total_days"]
    assert all(isinstance(day, Date) for day in index)


def test_wrap_evaluate_forwards_engine_annual_days(
    injected_bars: list[BarData],
) -> None:
    """回归守卫：`annual_days` 曾在寻优路径上被静默丢弃，子进程一律用 240。

    港股设 247 时，寻优排出来的 sharpe 与单跑 247 的值必须一致；
    否则整个排名、DSR 的年化折算、ewm_sharpe 全建在错的年化基数上。
    """
    hk = wrap_evaluate(_engine(injected_bars, annual_days=247), "sharpe_ratio")
    us = wrap_evaluate(_engine(injected_bars, annual_days=252), "sharpe_ratio")
    setting = {"entry_window": 20, "exit_window": 10}

    hk_stats = hk(setting)[2]
    us_stats = us(setting)[2]

    assert hk_stats["sharpe_ratio"] != us_stats["sharpe_ratio"]
    ratio = us_stats["sharpe_ratio"] / hk_stats["sharpe_ratio"]
    assert ratio == pytest.approx(math.sqrt(252 / 247), rel=1e-9)


def test_wrap_evaluate_binds_new_parameters_by_keyword(
    injected_bars: list[BarData],
) -> None:
    """partial 是位置绑定：新参数必须以关键字传，否则会与 setting 静默错位。"""
    func = wrap_evaluate(_engine(injected_bars), "sharpe_ratio", collect_returns=True)
    assert func.keywords == {                       # type: ignore[attr-defined]
        "risk_free": 0.0,
        "annual_days": ANNUAL_DAYS,
        "half_life": 120,
        "collect_returns": True,
    }
    # 位置参数在 setting 之前截止，setting 由优化器追加
    assert len(func.args) == 12                     # type: ignore[attr-defined]
