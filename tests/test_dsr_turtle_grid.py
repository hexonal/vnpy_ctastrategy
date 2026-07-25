"""DSR 在【真实策略网格】上的操作特性验证（本项目 long-only 海龟场景）。

和 test_deflated_sharpe_core.py 的分工
────────────────────────────────────────────────────────────────────────
core 那份把"一次寻优"模拟成【直接生成 N 条相关的高斯收益序列】，相关性靠
共同因子人工注入。那验证的是公式在理想条件下的操作特性。

本文件不生成收益序列，而是**跑真的策略网格**：在一条纯随机游走价格上跑
long-only 通道突破（海龟族，本项目实盘策略族），用 (entry_lookback,
exit_lookback) 组成参数网格，按"挑夏普最高的那组"选参数，再算 DSR。

这样产生的试验相关性是【回测本身induce的】，而不是人工注入的，并且策略收益
带着真实回测才有的三个毛病，这三个都不满足 Bailey 推导时的 iid 正态假设：

  1. 空仓期收益恒为 0 → 零膨胀，分布远非正态；
  2. 持仓期跨多日 → 收益自相关（Bailey 的 T 个独立观测假设被违反）；
  3. 相邻参数（entry=20 vs entry=22）的持仓几乎重合 → 试验间相关系数极高。

这三条恰恰是"DSR 在我们这儿还成立吗"的真正风险点。若在这种条件下假阳性率
仍 ≤5%，才说明这套检验在本项目能用，而不只是在教科书假设下能用。

判定基准（为什么是"假阳性率"而不是"某个数对不对"）
────────────────────────────────────────────────────────────────────────
统计检验的有效性只能用操作特性验证：喂给它【确定没有 edge】的数据，它必须
以不超过 α 的频率喊"显著"。本项目出现过样本外 −19.5% 而无机制拦截，对应的
就是"纯噪声挑出的最优参数被当成真 edge"——即下面 test_null_* 里那 20%+ 的
naive PSR 假阳性。DSR 把它压回 5% 以内，就是这套东西的全部价值。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vnpy_ctastrategy.deflated_sharpe import (
    DEFAULT_CONFIDENCE,
    DeflatedSharpeResult,
    deflate_optimization,
    deflated_sharpe_ratio,
    return_moments,
)

HK_ANNUAL_DAYS = 247          # 港股年化交易日
SAMPLE_DAYS = 600             # 本项目典型日线样本长度
DAILY_VOL = 0.02              # 日波动 2%，贴近 700.SEHK 量级


# ── 策略网格（long-only 通道突破，向量化）──────────────────────────────

def turtle_grid_sharpes(
    prices: np.ndarray,
    entry_lookbacks: np.ndarray,
    exit_lookbacks: np.ndarray,
    annual_days: int = HK_ANNUAL_DAYS,
) -> tuple[np.ndarray, np.ndarray]:
    """在一条价格序列上跑完整参数网格的 long-only 通道突破。

    规则（海龟简化版，现货 long-only，无杠杆无做空）：
        入场：收盘价 > 前一日为止的 entry_lookback 日最高收盘
        出场：收盘价 < 前一日为止的 exit_lookback 日最低收盘
    t 日持有的仓位吃 t→t+1 的收益，通道用 shift(1) 后的值，无前视。

    返回 (sharpes, strategy_returns)
        sharpes            shape (n_combos,)，年化夏普
        strategy_returns   shape (n_obs-1, n_combos)，每日策略收益
    """
    n = int(prices.size)
    rets = np.diff(prices) / prices[:-1]

    combos = [(int(e), int(x)) for e in entry_lookbacks for x in exit_lookbacks]
    n_combos = len(combos)
    max_lb = int(max(entry_lookbacks.max(), exit_lookbacks.max()))

    # 每个用到的 lookback 只算一次滚动极值
    roll_max: dict[int, np.ndarray] = {}
    roll_min: dict[int, np.ndarray] = {}
    for lb in {int(v) for v in entry_lookbacks} | {int(v) for v in exit_lookbacks}:
        window = np.lib.stride_tricks.sliding_window_view(prices, lb)
        rmax = np.full(n, np.nan)
        rmin = np.full(n, np.nan)
        rmax[lb - 1:] = window.max(axis=1)
        rmin[lb - 1:] = window.min(axis=1)
        roll_max[lb] = rmax
        roll_min[lb] = rmin

    entry_sig = np.zeros((n, n_combos), dtype=bool)
    exit_sig = np.zeros((n, n_combos), dtype=bool)
    for j, (e, x) in enumerate(combos):
        prior_high = np.roll(roll_max[e], 1)
        prior_low = np.roll(roll_min[x], 1)
        prior_high[:1] = np.nan          # roll 把末位绕到首位，手工作废
        prior_low[:1] = np.nan
        entry_sig[:, j] = prices > prior_high
        exit_sig[:, j] = prices < prior_low

    # 逐日推进仓位状态（对 combo 维度向量化，只在时间维循环）
    pos = np.zeros(n_combos, dtype=bool)
    pos_hist = np.zeros((n, n_combos), dtype=bool)
    for t in range(max_lb, n):
        pos = np.where(pos, ~exit_sig[t], entry_sig[t])
        pos_hist[t] = pos

    strat = pos_hist[:-1].astype(float) * rets[:, None]
    mean = strat.mean(axis=0)
    std = strat.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = np.where(std > 0.0, mean / std * math.sqrt(annual_days), 0.0)
    return sharpes, strat


def _grid(n_entry: int, n_exit: int) -> tuple[np.ndarray, np.ndarray]:
    return np.arange(10, 10 + n_entry * 2, 2), np.arange(5, 5 + n_exit * 2, 2)


def _run_once(
    rng: np.random.Generator,
    entry: np.ndarray,
    exits: np.ndarray,
    drift: float = 0.0,
) -> tuple[DeflatedSharpeResult, np.ndarray, np.ndarray, int]:
    """模拟一整轮寻优：造价格 → 跑网格 → 挑最优 → 算 DSR。"""
    rets = rng.standard_normal(SAMPLE_DAYS + 1) * DAILY_VOL + drift
    prices = 100.0 * np.exp(np.cumsum(rets))
    sharpes, strat = turtle_grid_sharpes(prices, entry, exits)

    best = int(np.argmax(sharpes))
    best_returns = strat[:, best]
    moments = return_moments(best_returns)
    result = deflated_sharpe_ratio(
        observed_sharpe=float(sharpes[best]),
        trial_sharpe_std=float(np.std(sharpes, ddof=1)),
        n_trials=int(sharpes.size),
        n_obs=int(best_returns.size),
        skew=moments.skew,
        kurtosis=moments.kurtosis,
        annual_days=HK_ANNUAL_DAYS,
    )
    return result, sharpes, strat, best


# ── 核心：零假设下的假阳性率 ──────────────────────────────────────────

@pytest.mark.parametrize(
    ("n_entry", "n_exit", "reps", "seed"),
    [(10, 10, 300, 20260725), (40, 25, 120, 20260726)],
)
def test_null_random_walk_false_positive_rate(
    n_entry: int, n_exit: int, reps: int, seed: int
) -> None:
    """【核心】纯随机游走上挑最优参数：DSR 假阳性率必须 ≤5%，而 naive PSR 远超。

    价格是零漂移几何随机游走 —— 定义上没有任何 edge。任何判"显著"都是假阳性。
    """
    rng = np.random.default_rng(seed)
    entry, exits = _grid(n_entry, n_exit)

    dsr_hits = 0
    psr_hits = 0
    best_sharpes = []
    for _ in range(reps):
        result, sharpes, _, best = _run_once(rng, entry, exits)
        best_sharpes.append(float(sharpes[best]))
        dsr_hits += int(result.significant)
        psr_hits += int(result.probabilistic_sharpe_ratio >= DEFAULT_CONFIDENCE)

    dsr_fpr = dsr_hits / reps
    psr_fpr = psr_hits / reps

    # 1) DSR 控制住了假阳性（检验有效的充要条件）
    assert dsr_fpr <= 0.06, f"DSR 假阳性率 {dsr_fpr:.1%} 超标，检验失效"

    # 2) 不做 deflation 的 PSR 假阳性率显著更高 —— 证明 deflation 真的在起作用，
    #    而不是"这套东西对谁都说不显著"
    assert psr_fpr >= 0.15, f"naive PSR 假阳性率仅 {psr_fpr:.1%}，对照组不成立"
    assert psr_fpr > dsr_fpr * 3

    # 3) 纯噪声也能挑出漂亮的夏普 —— 这正是 −19.5% 样本外事故的成因
    assert float(np.mean(best_sharpes)) > 0.4


def test_dsr_threshold_sits_above_the_noise_distribution() -> None:
    """门槛位置校验：DSR 门槛必须落在【纯噪声最优夏普分布】的上尾之外。

    这条比"纯噪声不显著"更直接地说明为什么裸夏普不能用于判断：把三个种子共
    600 轮寻优的结果汇总，纯噪声挑出的最优年化夏普 p95 已经到 1.5 附近 ——
    也就是说【看到 1.5 的年化夏普并不能排除"纯运气"】。DSR 门槛约 1.7，正好
    压在这个噪声分布的上尾之外，越过它才谈得上不像运气。

    注意不要把这条误读成"DSR 门槛以下的策略一定没用"：门槛是针对
    "跑了 N 组挑最好"这个动作的，不是对策略本身的判决。
    """
    entry, exits = _grid(10, 10)
    best_sharpes: list[float] = []
    required: list[float] = []
    for seed in (4242, 20260725, 7):
        rng = np.random.default_rng(seed)
        for _ in range(200):
            result, sharpes, _, best = _run_once(rng, entry, exits)
            best_sharpes.append(float(sharpes[best]))
            required.append(result.required_sharpe_annual)

    best_arr = np.asarray(best_sharpes)
    req_arr = np.asarray(required)

    # 1) 纯噪声也能挑出"看起来很能打"的夏普：上尾摸到 1.3+
    noise_p95 = float(np.percentile(best_arr, 95))
    assert noise_p95 >= 1.3, f"噪声最优夏普 p95 仅 {noise_p95:.2f}，网格可能太小"

    # 2) 门槛压在噪声分布上尾之外
    assert float(np.median(req_arr)) > noise_p95

    # 3) 越过门槛的比例 = 假阳性率，仍受控
    assert float(np.mean(best_arr >= req_arr)) <= 0.07


# ── 功效：有真 edge 时能不能认出来 ────────────────────────────────────

def test_power_increases_with_genuine_edge() -> None:
    """功效单调性：漂移越强 → DSR 判显著的频率越高。

    这里只断言【单调】不断言【高】：600 日 + 100 组参数下，DSR 门槛在年化夏普
    2 附近（见 required_sharpe_table），真 edge 也常常过不了。过不了是对的 ——
    数据就是不够，不是检验太严。断言"必须高功效"反而会逼出错误的实现。
    """
    entry, exits = _grid(10, 10)
    rates = []
    for drift in (0.0, 0.0010, 0.0018):
        rng = np.random.default_rng(777)
        hits = sum(_run_once(rng, entry, exits, drift=drift)[0].significant for _ in range(120))
        rates.append(hits / 120)

    assert rates[0] < rates[1] < rates[2], f"功效非单调：{rates}"
    assert rates[0] <= 0.06, f"零漂移下假阳性率 {rates[0]:.1%} 超标"
    assert rates[2] >= 0.25, f"强 edge 下功效仅 {rates[2]:.1%}，检验过钝"


# ── 与 vnpy 寻优返回值的接线 ──────────────────────────────────────────

def test_deflate_optimization_on_vnpy_shaped_grid_results() -> None:
    """把网格结果拼成 vnpy `evaluate()` 的 (setting, target, statistics) 三元组，
    走 `deflate_optimization()` 全链路 —— 验证接线，不只是验证公式。
    """
    rng = np.random.default_rng(20260727)
    entry, exits = _grid(10, 10)
    _, sharpes, strat, _ = _run_once(rng, entry, exits)

    combos = [(int(e), int(x)) for e in entry for x in exits]
    results = []
    for j, (e, x) in enumerate(combos):
        moments = return_moments(strat[:, j])
        results.append((
            {"entry_window": e, "exit_window": x},
            float(sharpes[j]),
            {
                "sharpe_ratio": float(sharpes[j]),
                "total_days": int(strat.shape[0]),
                "sharpe_skew": moments.skew,
                "sharpe_kurtosis": moments.kurtosis,
            },
        ))
    # vnpy 的 run_bf_optimization 会按目标值降序排好再返回
    results.sort(key=lambda item: item[1], reverse=True)

    result = deflate_optimization(results, annual_days=HK_ANNUAL_DAYS)

    # N 取到了全部试验，不是 top N
    assert result.n_trials == len(combos) == 100
    # 选中的就是排序后的第一组
    assert result.observed_sharpe_annual == pytest.approx(float(np.max(sharpes)))
    # 纯噪声 → 不显著
    assert not result.significant
    assert result.expected_max_sharpe_annual > 0.0
    # deflation 一定让结论更严
    assert result.deflated_sharpe_ratio < result.probabilistic_sharpe_ratio
    assert result.summary()


def test_strategy_returns_violate_bailey_iid_assumptions() -> None:
    """把本文件存在的理由钉死：策略收益确实零膨胀 + 自相关 + 试验高度相关。

    这三条都违反 Bailey 推导的假设。上面的假阳性率测试因此不是走过场 ——
    它证明的是"假设被违反时结论仍然成立"。
    """
    rng = np.random.default_rng(20260728)
    entry, exits = _grid(10, 10)
    _, sharpes, strat, best = _run_once(rng, entry, exits)

    best_returns = strat[:, best]

    # 1) 零膨胀：空仓日收益恒为 0
    zero_share = float(np.mean(best_returns == 0.0))
    assert zero_share > 0.2, f"空仓日占比仅 {zero_share:.1%}，网格设置可能不对"

    # 2) 试验间高度相关：相邻参数的收益序列相关系数很高
    corr = np.corrcoef(strat[:, 0], strat[:, 1])[0, 1]
    assert corr > 0.5, f"相邻参数相关系数仅 {corr:.2f}"

    # 3) 试验夏普的离散度是实打实的（V 不退化）
    assert float(np.std(sharpes, ddof=1)) > 0.1
