"""DSR 对抗性审查：把审查中实测到的【失效模式与边界】钉成可执行断言。

和现有两份 DSR 测试的分工
────────────────────────────────────────────────────────────────────────
`test_deflated_sharpe*.py`  验证公式在教科书假设下算得对。
`test_dsr_turtle_grid.py`   验证在"真网格 + 零漂移随机游走"下假阳性率受控。
本文件                       验证【上面两份没覆盖、且方向不安全】的那些地方。

审查结论一句话：公式实现是对的（PSR/门槛反解与 scipy 差 1e-16），但
`test_dsr_turtle_grid.py` 的 docstring 声称它验证了"收益自相关时结论仍成立"，
而它的 DGP（iid 高斯随机游走）根本产生不出自相关的策略收益 —— 那是鞅差序列。
真正引入自相关后，PSR/DSR 的名义 5% 水平会朝【不安全】方向失守。

本文件把这件事钉住，并给出修复方向的量化证据（HAC 有效样本量）。

跑法：
    .venv/bin/python -m pytest tests/test_dsr_adversarial_audit.py -q
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# 同目录测试模块，pytest prepend 模式下可直接导入
from test_dsr_turtle_grid import (
    _grid,
    turtle_grid_sharpes,
)

from vnpy_ctastrategy.deflated_sharpe import (
    deflate_optimization,
    expected_max_sharpe,
    minimum_sharpe_for_confidence,
    probabilistic_sharpe_ratio,
    return_moments,
)
from vnpy_ctastrategy.sharpe_inference import (
    deflate_optimization_results,
    newey_west_lags,
)
from vnpy_ctastrategy.sharpe_inference import (
    sharpe_variance_factor as hac_variance_factor,
)

HK_ANNUAL_DAYS = 247
SAMPLE_DAYS = 600
DAILY_VOL = 0.02
NOMINAL_ALPHA = 0.05


# ── 工具：AR(1) 收益 ──────────────────────────────────────────────────

def ar1_returns(rng: np.random.Generator, n: int, phi: float, vol: float) -> np.ndarray:
    """无条件均值恒为 0、无条件方差恒为 vol² 的 AR(1) 序列。

    真实夏普严格 = 0，所以任何"显著"判定都是货真价实的第一类错误。
    创新方差乘 (1−φ²) 以固定无条件方差，避免"φ 变大 ⇒ 方差变大"的混淆。
    """
    innovations = rng.standard_normal(n) * vol * math.sqrt(1.0 - phi * phi)
    out = np.empty(n, dtype=float)
    out[0] = innovations[0]
    for t in range(1, n):
        out[t] = phi * out[t - 1] + innovations[t]
    return out


def psr_rejection_rate(
    phi: float, reps: int, seed: int, use_hac: bool = False
) -> float:
    """真实夏普 = 0 的 AR(1) 序列上，PSR(0) ≥ 0.95 的频率。"""
    rng = np.random.default_rng(seed)
    lags = newey_west_lags(SAMPLE_DAYS)
    hits = 0
    for _ in range(reps):
        returns = ar1_returns(rng, SAMPLE_DAYS, phi, DAILY_VOL)
        moments = return_moments(returns)
        if moments.std <= 0.0:
            continue
        sharpe = moments.mean / moments.std

        n_obs = SAMPLE_DAYS
        if use_hac:
            v_hac = hac_variance_factor(returns, sharpe, lags, strict=False)
            v_iid = 1.0 - moments.skew * sharpe + (moments.kurtosis - 1.0) / 4.0 * sharpe**2
            if v_hac > 0.0:
                n_obs = max(2, int(1.0 + (SAMPLE_DAYS - 1) * v_iid / v_hac))

        psr = probabilistic_sharpe_ratio(
            sharpe, 0.0, n_obs, moments.skew, moments.kurtosis
        )
        hits += int(psr >= 0.95)
    return hits / reps


# ── 1. 控制组：iid 下名义水平确实成立 ────────────────────────────────

def test_psr_nominal_level_holds_under_iid() -> None:
    """φ=0（真 iid、真实夏普=0）时，PSR 的第一类错误率应贴着名义 5%。

    这是下面失守测试的对照组：先证明检验本身没写错，失守才归因于自相关。
    """
    rate = psr_rejection_rate(phi=0.0, reps=3000, seed=4242)
    assert 0.035 <= rate <= 0.065, f"iid 下第一类错误率 {rate:.2%} 偏离名义 5%"


# ── 2. 【核心发现】正自相关下名义水平朝不安全方向失守 ────────────────

@pytest.mark.parametrize(
    ("phi", "min_rate"),
    [(0.10, 0.062), (0.20, 0.080), (0.30, 0.100)],
)
def test_psr_nominal_level_fails_under_positive_autocorrelation(
    phi: float, min_rate: float
) -> None:
    """收益正自相关时，PSR/DSR 的第一类错误率显著超过名义 5%。

    真实夏普严格为 0（AR(1) 无条件均值 = 0），所以每一次"显著"都是假阳性。
    实测：φ=0.10→7.0%、φ=0.20→9.2%、φ=0.30→11.7%，即名义 5% 的 1.4~2.3 倍。

    机理：正自相关抬高样本均值的抽样方差，而 PSR 的 SE 仍按 T 个独立观测算
    （√(T−1)），SE 被低估 ⇒ t 统计量被高估 ⇒ 过度拒绝。

    DSR = PSR(SR*)，同一个 SE 进 DSR，所以这个偏差【原样传导到 DSR】，
    且方向是【不安全】的（把运气判成 edge）——正是本项目要防的那类事故。

    为什么现有的 test_dsr_turtle_grid 测不出来：它的标的收益是 iid 高斯，
    仓位只用 t−1 信息，故策略收益是鞅差序列、线性不自相关（见下面那条测试）。
    """
    rate = psr_rejection_rate(phi=phi, reps=3000, seed=4242)
    assert rate >= min_rate, (
        f"φ={phi} 下第一类错误率仅 {rate:.2%}，未复现失守 —— "
        "若此断言失败请重新审视本文件的结论，而不是直接放宽阈值"
    )
    assert rate > NOMINAL_ALPHA


# ── 3. 修复方向：HAC 有效样本量能把水平大致拉回来 ────────────────────

def test_hac_effective_sample_size_restores_calibration() -> None:
    """把 T 换成 HAC 有效样本量后，超额第一类错误被削掉大半。

    T_eff = 1 + (T−1)·V_iid / V_hac，V_hac 用 sharpe_inference 里已有的
    Newey-West 长期方差（该模块默认就是 method="hac"，只是【没接进 DSR 这条链】）。

    实测（φ=0.20，3000 次）：9.2% → 6.1%，超额部分削掉约 3/4。
    残余的过度拒绝来自 HAC 自身的小样本偏差，sharpe_inference 的模块 docstring
    第 21 行已明确记载这一点，故此处只断言"显著改善"，不断言"完全归位"。
    """
    phi = 0.20
    naive = psr_rejection_rate(phi=phi, reps=3000, seed=4242, use_hac=False)
    hac = psr_rejection_rate(phi=phi, reps=3000, seed=4242, use_hac=True)

    assert hac < naive, f"HAC 未改善：naive {naive:.2%} → hac {hac:.2%}"
    excess_removed = (naive - hac) / (naive - NOMINAL_ALPHA)
    assert excess_removed >= 0.5, f"HAC 只削掉 {excess_removed:.0%} 的超额错误"
    assert hac <= 0.075, f"HAC 后仍达 {hac:.2%}"


# ── 4. 钉死：现有网格测试的 DGP 产生不出自相关收益 ──────────────────

def test_turtle_grid_returns_are_serially_uncorrelated_by_construction() -> None:
    """test_dsr_turtle_grid 的 docstring 声称其 DGP 违反了"收益自相关"假设 —— 不成立。

    标的收益 iid + 仓位只依赖 t−1 信息 ⇒ 策略收益 pos_t·r_t 是鞅差序列，
    线性自相关期望为 0。实测 200 轮 lag-1 自相关均值 ≈ −0.01，95% 区间跨 0。

    这条不是否定那份测试的价值（它对"零膨胀 + 试验相关"的验证是真的），
    而是标定它的射程：**它没有、也不可能验证自相关鲁棒性**，那部分由本文件
    第 2/3 条负责。
    """
    entry, exits = _grid(10, 10)
    rng = np.random.default_rng(99)
    lag1 = []
    for _ in range(200):
        rets = rng.standard_normal(SAMPLE_DAYS + 1) * DAILY_VOL
        prices = 100.0 * np.exp(np.cumsum(rets))
        sharpes, strat = turtle_grid_sharpes(prices, entry, exits)
        best = strat[:, int(np.argmax(sharpes))]
        if best.std() > 0:
            lag1.append(float(np.corrcoef(best[:-1], best[1:])[0, 1]))

    mean_ac = float(np.mean(lag1))
    assert abs(mean_ac) < 0.05, f"策略收益 lag-1 自相关 {mean_ac:+.4f}，与鞅差预期不符"

    # 零膨胀带来的是【绝对值】自相关（波动聚集），不是收益自相关。
    # PSR 的 γ₄ 项能吃掉一部分零膨胀，但吃不掉均值抽样方差的膨胀。
    abs_ac = []
    rng = np.random.default_rng(99)
    for _ in range(50):
        rets = rng.standard_normal(SAMPLE_DAYS + 1) * DAILY_VOL
        prices = 100.0 * np.exp(np.cumsum(rets))
        sharpes, strat = turtle_grid_sharpes(prices, entry, exits)
        best = np.abs(strat[:, int(np.argmax(sharpes))])
        if best.std() > 0:
            abs_ac.append(float(np.corrcoef(best[:-1], best[1:])[0, 1]))
    assert float(np.mean(abs_ac)) > 0.2, "预期存在 |收益| 自相关（空仓期成串）"


# ── 5. 优先级：非正态修正远不如有效样本量值钱 ────────────────────────

def test_nonnormality_correction_is_dwarfed_by_sample_size_error() -> None:
    """日频下 PSR 的非正态修正只值 ~0.1 夏普，而 T 打对折值 ~0.8 夏普。

    日频 SR ≈ 1.7/√247 ≈ 0.11，分母 1 − γ₃·SR + (γ₄−1)/4·SR² 里两个修正项都是
    O(1e-2)。所以"DSR 处理了非正态"这个卖点在日频语境下兑现得很少；而它【没处理】
    的有效样本量问题，量级大一个数量级。

    这条测试的用途是防止有人把 DSR 当成"已经处理了收益非正态所以可以放心"的凭证。
    """
    root = math.sqrt(HK_ANNUAL_DAYS)
    sr_star = expected_max_sharpe(0.25 / root, 100)

    base = minimum_sharpe_for_confidence(sr_star, 600, 0.0, 3.0) * root
    fat = minimum_sharpe_for_confidence(sr_star, 600, -1.0, 30.0) * root
    half_t = minimum_sharpe_for_confidence(sr_star, 300, 0.0, 3.0) * root

    nonnormal_cost = abs(fat - base)
    sample_cost = abs(half_t - base)

    assert nonnormal_cost < 0.15, f"非正态修正 {nonnormal_cost:.3f} 超出预期量级"
    assert sample_cost > 0.35, f"T 减半代价 {sample_cost:.3f} 低于预期量级"
    assert sample_cost > 3 * nonnormal_cost


# ── 6. E[max] 在极小 N 下【低估】——与"小 N 偏保守"的说法相反 ────────

def test_expected_max_is_anticonservative_at_tiny_n() -> None:
    """Bailey 的 E[max] 极值近似在 N=2 时低估真值约 6~8%（不是高估）。

    有说法称"N 小时误差偏大但方向是高估 SR*，偏保守，安全"。实测不成立：
    N=2 低估（不安全方向），N=5~10 才转为高估。N≥100 后误差 <1%。

    实务影响很小（没人跑 N=2 的网格），但"小 N 总是保守"这个安全性论断本身是错的，
    不该被当作免检理由写进结论。
    """
    rng = np.random.default_rng(1)
    std = 0.5

    closed_2 = expected_max_sharpe(std, 2)
    mc_2 = float(np.mean(np.max(rng.standard_normal((60000, 2)) * std, axis=1)))
    assert closed_2 < mc_2, "N=2 处闭式解未低估，结论需复核"
    assert (mc_2 - closed_2) / mc_2 > 0.04

    closed_100 = expected_max_sharpe(std, 100)
    mc_100 = float(np.mean(np.max(rng.standard_normal((60000, 100)) * std, axis=1)))
    assert abs(closed_100 - mc_100) / mc_100 < 0.02, "N=100 处误差应 <2%"


# ── 7. 【集成风险】同包内两个 deflate_* 在非夏普目标下结论相反 ───────

def test_two_deflate_entrypoints_diverge_on_nonsharpe_target() -> None:
    """`sharpe_inference.deflate_optimization_results` 读 row[1]（寻优目标值），
    在目标 ≠ sharpe_ratio 时把目标值当夏普用，给出错误的"可信"绿灯。

    本项目寻优常用 `return_drawdown_ratio` 当目标，其量级（此例 3.0）远高于夏普
    （此例 0.9）。同一份寻优结果：
        sharpe_inference.deflate_optimization_results → DSR 0.98, trustworthy=True
        deflated_sharpe.deflate_optimization          → DSR 0.81, significant=False

    两个函数名字接近、同包共存、都可 import。若把 DSR 当上实盘的验收闸，
    调错入口 = 闸门失效。deflated_sharpe.deflate_optimization 是正确的那个
    （它读 statistics["sharpe_ratio"]）。
    """
    results = []
    for i in range(50):
        rdr = 3.0 - i * 0.05          # 寻优目标 return_drawdown_ratio
        sharpe = 0.9 - i * 0.01       # 真实年化夏普
        results.append((
            {"entry_window": 10 + i},
            rdr,
            {
                "sharpe_ratio": sharpe,
                "total_days": SAMPLE_DAYS,
                "return_drawdown_ratio": rdr,
            },
        ))
    results.sort(key=lambda row: row[1], reverse=True)

    # 修复后：喂非夏普目标必须直接拒绝，而不是把 return_drawdown_ratio 当夏普
    # 算出一个看似可信的数字。之前它返回 best_sharpe_annual=3.0、
    # trustworthy=True，与正确入口的结论完全相反 —— 一道本该挡住多重比较的闸
    # 对错误输入盖了章。
    with pytest.raises(ValueError, match="夏普族"):
        deflate_optimization_results(
            results, n_periods=SAMPLE_DAYS, target_name="return_drawdown_ratio",
            annual_days=HK_ANNUAL_DAYS,
        )

    # 显式声明是夏普目标时照常工作（此时 row[1] 才真的是夏普）
    sharpe_rows = [(row[0], row[2]["sharpe_ratio"], row[2]) for row in results]
    sibling = deflate_optimization_results(
        sharpe_rows, n_periods=SAMPLE_DAYS, target_name="sharpe_ratio",
        annual_days=HK_ANNUAL_DAYS,
    )
    audited = deflate_optimization(results, annual_days=HK_ANNUAL_DAYS)

    # 两个入口现在读的是同一个量，结论一致
    assert sibling["best_sharpe_annual"] == pytest.approx(0.9)
    assert audited.observed_sharpe_annual == pytest.approx(0.9)
    assert sibling["trustworthy"] is False
    assert audited.significant is False
