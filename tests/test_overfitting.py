"""Walk-Forward 与 CSCV/PBO 的验证测试。

本文件的重点不是"函数跑不跑得通"，而是**方法本身有没有效**。核心是三类测试：

  1. 零假设下的表现（size）—— 对纯随机序列，方法必须判"不显著 / PBO≈0.5"。
     一个把噪声判成 alpha 的检验比没有检验更危险，因为它会给出虚假的安全感。
  2. 有真实效应时的表现（power）—— 对确实有 edge 的序列必须判"显著 / PBO≈0"。
     只会说"不显著"的检验同样没用。
  3. 构造性反例 —— 人为造一个"样本内最优必然样本外最差"的矩阵，PBO 必须 >0.5。

另有一组"钉住已知性质"的测试（并列处理、单次估计的离散度、快慢路径一致性），
防止后续重构悄悄改掉这些性质。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
from pandas import DataFrame
from vnpy.trader.constant import Interval

from vnpy_ctastrategy.backtesting import BacktestingEngine
from vnpy_ctastrategy.overfitting import (
    EngineRunner,
    PBOResult,
    annualised_sharpe,
    argmax_selector,
    average_config_correlation,
    cscv_pbo,
    cscv_pbo_stability,
    daily_log_returns,
    make_walk_forward_splits,
    pbo_null_distribution,
    pbo_verdict,
    pbo_verdict_calibrated,
    plateau_selector,
    recommend_n_blocks,
    returns_matrix,
    run_walk_forward,
    sample_size_diagnosis,
    sharpe_standard_error,
    stitch_daily_frames,
    assess_significance,
)


ANNUAL_DAYS = 252
CAPITAL = 1_000_000.0


# ══════════════════════════════════════════════════════════════════════
# 工具：与引擎口径对齐的假 runner
# ══════════════════════════════════════════════════════════════════════

class FakeRunner:
    """把一张预先生成好的 (T, N) 逐日盈亏表当成回测引擎。

    关键性质：**同一组参数在同一天的盈亏与窗口无关** —— 先生成整段序列，再按窗口切片。
    真实引擎不完全满足这一点（折起点空仓），但对"检验方法本身有没有效"来说，
    引入引擎的路径依赖只会把噪声源搅混，反而看不清方法的 size / power。
    """

    def __init__(self, dates: list[datetime], pnl: np.ndarray, settings: list[dict]) -> None:
        if pnl.shape != (len(dates), len(settings)):
            raise ValueError(f"pnl shape {pnl.shape} 与 dates/settings 不匹配")
        self.dates = dates
        self.pnl = pnl
        self.keys = [self._key(s) for s in settings]

    @staticmethod
    def _key(setting: dict) -> tuple:
        return tuple(sorted(setting.items()))

    def __call__(self, setting: dict, start: datetime, end: datetime) -> DataFrame:
        col = self.keys.index(self._key(setting))
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
        """最小 statistics 面板：只保留 walk-forward 用得到的两个键。"""
        if df is None or df.empty:
            return {"sharpe_ratio": 0.0, "annual_return": 0.0}
        r = daily_log_returns(df["net_pnl"].to_numpy(), CAPITAL)
        total = (float(np.sum(df["net_pnl"].to_numpy())) / CAPITAL) * 100.0
        return {
            "sharpe_ratio": annualised_sharpe(r, ANNUAL_DAYS),
            "annual_return": total / len(df) * ANNUAL_DAYS,
        }


def _dates(n: int) -> list[datetime]:
    base = datetime(2024, 1, 2)
    return [base + timedelta(days=i) for i in range(n)]


def _grid(n: int) -> list[dict]:
    return [{"window": 10 + i} for i in range(n)]


# ══════════════════════════════════════════════════════════════════════
# 1. 口径对齐 —— 与 calculate_statistics 必须是同一个数
# ══════════════════════════════════════════════════════════════════════

def test_daily_log_returns_matches_engine() -> None:
    """本模块的日收益必须与 `calculate_statistics` 写进 df["return"] 的那一列逐位相同。

    口径一旦分叉，PBO 里的 Sharpe 和面板上的 sharpe_ratio 就不是同一个数，
    两边的结论无法互相印证 —— 这条测试是防止分叉的锚。
    """
    rng = np.random.default_rng(11)
    pnl = rng.normal(0, 8_000, size=200)

    df = DataFrame(
        {
            "net_pnl": pnl,
            "commission": np.zeros(200), "slippage": np.zeros(200),
            "turnover": np.zeros(200), "trade_count": np.zeros(200),
        },
        index=[d.date() for d in _dates(200)],
    )
    engine = BacktestingEngine()
    engine.capital = int(CAPITAL)
    engine.annual_days = ANNUAL_DAYS
    stats = engine.calculate_statistics(df=df, output=False)

    ours = daily_log_returns(pnl, CAPITAL)
    np.testing.assert_allclose(ours, df["return"].to_numpy(), rtol=0, atol=1e-15)

    # Sharpe 也必须对上
    assert annualised_sharpe(ours, ANNUAL_DAYS) == pytest.approx(stats["sharpe_ratio"], rel=1e-10)


def test_annualised_sharpe_is_zero_when_flat() -> None:
    """从不交易的参数组应当排在"平庸"，不是"缺失" —— 与 vnpy 的 else 分支一致。"""
    assert annualised_sharpe(np.zeros(100), ANNUAL_DAYS) == 0.0
    assert annualised_sharpe(np.array([0.01]), ANNUAL_DAYS) == 0.0


def test_daily_log_returns_rejects_nonpositive_capital() -> None:
    with pytest.raises(ValueError):
        daily_log_returns([1.0, 2.0], 0.0)


# ══════════════════════════════════════════════════════════════════════
# 2. Sharpe 标准误 —— 复现本项目那条 134 日曲线
# ══════════════════════════════════════════════════════════════════════

def test_sharpe_standard_error_reproduces_the_134_day_case() -> None:
    """项目实测：134 日、Sharpe=−1.68 的曲线，标准误约 1.34-1.38，|t|<2。

    这正是现有面板缺的那一行：−1.68 看着像"策略废了"，其实连符号都没到 2σ。
    """
    se = sharpe_standard_error(-1.68, 134, 240)
    assert 1.30 < se < 1.40
    assert abs(-1.68 / se) < 1.96          # 连符号都没到 2σ

    se252 = sharpe_standard_error(-1.68, 134, 252)
    assert 1.33 < se252 < 1.42


def test_sharpe_standard_error_shrinks_with_sample() -> None:
    a = sharpe_standard_error(1.0, 250, ANNUAL_DAYS)
    b = sharpe_standard_error(1.0, 1000, ANNUAL_DAYS)
    assert b < a
    assert a / b == pytest.approx(2.0, rel=0.02)   # SE ∝ 1/√T


# ══════════════════════════════════════════════════════════════════════
# 3. 显著性检验的 size 与 power —— 【方法有效性的核心测试】
# ══════════════════════════════════════════════════════════════════════

def test_random_series_is_judged_insignificant() -> None:
    """纯随机日收益必须判为不显著。这是"方法本身是否有效"的第一块试金石。"""
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.012, size=300)
    res = assess_significance(r, ANNUAL_DAYS, n_bootstrap=2000, seed=1)

    assert not res.significant
    assert res.p_block_bootstrap > 0.05
    assert abs(res.t_stat) < 1.96


def test_false_positive_rate_is_near_nominal_alpha() -> None:
    """size 检验：对 200 条纯噪声序列，α=5% 的检验误报率不应明显超过 5%。

    只测一条随机序列说明不了什么（单次可能恰好碰上尾部）。这里做 200 次重复，
    统计真实的第一类错误率 —— 这才是"检验有效"的定义。
    上界放到 10%：200 次重复下 5% 真实水平的二项 95% 上界约 8.5%，留一点余量。
    """
    hits = 0
    n_rep = 200
    for s in range(n_rep):
        rng = np.random.default_rng(1000 + s)
        r = rng.normal(0.0, 0.012, size=300)
        res = assess_significance(r, ANNUAL_DAYS, n_bootstrap=600, seed=s)
        if res.p_block_bootstrap < 0.05:
            hits += 1
    rate = hits / n_rep
    assert rate <= 0.10, f"误报率 {rate:.3f} 远高于名义 5%，检验失效"


def test_real_edge_is_detected() -> None:
    """power 检验：一条真实 Sharpe≈1.6 的 500 日序列必须被判显著。

    只会说"不显著"的检验和没有检验一样没用。
    """
    rng = np.random.default_rng(5)
    r = rng.normal(0.0, 0.01, size=500) + 0.001      # 年化 Sharpe ≈ 1.6
    res = assess_significance(r, ANNUAL_DAYS, n_bootstrap=2000, seed=2)

    assert res.significant
    assert res.p_block_bootstrap < 0.05
    assert res.sharpe > 1.0


def test_significance_check_refuses_to_judge_tiny_samples() -> None:
    """样本太短时返回 nan 而不是一个假的 p 值 —— 诚实优于好看。"""
    res = assess_significance([0.01, -0.01, 0.02], ANNUAL_DAYS)
    assert res.n_obs == 3
    assert not res.significant
    assert np.isnan(res.p_block_bootstrap)


# ══════════════════════════════════════════════════════════════════════
# 4. CSCV / PBO —— size、power、构造性反例
# ══════════════════════════════════════════════════════════════════════

def _noise_matrix(seed: int, n_obs: int = 611, n_configs: int = 50) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 0.01, size=(n_obs, n_configs))


def test_pbo_on_pure_noise_centres_on_one_half() -> None:
    """【核心有效性测试】N 组毫无差别的随机参数，PBO 的期望必须落在 0.5 附近。

    若实现有误（比如排名方向反了、或 IS/OOS 弄混），这个数会系统性偏离 0.5。
    单次估计噪声很大，所以取 30 次重复的均值。
    """
    values = [cscv_pbo(_noise_matrix(s), n_blocks=16, annual_days=ANNUAL_DAYS).pbo for s in range(30)]
    mean = float(np.mean(values))
    assert 0.40 <= mean <= 0.60, f"纯噪声下 PBO 均值 {mean:.3f} 偏离 0.5，实现可疑"


def test_single_pbo_estimate_is_very_noisy_on_600_days() -> None:
    """【钉住小样本真相】611 日样本上，单次 PBO 估计的标准差约 0.15。

    含义：一个 PBO=0.30 的回测，在完全没有选参能力时也有约 20% 概率出现。
    这条测试存在的目的是防止有人日后把"报单一个 PBO 数字"当成结论 ——
    如果哪天这个离散度真的降下去了，是样本变长了，不是方法变准了。
    """
    values = np.array([cscv_pbo(_noise_matrix(s), n_blocks=16).pbo for s in range(30)])
    assert values.std(ddof=1) > 0.08, "离散度异常小，检查是否误把组合当独立样本"
    assert values.max() - values.min() > 0.3


def test_pbo_is_zero_when_one_config_truly_dominates() -> None:
    """power 检验：有一列存在真实且持续的优势时，PBO 必须接近 0。"""
    m = _noise_matrix(1)
    m[:, 7] += 0.004                      # 持续正漂移，与时间切分无关
    result = cscv_pbo(m, n_blocks=16, annual_days=ANNUAL_DAYS)

    assert result.pbo < 0.05
    assert result.rank_ic > 0.0                  # 样本内排名确实预测样本外排名
    assert int(np.argmax(result.selected_counts)) == 7
    assert result.verdict.startswith("稳健")


def test_rank_ic_is_high_when_the_whole_ranking_is_real() -> None:
    """只有一列占优时 rank IC 很小（另外 49 列仍是噪声，截面相关被稀释）；
    只有当【整条排名】都真实存在时，rank IC 才会高。这条测试把两种情形区分开，
    防止把 rank IC 误当成"有没有 alpha"的开关。"""
    m = _noise_matrix(1)
    m += np.linspace(0.0, 0.004, m.shape[1])[None, :]      # 逐列递增的真实优势
    result = cscv_pbo(m, n_blocks=16)
    assert result.rank_ic > 0.6
    assert result.pbo < 0.05


def test_degradation_slope_is_negative_even_for_a_genuine_edge() -> None:
    """钉住一个反直觉但必须讲清的性质：退化回归斜率【天然为负】，不可当过拟合证据。

    同一列的 IS 与 OOS 是互补的两半，二者之和恒等于该列的全样本表现，
    所以即便某列确有真实优势，跨组合回归的斜率仍会被这个恒等式压成 −1 附近。
    如果哪天有人拿"斜率是负的"去论证过拟合，这条测试就是反例。
    """
    m = _noise_matrix(1)
    m[:, 7] += 0.004
    result = cscv_pbo(m, n_blocks=16)
    assert result.pbo < 0.05                     # 这明明是"没有过拟合"的情形
    assert result.degradation_slope < 0          # 但斜率照样是负的


def test_designed_overfit_matrix_gives_pbo_above_half() -> None:
    """构造性反例：样本内越好、样本外必然越差的矩阵，PBO 必须 >0.5。

    构造方法：给每组参数一张【块级收益表】，并把每一列在所有块上的和强制归零。
    于是任一组合中，样本内表现最好的那列，其样本外表现必然最差（和为零 ⇒ 一半好、
    另一半必然同等地差）—— 这就是"在噪声上选参"的纯净形态。

    注意不能用"奇偶块交替符号、各列载荷不同"来构造：Sharpe 是尺度无关的，
    载荷会在分子分母上同时约掉，各列 Sharpe 反而完全相同，做不出效果。
    """
    n_blocks, block, n_cfg = 16, 38, 30
    rng = np.random.default_rng(4)
    block_mean = rng.standard_normal((n_blocks, n_cfg))
    block_mean -= block_mean.mean(axis=0, keepdims=True)      # 每列跨块求和 = 0
    m = np.repeat(block_mean * 0.01, block, axis=0)
    m += rng.normal(0.0, 0.001, size=m.shape)

    result = cscv_pbo(m, n_blocks=n_blocks, annual_days=ANNUAL_DAYS)
    assert result.pbo > 0.9
    assert result.rank_ic < -0.5                 # 排名系统性反转
    assert result.verdict.startswith("反向有害")


def test_rank_ic_is_near_zero_on_pure_noise() -> None:
    """纯噪声下截面 rank IC 应当围绕 0。"""
    ics = [cscv_pbo(_noise_matrix(s), n_blocks=16).rank_ic for s in range(20)]
    assert abs(float(np.mean(ics))) < 0.15


def test_identical_columns_give_one_half_not_one() -> None:
    """并列处理：所有列完全相同 = 毫无区分度，应判 0.5（无信息），不是 1.0。

    原文的 P(λ≤0) 在全并列时会给出 1.0（"全部过拟合"），是明显误判。
    本实现用 P(λ<0)+0.5·P(λ=0)，连续情形与原文等价，只在退化情形更诚实。
    """
    col = np.random.default_rng(2).normal(0, 0.01, size=(608, 1))
    m = np.repeat(col, 20, axis=1)
    result = cscv_pbo(m, n_blocks=16)

    assert result.pbo == pytest.approx(0.5)
    assert result.tie_fraction == pytest.approx(1.0)


def test_fast_path_matches_generic_path() -> None:
    """块矩快路径与逐组合拼接的通用路径必须给出同一个 PBO。

    快路径把 C(16,8)=12870 次子矩阵拼接压成一次矩阵乘法（快约 30 倍），
    但一旦算错就是静默错误 —— 用小规模的 S=8 做等价性对拍。
    """
    m = _noise_matrix(9, n_obs=240, n_configs=12)

    def generic(sub: np.ndarray) -> np.ndarray:
        return np.array([annualised_sharpe(sub[:, i], ANNUAL_DAYS) for i in range(sub.shape[1])])

    fast = cscv_pbo(m, n_blocks=8, annual_days=ANNUAL_DAYS)
    slow = cscv_pbo(m, n_blocks=8, annual_days=ANNUAL_DAYS, performance=generic)

    assert fast.pbo == pytest.approx(slow.pbo)
    np.testing.assert_allclose(fast.logits, slow.logits, rtol=1e-9, atol=1e-9)


def test_cscv_input_validation() -> None:
    m = _noise_matrix(0, n_obs=200, n_configs=5)
    with pytest.raises(ValueError, match="偶数"):
        cscv_pbo(m, n_blocks=7)
    with pytest.raises(ValueError, match="2 组参数"):
        cscv_pbo(m[:, :1], n_blocks=4)
    bad = m.copy()
    bad[3, 2] = np.nan
    with pytest.raises(ValueError, match="nan"):
        cscv_pbo(bad, n_blocks=4)
    with pytest.raises(ValueError, match="超过上限"):
        cscv_pbo(_noise_matrix(0, 2000, 5), n_blocks=24)


def test_pbo_stability_across_block_phases() -> None:
    """块网格相位平移后 PBO 会变 —— 报告必须给出区间而不是单点。"""
    results, summary = cscv_pbo_stability(_noise_matrix(6), n_blocks=16, n_offsets=8)
    assert len(results) == 8
    assert summary["pbo_min"] <= summary["pbo_median"] <= summary["pbo_max"]
    assert summary["pbo_spread"] >= 0.0


def test_low_config_count_raises_warning() -> None:
    result = cscv_pbo(_noise_matrix(0, 400, 4), n_blocks=8)
    assert any("组参数" in w for w in result.warnings)


# ══════════════════════════════════════════════════════════════════════
# 5. 零分布校准
# ══════════════════════════════════════════════════════════════════════

def test_null_distribution_is_wide_and_centred_near_half() -> None:
    """零分布的形状必须与实测一致：均值≈0.45-0.50，标准差≈0.15。"""
    null = pbo_null_distribution(608, 50, 16, ANNUAL_DAYS, correlation=0.0, n_sims=80, seed=3)
    assert 0.40 <= null.mean <= 0.58
    assert 0.10 <= null.sd <= 0.25
    assert null.q05 < null.q50 < null.q95


def test_null_pvalue_does_not_flag_noise_as_robust() -> None:
    """把一条纯噪声矩阵拿去跟零分布比，不应被判为"通过"。

    这是把 PBO 从"数字对民间阈值"升级成"检验"之后，最该守住的一条：
    随机数据必须过不了关。
    """
    null = pbo_null_distribution(608, 50, 16, ANNUAL_DAYS, n_sims=80, seed=3)
    flagged = 0
    for s in range(20):
        pbo = cscv_pbo(_noise_matrix(500 + s), n_blocks=16).pbo
        if pbo_verdict_calibrated(pbo, null).startswith("通过"):
            flagged += 1
    assert flagged <= 2, f"20 条纯噪声里有 {flagged} 条被判通过，校准失效"


def test_null_pvalue_flags_a_true_edge() -> None:
    """有真实优势时，p 值必须很小。"""
    null = pbo_null_distribution(608, 50, 16, ANNUAL_DAYS, n_sims=80, seed=3)
    m = _noise_matrix(1)
    m[:, 7] += 0.004
    pbo = cscv_pbo(m, n_blocks=16).pbo
    assert null.p_value(pbo) < 0.05
    assert pbo_verdict_calibrated(pbo, null).startswith("通过")


def test_average_config_correlation() -> None:
    base = np.random.default_rng(8).normal(0, 0.01, size=(300, 1))
    almost_same = base + np.random.default_rng(9).normal(0, 0.0005, size=(300, 6))
    assert average_config_correlation(almost_same) > 0.9
    assert abs(average_config_correlation(_noise_matrix(0, 2000, 20))) < 0.1


# ══════════════════════════════════════════════════════════════════════
# 6. 块数推荐与样本诊断
# ══════════════════════════════════════════════════════════════════════

def test_recommend_n_blocks_on_611_days() -> None:
    """611 日、最小块长 20 日 → S=16 合法（块长 38）。"""
    s, notes = recommend_n_blocks(611, min_block_obs=20, max_blocks=16)
    assert s == 16
    assert 611 // s == 38
    assert notes == []


def test_recommend_n_blocks_degrades_when_holding_is_long() -> None:
    """持仓 30 日 → 要求块长 ≥60 日 → S 必须降到 10。"""
    s, notes = recommend_n_blocks(611, min_block_obs=60, max_blocks=16)
    assert s == 10
    assert 611 // s >= 60
    assert notes and "降到" in notes[0]


def test_recommend_n_blocks_refuses_tiny_samples() -> None:
    s, notes = recommend_n_blocks(30)
    assert s == 0
    assert notes


def test_sample_diagnosis_flags_low_power_on_611_days() -> None:
    """611 日 / S=16 的诊断必须明确指出这是低功效样本。

    IS/OOS 各 304 期 → Sharpe 标准误≈0.91 → 最小可检出 Sharpe≈2.5，
    远高于真实 CTA 的 0.5-1.0。这条数字化的结论必须出现在诊断里。
    """
    d = sample_size_diagnosis(611, 16, ANNUAL_DAYS, trades_total=22, median_holding_days=18)
    assert d.obs_per_side == 304
    assert 0.85 < d.sharpe_se_per_side < 0.95
    assert d.min_detectable_sharpe > 2.0
    assert any("低功效" in n for n in d.notes)
    assert any("20 笔" in n for n in d.notes)      # 每侧 11 笔 < 20 笔下限


# ══════════════════════════════════════════════════════════════════════
# 7. Walk-Forward 切分几何 —— 前视泄漏必须为零
# ══════════════════════════════════════════════════════════════════════

def test_no_leakage_between_train_and_test() -> None:
    """任何一折的测试窗都必须严格晚于其训练窗。"""
    dts = _dates(611)
    splits = make_walk_forward_splits(dts, train_bars=252, test_bars=63)
    assert splits
    for sp in splits:
        assert sp.train_start <= sp.train_end < sp.test_start <= sp.test_end
        assert sp.train_bars == 252
        assert sp.test_bars == 63


def test_test_windows_do_not_overlap_and_tile_the_sample() -> None:
    """测试窗必须首尾相接、互不重叠 —— 否则拼接曲线会重复计入同一天。"""
    dts = _dates(611)
    splits = make_walk_forward_splits(dts, 252, 63)
    assert len(splits) == (611 - 252) // 63          # = 5 折
    for a, b in zip(splits[:-1], splits[1:], strict=True):
        assert a.test_end < b.test_start
    covered = sum(sp.test_bars for sp in splits)
    assert covered == 5 * 63


def test_anchored_mode_expands_training_window() -> None:
    dts = _dates(611)
    rolling = make_walk_forward_splits(dts, 252, 63, anchored=False)
    anchored = make_walk_forward_splits(dts, 252, 63, anchored=True)
    assert len(rolling) == len(anchored)
    assert all(sp.train_bars == 252 for sp in rolling)
    assert [sp.train_bars for sp in anchored] == sorted(sp.train_bars for sp in anchored)
    assert all(sp.train_start == dts[0] for sp in anchored)
    assert anchored[-1].train_bars > anchored[0].train_bars


def test_splits_reject_bad_arguments() -> None:
    dts = _dates(100)
    with pytest.raises(ValueError):
        make_walk_forward_splits(dts, 1, 10)
    with pytest.raises(ValueError):
        make_walk_forward_splits(dts, 50, 0)
    assert make_walk_forward_splits(dts, 90, 20) == []      # 样本不够，返回空而不是崩


def test_stitch_rejects_overlapping_windows() -> None:
    """拼接时若发现重复日期直接报错 —— 静默重复计入是最难查的一类错。"""
    idx = [d.date() for d in _dates(10)]
    a = DataFrame({"net_pnl": np.ones(10)}, index=idx)
    with pytest.raises(ValueError, match="重叠"):
        stitch_daily_frames([a, a])


# ══════════════════════════════════════════════════════════════════════
# 8. 选参规则
# ══════════════════════════════════════════════════════════════════════

def test_argmax_selector_picks_the_single_best() -> None:
    settings = _grid(5)
    scores = np.array([0.1, 0.2, 9.0, 0.3, 0.4])
    assert argmax_selector(settings, scores) == 2


def test_plateau_selector_prefers_a_plateau_over_a_spike() -> None:
    """噪声尖峰 vs 参数高原：高原选参必须选高原中心。

    构造：下标 2 是孤立尖峰（邻居极差），下标 6-8 是连成一片的高原。
    单点最优会选尖峰（典型过拟合形状），高原选参应选 7。
    """
    settings = _grid(10)
    scores = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 1.8, 2.0, 2.2, 2.0, 0.0])
    assert argmax_selector(settings, scores) == 2
    assert plateau_selector(settings, scores) == 7


def test_plateau_selector_handles_multi_parameter_grids() -> None:
    """二维网格：内部有一片 3×3 的高原，边角有一个更高的孤立尖峰。

    边角尖峰的邻居更少 —— 若不做边界惩罚，它会靠"少几个拖后腿的邻居"赢下来。
    这条测试专门钉住那个惩罚。
    """
    settings = [{"a": a, "b": b} for a in (10, 20, 30, 40) for b in (1.0, 2.0, 3.0)]
    scores = np.zeros(len(settings))
    for a in (10, 20, 30):
        for b in (1.0, 2.0, 3.0):
            scores[settings.index({"a": a, "b": b})] = 1.0
    scores[settings.index({"a": 40, "b": 3.0})] = 3.0        # 边角孤立尖峰

    assert settings[argmax_selector(settings, scores)] == {"a": 40, "b": 3.0}
    assert settings[plateau_selector(settings, scores)] == {"a": 20, "b": 2.0}


# ══════════════════════════════════════════════════════════════════════
# 9. Walk-Forward 全流程 —— size 与 power
# ══════════════════════════════════════════════════════════════════════

def _wf_pipeline(seed: int, edge_col: int | None = None, n_settings: int = 20) -> object:
    dates = _dates(611)
    settings = _grid(n_settings)
    rng = np.random.default_rng(seed)
    pnl = rng.normal(0.0, 12_000.0, size=(len(dates), n_settings))
    if edge_col is not None:
        pnl[:, edge_col] += 4_000.0        # 一列有真实优势（年化 Sharpe≈5.3）
    runner = FakeRunner(dates, pnl, settings)
    splits = make_walk_forward_splits(dates, 252, 63)
    return run_walk_forward(
        runner=runner, settings=settings, splits=splits, target_name="sharpe_ratio",
        statistics_func=FakeRunner.statistics, annual_days=ANNUAL_DAYS,
        capital=CAPITAL, n_bootstrap=800, seed=seed,
    )


def test_walk_forward_on_random_data_is_not_significant() -> None:
    """【核心有效性测试】参数之间毫无真实差异时，Walk-Forward 必须判"不通过"。

    这正是本项目栽过的那个跟头的实验室版本：样本内挑出来的最优参数，
    样本外只是噪声。方法必须能在这里说"不"。
    """
    report = _wf_pipeline(seed=42)
    assert len(report.folds) == 5
    assert not report.significance.significant
    assert report.verdict.startswith("不通过")
    # 每折都换一套参数 = 在拟合噪声，稳定性指标必须把它暴露出来
    assert report.parameter_stability < 0.7


def test_walk_forward_false_positive_rate_is_controlled() -> None:
    """size 检验：40 次纯噪声 Walk-Forward，判"通过"的比例不应超过 10%。"""
    passes = sum(1 for s in range(40) if _wf_pipeline(seed=200 + s).verdict.startswith("通过"))
    assert passes / 40 <= 0.10, f"纯噪声下有 {passes}/40 判通过，流程失效"


def test_walk_forward_detects_a_genuine_edge() -> None:
    """power 检验：某组参数确有优势时，Walk-Forward 必须选中它并判显著。"""
    report = _wf_pipeline(seed=17, edge_col=6)
    assert report.significance.significant
    assert report.parameter_stability >= 0.8          # 每折都该选中同一组
    assert report.efficiency_median > 0.3
    assert all(f.chosen_setting == {"window": 16} for f in report.folds)


def test_walk_forward_reports_efficiency_and_stitches_oos_curve() -> None:
    report = _wf_pipeline(seed=17, edge_col=6)
    assert len(report.oos_daily_df) == 5 * 63
    assert not report.oos_daily_df.index.duplicated().any()
    assert np.isfinite(report.efficiency_median)
    text = report.as_dict()
    assert text["n_folds"] == 5
    assert "significance" in text


def test_walk_forward_flags_optimization_that_loses_to_the_baseline() -> None:
    """判据第 4 条：优化打不过默认参数时必须判不通过。

    构造：默认参数（下标 0）才是真正有优势的那一组，但它的优势【只出现在测试窗】，
    训练窗里另一组噪声更好看 —— 于是优化会系统性选错。
    这一条常被忽略，但它是"优化本身是不是负贡献"的唯一直接检验。
    """
    dates = _dates(611)
    settings = _grid(12)
    rng = np.random.default_rng(31)
    pnl = rng.normal(0.0, 12_000.0, size=(len(dates), 12))
    splits = make_walk_forward_splits(dates, 252, 63)
    test_days = {d for sp in splits for d in dates if sp.test_start <= d <= sp.test_end}
    mask = np.array([d in test_days for d in dates])
    pnl[mask, 0] += 4_000.0                       # 默认参数在样本外强

    runner = FakeRunner(dates, pnl, settings)
    report = run_walk_forward(
        runner=runner, settings=settings, splits=splits,
        statistics_func=FakeRunner.statistics, baseline_setting=settings[0],
        annual_days=ANNUAL_DAYS, capital=CAPITAL, n_bootstrap=800, seed=1,
    )
    assert report.baseline_statistics is not None
    assert report.baseline_statistics["sharpe_ratio"] > report.oos_statistics["sharpe_ratio"]
    assert "负贡献" in report.verdict


def test_walk_forward_requires_non_empty_inputs() -> None:
    dates = _dates(100)
    settings = _grid(3)
    runner = FakeRunner(dates, np.zeros((100, 3)), settings)
    splits = make_walk_forward_splits(dates, 60, 20)
    with pytest.raises(ValueError, match="settings 为空"):
        run_walk_forward(runner, [], splits, statistics_func=FakeRunner.statistics)
    with pytest.raises(ValueError, match="splits 为空"):
        run_walk_forward(runner, settings, [], statistics_func=FakeRunner.statistics)


def test_walk_forward_requires_statistics_func_for_custom_runner() -> None:
    dates = _dates(100)
    settings = _grid(3)
    runner = FakeRunner(dates, np.zeros((100, 3)), settings)
    splits = make_walk_forward_splits(dates, 60, 20)
    with pytest.raises(ValueError, match="statistics_func"):
        run_walk_forward(runner, settings, splits)


# ══════════════════════════════════════════════════════════════════════
# 10. 辅助函数
# ══════════════════════════════════════════════════════════════════════

def test_returns_matrix_aligns_on_the_date_union() -> None:
    """某组参数缺某天记录时应补 0 收益（当天没交易 = 净值不变），而不是错位。"""
    idx = [d.date() for d in _dates(5)]
    a = DataFrame({"net_pnl": [100.0] * 5, "trade_count": [1] * 5}, index=idx)
    b = DataFrame({"net_pnl": [100.0] * 3, "trade_count": [1] * 3}, index=idx[:3])
    matrix, dropped = returns_matrix([a, b, DataFrame()], CAPITAL)

    assert matrix.shape == (5, 2)
    assert dropped == [2]
    assert matrix[3, 1] == 0.0 and matrix[4, 1] == 0.0
    assert matrix[0, 0] == pytest.approx(matrix[0, 1])


def test_pbo_verdict_thresholds() -> None:
    assert pbo_verdict(0.05).startswith("稳健")
    assert pbo_verdict(0.20).startswith("边缘")
    assert pbo_verdict(0.45).startswith("不可采信")
    assert pbo_verdict(0.80).startswith("反向有害")
    assert pbo_verdict(float("nan")) == "无法判定"


def test_pbo_result_as_dict_is_flat_and_serialisable() -> None:
    result: PBOResult = cscv_pbo(_noise_matrix(0, 240, 10), n_blocks=8)
    d = result.as_dict()
    assert isinstance(d["pbo"], float)
    assert all(not isinstance(v, np.ndarray) for v in d.values())


def test_engine_runner_statistics_runs_without_database() -> None:
    """EngineRunner.statistics 只用 calculate_statistics，不碰数据库 ——
    因此拼接出来的样本外曲线能拿到与普通回测【完全同一套】指标（含 RAR/R³）。"""
    runner = EngineRunner(
        strategy_class=None,                     # type: ignore[arg-type]
        vt_symbol="700.SEHK", interval=Interval.DAILY,
        rate=0.0013, slippage=0.0, size=1, pricetick=0.01, capital=int(CAPITAL),
        start=datetime(2024, 1, 1), end=datetime(2026, 7, 1), annual_days=ANNUAL_DAYS,
    )
    rng = np.random.default_rng(21)
    df = DataFrame(
        {
            "net_pnl": rng.normal(500, 8_000, size=200),
            "commission": np.zeros(200), "slippage": np.zeros(200),
            "turnover": np.zeros(200), "trade_count": np.zeros(200),
        },
        index=[d.date() for d in _dates(200)],
    )
    stats = runner.statistics(df)
    assert "sharpe_ratio" in stats
    assert "r_cubed" in stats                    # fork 加的稳健指标也在
    assert stats["total_days"] == 200
