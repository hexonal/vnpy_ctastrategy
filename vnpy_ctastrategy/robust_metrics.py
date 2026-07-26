"""稳健性业绩评价指标：RAR / R-Cubed / Robust Sharpe。

来源：VeighNa 社区帖 https://www.vnpy.com/forum/topic/32894
本模块严格按帖子定义实现，作为独立文件存在，不改上游任何函数体 —— 便于同步上游。

三个指标的共同出发点：常规年化收益只看首尾净值（`(end/start-1)/days*annual_days`），
一条"最后几天暴涨拉起来"的曲线和一条稳步上行的曲线可以给出完全相同的年化数字。
这三个指标改为对【累计收益曲线本身】做回归，因而对曲线形状敏感。

    RAR（Regressed Annual Return，回归年化收益）
        对累计收益百分比序列做【过原点】线性回归 y = a·x（无常数项），
        x 为 1..n 的时间序号，y 为累计收益百分比；RAR = a × annual_days。

    R-Cubed（简化版）
        R³ = RAR / 前 N 大回撤幅度的平均值（默认 N=5）。
        帖子作者去掉了原始公式中的回撤周期项，称简化版表现更好。

    Robust Sharpe（稳健夏普）
        RobustSharpe = RAR / (日收益标准差 × √annual_days)，
        即把常规夏普的分子从"简单年化收益"换成 RAR。

━━━ 使用前必读：这三个指标【不是】统计检验 ━━━

它们都是样本内描述统计量。RAR 高不代表策略在样本外成立，也不回答"这是不是运气"。
真正回答该问题的是 Deflated Sharpe Ratio、block permutation p 值、PBO 等，本模块不提供。

另有一个必须知道的结构性性质（本模块 `rar_sample_weights()` 可复现）：
过原点回归的斜率 a = Σ(x_i·y_i)/Σ(x_i²)，展开到【单期收益】r_j 上后，
第 j 期的隐含权重正比于 (n² − j²)，即随时间【单调递减到接近零】。
n=250 时：第 1 期权重约为等权的 1.50×，第 145 期(≈58%处)与等权相等，第 250 期仅约 0.012×。

含义：RAR 系统性地低估最近期的表现。用于"样本外近期是否仍然有效"这类判断时，
它恰好把最有决策价值的近期数据权重压到接近零。因此本模块的建议用法是：
把它当作【曲线形状的诊断量】与常规年化收益【并列】阅读，
不要单独用它下结论，更不要把它当作参数寻优的目标函数。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

DEFAULT_TOP_N_DRAWDOWNS: int = 5


@dataclass(frozen=True)
class RobustMetrics:
    """三个指标 + 计算它们所依据的中间量（便于核对，不必反查）。"""

    regressed_annual_return: float      # RAR
    r_cubed: float                      # R³ = RAR / 前 N 大回撤均值
    robust_sharpe: float                # RAR / (日收益std × √annual_days)

    regression_slope: float             # 过原点回归的斜率（每期累计收益百分点）
    avg_top_drawdown: float             # 前 N 大回撤幅度的平均值（正数，百分点）
    drawdown_episode_count: int         # 实际识别出的独立回撤段数
    return_std: float                   # 日收益标准差（百分点）
    annual_days: int
    sample_size: int                    # 参与回归的期数 n

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def cumulative_return_curve(balance: np.ndarray, capital: float) -> np.ndarray:
    """净值序列 → 累计收益百分比序列（相对初始资金）。

    这是 RAR 回归的 y 轴。用简单收益 (balance/capital - 1) * 100 而非对数收益，
    与 vnpy 的 total_return 口径一致，两者可直接对比。
    """
    if capital <= 0:
        raise ValueError(f"capital 必须为正，收到 {capital}")
    return (np.asarray(balance, dtype=float) / capital - 1.0) * 100.0


def regressed_annual_return(
    cumulative_pct: np.ndarray, annual_days: int
) -> tuple[float, float]:
    """帖子的 RAR：对累计收益曲线做过原点线性回归，斜率 × annual_days。

    返回 (RAR, slope)。

    过原点（无截距）是帖子明确指定的：y = a·x。这与"曲线从 0 开始"的事实一致 ——
    第 0 期累计收益必然为 0，强制截距为 0 才不会让回归去拟合一个虚假的起点偏移。
    最小二乘解：a = Σ(x_i·y_i) / Σ(x_i²)。
    """
    y = np.asarray(cumulative_pct, dtype=float)
    n = y.size
    if n == 0:
        return 0.0, 0.0
    x = np.arange(1, n + 1, dtype=float)
    denominator = float(np.sum(x * x))
    if denominator == 0:
        return 0.0, 0.0
    slope = float(np.sum(x * y) / denominator)
    return slope * annual_days, slope


def drawdown_episodes(balance: np.ndarray) -> list[float]:
    """把净值曲线切成【独立的回撤段】，返回每段的最大回撤幅度（正数，百分点）。

    一段回撤定义为：从一个净值新高开始，到下一次创出新高为止。取该段内的最深跌幅。
    这样切分是必要的 —— 若直接对逐日回撤序列取"最大的 5 个值"，它们几乎总是落在
    同一次大回撤的谷底附近，五个数其实是同一段回撤，平均下来就退化成最大回撤本身，
    R³ 也就退化成 return_drawdown_ratio。

    仍在回撤中（尚未收复）的最后一段也计入，否则一条正在深度回撤中的曲线
    反而会因为"还没结束"而不被计分。
    """
    values = np.asarray(balance, dtype=float)
    if values.size == 0:
        return []

    episodes: list[float] = []
    peak = values[0]
    trough = values[0]

    for value in values[1:]:
        if value >= peak:
            # 创出新高：结算上一段
            if trough < peak:
                episodes.append((peak - trough) / peak * 100.0)
            peak = value
            trough = value
        elif value < trough:
            trough = value

    if trough < peak:                       # 收尾：仍未收复的一段
        episodes.append((peak - trough) / peak * 100.0)
    return episodes


def average_top_drawdowns(
    balance: np.ndarray, top_n: int = DEFAULT_TOP_N_DRAWDOWNS
) -> tuple[float, int]:
    """前 top_n 大回撤段的平均幅度（正数，百分点），以及识别出的总段数。

    段数不足 top_n 时对现有段取平均（而不是补零 —— 补零会凭空压低分母、
    虚高 R³）。段数是判断 R³ 可信度的关键：只有 1-2 段时这个"平均"没有统计意义，
    调用方应据此决定要不要采信，故一并返回。
    """
    episodes = drawdown_episodes(balance)
    if not episodes:
        return 0.0, 0
    top = sorted(episodes, reverse=True)[:top_n]
    return float(np.mean(top)), len(episodes)


def calculate_robust_metrics(
    balance: np.ndarray,
    daily_returns: np.ndarray,
    capital: float,
    annual_days: int,
    top_n_drawdowns: int = DEFAULT_TOP_N_DRAWDOWNS,
) -> RobustMetrics:
    """一次算齐三个指标。

    balance        每日净值序列
    daily_returns  每日收益率序列（小数，非百分比；与 vnpy daily_df["return"] 同口径）
    capital        初始资金
    annual_days    年化交易日数（港股约 247、美股 252 —— 由调用方按市场传入）

    分母为 0 的情形一律返回 0.0，与 vnpy 既有统计的处理方式保持一致
    （见 backtesting.py 里 sharpe_ratio / return_drawdown_ratio 的 else 分支）。
    """
    balance = np.asarray(balance, dtype=float)
    returns = np.asarray(daily_returns, dtype=float)

    cumulative = cumulative_return_curve(balance, capital)
    rar, slope = regressed_annual_return(cumulative, annual_days)

    avg_drawdown, episode_count = average_top_drawdowns(balance, top_n_drawdowns)
    r_cubed = rar / avg_drawdown if avg_drawdown else 0.0

    return_std = float(np.std(returns, ddof=1) * 100.0) if returns.size > 1 else 0.0
    annualized_std = return_std * float(np.sqrt(annual_days))
    robust_sharpe = rar / annualized_std if annualized_std else 0.0

    return RobustMetrics(
        regressed_annual_return=rar,
        r_cubed=r_cubed,
        robust_sharpe=robust_sharpe,
        regression_slope=slope,
        avg_top_drawdown=avg_drawdown,
        drawdown_episode_count=episode_count,
        return_std=return_std,
        annual_days=annual_days,
        sample_size=int(cumulative.size),
    )


def rar_sample_weights(n: int) -> np.ndarray:
    """RAR 隐含在每一期【单期收益】上的权重，已归一化到均值 1（等权 = 1.0）。

    诊断用，不参与指标计算。把过原点回归的斜率展开到单期收益 r_j 上可得
    权重正比于 (n² − j²)：越靠后的样本权重越低，末期趋近于 0。
    提供此函数是为了让"RAR 低估近期表现"这一性质可被直接验证，而不是只写在文档里。
    """
    if n <= 0:
        return np.array([], dtype=float)
    j = np.arange(1, n + 1, dtype=float)
    weights = (n * n - j * j)
    total = weights.sum()
    if total == 0:
        return np.zeros(n, dtype=float)
    return weights / total * n
